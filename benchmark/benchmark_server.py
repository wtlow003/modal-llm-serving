"""Benchmark for online serving performance.

NOTE: This benchmark script is heavily referenced from vLLM's existing benchmark script. In specific,

    1. https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
    2. https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py

Additional enhancement(s) were made to benchmark static batching, multiple benchmark runs, metrics display etc.

To run the benchmark script, run the following command:

    [vLLM]
    python benchmark/benchmark_server.py --backend vllm \
        --model "mistralai--mistral-7b-instruct" \
        --num-request 1000 \
        --request-rate 64 \
        --num-benchmark-runs 3 \
        --max-input-len 1024 \
        --max-output-len 1024 \
        --base-url "https://xxx--vllm-mistralai--mistral-7b-instruct-v02-serve.modal.run"


    [TGI]
    python benchmark/benchmark_server.py --backend tgi \
        --model "mistralai/Mistral-7B-Instruct" \
        --num-request 1000 \
        --request-rate 64 \
        --num-benchmark-runs 3 \
        --max-input-len 1024 \
        --max-output-len 1024 \
        --base-url "https://xxx--tgi-mistralai--mistral-7b-instruct-v02-serve.modal.run"

    [LMDEPLOY]
    python benchmark/benchmark_server.py --backend lmdeploy \
        --model "mistralai--mistral-7b-instruct" \
        --num-request 1000 \
        --request-rate 64 \
        --num-benchmark-runs 3 \
        --max-input-len 1024 \
        --max-output-len 1024 \
        --base-url "https://xxx--lmdeploy-mistralai--mistral-7b-instruct-v02-serve.modal.run"
"""

import argparse
import asyncio
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Union

import aiohttp
import loguru
import numpy as np
from dataset import sample_dolly_dataset
from metrics import calculate_metrics, generate_metrics_display
from request_funcs import (
    ASYNC_REQUEST_FUNC,
    RequestFuncInput,
    RequestFuncOutput,
    get_request,
)
from tqdm.asyncio import tqdm

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import (  # noqa: E402
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

##############################################
############## UTILS FUNCTIONS ###############
##############################################


def setup_logging() -> loguru.logger:  # type: ignore
    logger = loguru.logger
    logger.remove()

    # customise logging format
    logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stdout, format=logger_format)

    return logger


##############################################
################ CONSTANTS ###################
##############################################

LOGGER = setup_logging()
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


async def benchmark(
    backend: str,
    api_url: str,
    model: str,
    dataset: List[List[Union[List[Dict[str, str]], int]]],
    request_rate: float,
    session: aiohttp.ClientSession,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    logger,
    enable_tqdm: bool,
):
    """_summary_

    Args:
        api_url (str): _description_
        model (str): _description_
        input_requests (List[List[Dict[str, str]]]): _description_
        request_rate (float): _description_
        session (aiohttp.ClientSession): _description_
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): _description_
        disable_tqdm (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    request_func: Callable = ASYNC_REQUEST_FUNC[backend]

    test_dataset = dataset[0]
    test_input = RequestFuncInput(
        api_url=api_url,
        model=model,
        messages=test_dataset[0],  # type: ignore
        output_len=test_dataset[1],  # type: ignore
    )
    logger.info(f"Initial test prompt: {test_input}")
    test_output = await request_func(session=session, request_func_input=test_input)
    if not test_output.success:
        logger.error(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        logger.info("Initial test run completed. Starting main benchmark run...")

    pbar: Optional[tqdm] = tqdm(total=len(dataset)) if enable_tqdm else None

    benchmark_start_ts = time.perf_counter()
    tasks = []
    async for request in get_request(dataset, request_rate):
        request_func_input = RequestFuncInput(
            api_url=api_url,
            model=model,
            messages=request[0],  # type: ignore
            output_len=request[1],  # type: ignore
        )
        tasks.append(
            asyncio.create_task(
                request_func(
                    session=session, request_func_input=request_func_input, pbar=pbar
                )
            )
        )

    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if enable_tqdm:
        pbar.close()  # type: ignore

    benchmark_duration = time.perf_counter() - benchmark_start_ts

    metrics = calculate_metrics(
        outputs=outputs, duration=benchmark_duration, tokenizer=tokenizer, logger=LOGGER
    )
    return metrics


async def main(args: argparse.Namespace, logger):
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """
    logger.info(f"Benchmark args: {args}")
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = sample_dolly_dataset(
        num_prompts=args.num_requests,
        dolly_category="summarization",
        tokenizer=tokenizer,
        seed=args.seed,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        logger=logger,
    )
    logger.info(f"Prepared {len(dataset)} prompts")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        run_metrics = []
        for i in range(args.num_benchmark_runs):
            logger.info(f"Starting benchmark run: {i+1}")
            metrics = await benchmark(
                backend=args.backend,
                api_url=args.base_url + args.endpoint,
                model=args.model,
                dataset=dataset,
                request_rate=args.request_rate,
                session=session,
                tokenizer=tokenizer,
                enable_tqdm=args.enable_tqdm,
                logger=logger,
            )
            run_metrics.append(metrics)

    # display benchmark metrics across all runs
    generate_metrics_display(run_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "tgi", "lmdeploy", "tensorrt-llm"],
        help="Serving framework to benchmark.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of request to be made.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=float,
        help="Max concurrent requests.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--num-benchmark-runs",
        type=int,
        default=3,
        help="Number of benchmark runs to completed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=1024,
        help="Input length for each request.",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=1024,
        help="Output length for each request.",
    )
    args = parser.parse_args()
    asyncio.run(main(args=args, logger=LOGGER))
