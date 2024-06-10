import asyncio
import json
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Dict, List, Optional, Union

import aiohttp
from tqdm.asyncio import tqdm


@dataclass
class RequestFuncInput:
    api_url: str = ""
    model: str = ""
    messages: List[Dict[str, str]] = field(default_factory=(List[Dict[str, str]]))
    output_len: int = 2046
    best_of: int = 1
    temperature: float = 0.0
    stream: bool = True


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    error: str = ""


async def async_request_openai_api_chat_completions(
    session: aiohttp.ClientSession,
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """_summary_

    Args:
        request_func_input (_type_): _description_
        pbar (Optional[tqdm], optional): _description_. Defaults to None.

    Returns:
        RequestFuncOutput: _description_
    """
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "v1/chat/completions"
    ), "vLLM Chat Completions API URL must end with 'v1/chat/completions'"

    payload = {
        "model": request_func_input.model,
        "messages": request_func_input.messages,
        "temperature": request_func_input.temperature,
        "max_tokens": request_func_input.output_len,
        "stream": request_func_input.stream,
    }
    # headers = {"Authorization": f"Bearer {api_key}"}

    output = RequestFuncOutput()

    generated_text = ""
    ttft = 0.0
    start_ts = time.perf_counter()
    most_recent_ts = start_ts
    latency = 0.0
    try:
        async with session.post(url=api_url, json=payload) as response:
            if response.status == 200:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    chunk = chunk_bytes.decode("utf-8")[len("data: ") :]
                    # last chunk indicator for some api setup
                    if chunk == "[DONE]":
                        # overall time from start of request to end of request
                        latency = time.perf_counter() - start_ts
                    # normal generation process of token
                    else:
                        token_ts = time.perf_counter()
                        data = json.loads(chunk)
                        delta = data["choices"][0]["delta"]
                        if delta.get("content", None):
                            # first token
                            if ttft == 0.0:
                                ttft = token_ts - start_ts
                                output.ttft = ttft

                            # decoding process
                            # ref: https://github.com/vllm-project/vllm/blob/5d7e3d0176e0dbcf144c64b7d14d996c55e36c50/benchmarks/backend_request_func.py#L268-L274
                            else:
                                output.itl.append(token_ts - most_recent_ts)

                            generated_text += delta["content"]
                            # treat latest processed token as last time marker
                        most_recent_ts = token_ts

                # iteration through all chunks
                output.latency = (
                    latency if latency else time.perf_counter() - start_ts
                )  # if chunk api does not return "END" in last chunk
                output.generated_text = generated_text
                output.success = True

    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def poisson_interarrival_generator(rate):
    """Generator to yield Poisson-distributed inter-arrival times.
    Args:
        rate (_type_): _description_
    """
    while True:
        # Generate a uniform random number between 0 and 1
        U = random.random()
        # Calculate the inter-arrival time using the inverse transform method
        T = -math.log(U) / rate
        yield T


async def get_request(
    dataset: List[List[Union[List[Dict[str, str]], int]]],
    request_rate: float,
) -> AsyncGenerator[List[List[Dict[str, str]] | int], None]:
    """_summary_

    Args:
        input_request (_type_): _description_
        request_rate (float): _description_

    Returns:
        AsyncGenerator: _description_

    Yields:
        Iterator[AsyncGenerator]: _description_
    """
    poisson_generator = poisson_interarrival_generator(request_rate)
    for request in iter(dataset):
        yield request

        if request_rate == float("inf"):
            continue

        interval = next(poisson_generator)
        await asyncio.sleep(interval / 1000.0)


ASYNC_REQUEST_FUNC: Dict[str, Callable] = {
    "vllm": async_request_openai_api_chat_completions,
    "tgi": async_request_openai_api_chat_completions,
}
