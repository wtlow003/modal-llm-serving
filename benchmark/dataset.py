import os
from typing import Dict, List, Union

import datasets
import loguru

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import (  # noqa: E402
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def sample_dolly_dataset(
    num_prompts: int,
    dolly_category: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    seed: int,
    logger: loguru.logger,  # type: ignore
    max_input_len: int = 1024,
    max_output_len: int = 1024,
) -> List[List[Union[List[Dict[str, str]], int]]]:
    """Sample benchmark prompts and guided output length from "databricks/databricks-dolly-15k" dataset.

    Args:
        num_prompts (int): _description_
        dolly_category (str): _description_
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): _description_
        seed (int): _description_
        logger (loguru.logger): _description_
        max_output_len (int, optional): _description_. Defaults to 1024.

    Raises:
        ValueError: _description_

    Returns:
        List[List[Union[List[Dict[str, str]], int]]]: _description_
    """
    dataset = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = dataset.filter(lambda row: row["category"] == dolly_category)

    # shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    processed_dataset: List[List[Union[List[Dict[str, str]], int]]] = []
    total = 0

    while total < num_prompts:
        for row in dataset:
            prompt = ""
            if row["context"]:  # type: ignore
                prompt += f"{row['instruction']}\nContext:\n{row['context']}"  # type: ignore
            else:
                prompt += f"{row['instruction']}"  # type: ignore

            # check input len
            input_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)  # type: ignore
            if input_len > max_input_len:
                continue

            # retrieve guided output len given the example response
            output_len = len(
                tokenizer(row["response"], add_special_tokens=False).input_ids  # type: ignore
            )
            if output_len > max_output_len:
                continue

            processed_dataset.append(
                [[{"role": "user", "content": prompt}], output_len]
            )
            total += 1

            if total == num_prompts:
                break

        if total < num_prompts:
            logger.error(
                f"Unable to fulfill requirement for {num_prompts} prompts. Please lower the total prompts requirements"
            )
            raise ValueError(
                f"Unable to fulfill requirement for {num_prompts} prompts. Please lower the total prompts requirements"
            )

    return processed_dataset
