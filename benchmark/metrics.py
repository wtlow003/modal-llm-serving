from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from rich import box
from rich.align import Align
from rich.console import Console
from rich.table import Table


@dataclass
class BenchmarkMetrics:
    completed: int
    failed: int
    req_throughput: float
    token_throughput: float
    mean_itl_ms: np.floating
    median_itl_ms: np.floating
    p95_itl_ms: np.floating
    p99_itl_ms: np.floating
    mean_ttft_ms: np.floating
    median_ttft_ms: np.floating
    p99_ttft_ms: np.floating
    mean_tpot_ms: np.floating
    median_tpot_ms: np.floating
    p99_tpot_ms: np.floating


def calculate_metrics(outputs, duration, tokenizer, logger):
    """_summary_

    Args:
        outputs (_type_): _description_
        duration (_type_): _description_
        tokenizer (_type_): _description_

    Returns:
        _type_: _description_
    """
    actual_output_lens = []
    completed = 0
    failed = 0
    itls = []
    tpots = []
    ttfts = []

    for i in range(len(outputs)):
        if outputs[i].success:
            completed += 1
            output_len = len(
                tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids
            )
            actual_output_lens.append(output_len)
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
        else:
            failed += 1
            actual_output_lens.append(0)

    if completed == 0:
        logger.warning(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    metrics = BenchmarkMetrics(
        completed=completed,
        failed=failed,
        req_throughput=(completed / duration) * 60,
        token_throughput=sum(actual_output_lens) / duration,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        p95_itl_ms=np.percentile(itls or 0, 95) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics


def generate_metrics_display(metrics: List[BenchmarkMetrics]):
    """_summary_

    Args:
        metrics (_type_): _description_

    Returns:
        _type_: _description_
    """

    ROW_NAMES_MAPPING: Dict[str, str] = {
        "completed": "Completed Requests",
        "failed": "Failed Requests",
        "req_throughput": "Request Throughput (req/min)",
        "token_throughput": "Token Throughput (token/s)",
        "mean_ttft_ms": "Mean Time to First Token (ms)",
        "median_ttft_ms": "Median Time to First Token (ms)",
        "p99_ttft_ms": "p99 Time to First Token (ms)",
        "mean_tpot_ms": "Mean Time Per Output Token (ms)",
        "median_tpot_ms": "Median Time Per Output Token (ms)",
        "p99_tpot_ms": "p99 Time Per Output Token (ms)",
        "mean_itl_ms": "Mean Inter-Token Latency (ms)",
        "median_itl_ms": "Median Inter-Token Latency (ms)",
        "p95_itl_ms": "p95 Inter-Token Latency (ms)",
        "p99_itl_ms": "p99 Inter-Token Latency (ms)",
    }

    # setup table
    table = Table(
        title="Benchmark Metrics",
        style="spring_green3",
        box=box.DOUBLE,
        expand=True,
        highlight=True,
    )

    # defining columns
    table.add_column("", header_style="bold")
    for i in range(len(metrics)):
        table.add_column(f"Run {i+1}", header_style="bold")
    table.add_column("Avg", header_style="bold")

    # defining rows
    for field_name in BenchmarkMetrics.__annotations__:
        row_rendarables = []
        row_rendarables.append(f"[bold]{ROW_NAMES_MAPPING[field_name]}[/bold]")
        field_values = [getattr(metric, field_name) for metric in metrics]
        # compute average of all runs
        field_values.append(np.mean(field_values))
        row_rendarables.extend(
            [
                (
                    f"{val:.2f}"
                    if field_name not in ["completed", "failed"]
                    else f"{int(val)}"
                )
                for val in field_values
            ]
        )
        # add row for field name
        table.add_row(*row_rendarables)

    metrics_table = Align.center(table, vertical="middle")
    console = Console()
    console.print("\n", metrics_table)
