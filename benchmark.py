import os
import random
import json
import asyncio
import time
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pydantic import BaseModel
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dataclasses import dataclass

from metrics import Metrics
from backend_request_func import (
    RequestFuncInput,
    RequestFuncOutput, 
    async_request_openai_chat_completions
)

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]

class LoadSpec(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    dataset_path: str = "./ShareGPT_V3_unfiltered_cleaned_split.json"
    dataset_name: str = "sharegpt"
    model: str = "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8"
    tokenizer: Optional[str] = None
    backend: str = "openai-chat"
    endpoint: str = "/v1/chat/completions"
    num_prompts: int = 1000
    request_rate: float = 10.0
    seed: int = 42
    trust_remote_code: bool = False
    random_input_len: int = 256
    random_output_len: int = 256
    random_range_ratio: float = 0.7
    other_args: Optional[str] = None 
    save_result: bool = False
    sonnet_input_len: int = 512
    sonnet_output_len: int = 16
    sonnet_prefix_len: int = 386


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset

def sample_random_requests(
        input_len: int, output_len: int, num_prompts: int, range_ratio: float,
        tokenizer: PreTrainedTokenizerBase) -> List[Tuple[str, int, int]]:

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode([(offsets[i] + i + j) % tokenizer.vocab_size
                                   for j in range(input_lens[i])])
        input_requests.append(
            (prompt, int(input_lens[i]), int(output_lens[i])))

    return input_requests

def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(
            prefix_lines +
            random.sample(poem_lines, num_input_lines - num_prefix_lines))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len))

    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    failed = 0
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)
            failed += 1
    print(f"Failed requests: {failed}")
    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.median(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.mean(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens


async def benchmark(
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    prometheus_metrics: Metrics,
):
    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len = input_requests[0]
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        best_of=best_of,
        use_beam_search=use_beam_search,
    )
    test_output = await async_request_openai_chat_completions(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # initialize prometheus metrics
    prometheus_metrics.intitialize_metrics()

    # start benchmark
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                async_request_openai_chat_completions(
                    request_func_input=request_func_input,
                    pbar=pbar,
                    metrics=prometheus_metrics,
                )
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
    )
    
    # update prometheus metrics
    prometheus_metrics.bench_duration.labels(
        **prometheus_metrics.labels
    ).set(benchmark_duration)

    prometheus_metrics.request_throughput.labels(
        **prometheus_metrics.labels
    ).set(metrics.request_throughput)

    prometheus_metrics.output_token_throughput.labels(
        **prometheus_metrics.labels
    ).set(metrics.output_throughput)

    prometheus_metrics.total_token_throughput.labels(
        **prometheus_metrics.labels
    ).set(metrics.total_token_throughput)

    # ttft
    prometheus_metrics.ttft_mean.labels(
        **prometheus_metrics.labels
    ).set(metrics.mean_ttft_ms/1000)
    prometheus_metrics.ttft_median.labels(
        **prometheus_metrics.labels
    ).set(metrics.median_ttft_ms/1000)

    for p, val in metrics.percentiles_ttft_ms:
        if p==99:
             prometheus_metrics.ttft_p99.labels(
                **prometheus_metrics.labels
            ).set(val/1000)
             
    # tpot
    prometheus_metrics.tpot_mean.labels(
        **prometheus_metrics.labels
    ).set(metrics.mean_tpot_ms/1000)
    prometheus_metrics.tpot_median.labels(
        **prometheus_metrics.labels
    ).set(metrics.median_tpot_ms/1000)

    for p, val in metrics.percentiles_tpot_ms:
        if p==99:
             prometheus_metrics.tpot_p99.labels(
                **prometheus_metrics.labels
            ).set(val/1000)
             
    # itl
    prometheus_metrics.itl_mean.labels(
        **prometheus_metrics.labels
    ).set(metrics.mean_itl_ms/1000)
    prometheus_metrics.itl_median.labels(
        **prometheus_metrics.labels
    ).set(metrics.median_itl_ms/1000)

    for p, val in metrics.percentiles_itl_ms:
        if p==99:
             prometheus_metrics.itl_p99.labels(
                **prometheus_metrics.labels
            ).set(val/1000)

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        # "generated_texts": [output.generated_text for output in outputs],
        # "errors": [output.error for output in outputs],
    }

    # update prometheus metrics


    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function print and add statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


async def start_benchmark(spec: LoadSpec, prometheus_metrics: Metrics):
    print(spec)
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    
    percentile_metrics = "ttft,tpot,itl"
    metric_percentiles = "99"
    backend = "openai-chat"
    result_dir = None

    model_id = spec.model
    tokenizer_id = spec.tokenizer if spec.tokenizer is not None else spec.model

    
    api_url = f"http://{spec.host}:{spec.port}{spec.endpoint}"
    base_url = f"http://{spec.host}:{spec.port}"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=spec.trust_remote_code
    )

    if spec.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=spec.dataset_path,
            num_requests=spec.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=None,
        )
    
    elif spec.dataset_name == "sonnet":
        
        input_requests = sample_sonnet_requests(
            dataset_path=spec.dataset_path,
            num_requests=spec.num_prompts,
            input_len=spec.sonnet_input_len,
            output_len=spec.sonnet_output_len,
            prefix_len=spec.sonnet_prefix_len,
            tokenizer=tokenizer,
        )
        input_requests = [(prompt, prompt_len, output_len)
                            for prompt, prompt_formatted, prompt_len,
                            output_len in input_requests]

    elif spec.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=spec.random_input_len,
            output_len=spec.random_output_len,
            num_prompts=spec.num_prompts,
            range_ratio=spec.random_range_ratio,
            tokenizer=tokenizer,
        )

    else:
        raise ValueError(f"Unknown dataset: {spec.dataset_name}")

    print(f"prepared inputs: {len(input_requests)}")
    print(f"Example: {input_requests[42]}")

    benchmark_result = await benchmark(
        api_url=api_url,
        model_id=model_id,
        tokenizer=tokenizer,
        input_requests=input_requests,
        best_of=1,
        use_beam_search=False,
        request_rate=spec.request_rate,
        disable_tqdm=False,
        selected_percentile_metrics=percentile_metrics.split(","),
        selected_percentiles=[
            float(p) for p in metric_percentiles.split(",")
        ],
        prometheus_metrics=prometheus_metrics,
    )

    # Save config and results to json
    result_json: Dict[str, Any] = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["backend"] = backend
    result_json["model_id"] = model_id
    result_json["tokenizer_id"] = tokenizer_id
    result_json["best_of"] = 1
    result_json["use_beam_search"] = False
    result_json["num_prompts"] = spec.num_prompts


    # Traffic
    result_json["request_rate"] = (
        spec.request_rate if spec.request_rate < float("inf") else "inf")

    # Merge with benchmark result
    result_json = {**result_json, **benchmark_result}

    # Save to file
    base_model_id = model_id.split("/")[-1]
    file_name = f"{backend}-{spec.request_rate}qps-{base_model_id}-{current_dt}.json"  #noqa
    if result_dir:
        file_name = os.path.join(result_dir, file_name)
    with open(file_name, "w") as outfile:
        json.dump(result_json, outfile)

    # return benchmark_result
    return result_json