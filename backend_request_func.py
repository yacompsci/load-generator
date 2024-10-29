import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
from tqdm.asyncio import tqdm
from metrics import Metrics

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    metrics: Optional[Metrics] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if metrics is not None:
                    # increment total request count
                    metrics.total_request_count.labels(**metrics.labels).inc()
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    if metrics is not None:
                        # update metrics
                        metrics.successful_request_count.labels(**metrics.labels).inc()

                        metrics.input_tokens.labels(**metrics.labels).inc(
                            output.prompt_len
                        )
                        metrics.generated_tokens.labels(**metrics.labels).inc(
                            len(output.itl) + 1
                        )

                        metrics.ttft.labels(**metrics.labels).observe(
                            output.ttft
                        )
                        if len(output.itl) != 0:
                            metrics.tpot.labels(**metrics.labels).observe(
                                (output.latency - output.ttft) / (len(output.itl))
                            )
                        for itl in output.itl:
                            metrics.itl.labels(**metrics.labels).observe(
                                itl
                            )
                else:
                    output.error = response.reason or ""
                    output.success = False
                    if metrics is not None:
                        # upadate metrics
                        metrics.failed_request_count.labels(**metrics.labels).inc()
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            print(Exception)

    if pbar:
        pbar.update(1)
    return output


# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
