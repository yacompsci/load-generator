import os
import traceback

import requests
import socket
import time
from datetime import datetime
from typing import Dict

from benchmark import start_benchmark, LoadSpec
from utils import (
    sanitize_label, 
    save_result, 
    get_instance_name,
    get_specs,
)
from metrics import Metrics

SERVED_MODEL = os.environ.get("SERVED_MODEL")
SPECS = [
    {
        "host": "localhost",
        "port": 8000,
        "num_prompts": 100,
        "request_rate": 6,
        "model": SERVED_MODEL,
        "backend": "openai-chat",
        "endpoint": "/v1/chat/completions",
        "dataset_path": "app/ShareGPT_V3_unfiltered_cleaned_split.json",
        "dataset_name": "sharegpt"
    },  
    {
        "host": "localhost",
        "port": 8000,
        "num_prompts": 100,
        "request_rate": 12,
        "model": SERVED_MODEL,
        "backend": "openai-chat",
        "endpoint": "/v1/chat/completions",
        "dataset_path": "app/sonnet.txt",
        "dataset_name": "sonnet",
        "sonnet_input_len": 512,
        "sonnet_prefix_len": 128,
        "sonnet_output_len": 16
    },  
]


VLLM_SERVER_URL = "http://localhost:8000/health"
RETRY_INTERVAL = 5

def check_vllm_health():
    while True:
        try:
            response = requests.get(VLLM_SERVER_URL, timeout=5)
            if response.status_code == 200:
                return
            else:
                print(f"Server returned status code {response.status_code}. Retrying...")
                
        except requests.exceptions.RequestException as e:
            print(f"Could not connect to the remote server: {e}. Retrying in {RETRY_INTERVAL} seconds...")
        
        time.sleep(RETRY_INTERVAL)

async def run_benchmarks(metrics: Metrics):
    
    check_vllm_health()
    print("Running benchmark post-startup")
    
    instance_name = get_instance_name()

    async def run_benchmark(spec_dict: Dict[str, str]):
        spec = LoadSpec(**spec_dict)
        metrics.labels = {
            "dataset": sanitize_label(spec.dataset_name),
            "model": sanitize_label(spec.model),
        }
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
        result_file_name = f"{metrics.labels['dataset']}_{metrics.labels['model']}_{instance_name}_{timestamp}"
       
        try:
            result = await start_benchmark(spec, metrics)
            result["timestamp"] = timestamp
            result["instance_name"] = instance_name
            save_result(
                result_dict=result,
                file_name=result_file_name,
            )
        except Exception as e:
            print(f"Error running benchmark: {e}")
            traceback.print_exc()

    specs = get_specs() or SPECS
    for spec in specs:
        spec["model"] = SERVED_MODEL
        print(f"starting {spec['dataset_name']} benchmark...")
        await run_benchmark(spec)
        # threading.Thread(target=run_benchmark, args=[spec]).start()
