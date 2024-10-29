from typing import Optional, Dict, Any, List, Literal
from fastapi import FastAPI, HTTPException, BackgroundTasks
from prometheus_client import make_asgi_app

from on_startup import run_benchmarks
from metrics import Metrics
from utils import sanitize_label
from benchmark import start_benchmark, LoadSpec

app = FastAPI()

# Global variables to keep track of the benchmark
benchmark_status: Literal["not started", "running", "finished"] = "not started"
benchmark_result: Optional[Dict[str, Any]] = None

# Prometheus metrics
metrics = Metrics(
    labelnames=[
        "dataset",
        "model",
    ]
)

@app.on_event("startup")
async def startup_event():
    global metrics
    await run_benchmarks(metrics)

app.mount("/metrics", make_asgi_app())

async def background_benchmark(load_spec, metrics):
    global benchmark_result, benchmark_status
    try:
        benchmark_status = "running"
        benchmark_result = await start_benchmark(load_spec, metrics)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        benchmark_result = {"error": str(e)}
    finally:
        benchmark_status = "finished"

@app.post("/start-load")
async def start_load(load_spec: LoadSpec, background_tasks: BackgroundTasks):
    global benchmark_status, benchmark_result, metrics

    if benchmark_status == "running":
        raise HTTPException(status_code=400, detail="Load generation is already running")

    # Reset result
    benchmark_result = None

    metrics.labels = {
        "dataset": sanitize_label(load_spec.dataset_name),
        "model": sanitize_label(load_spec.model),
    }
    background_tasks.add_task(background_benchmark, load_spec, metrics)

    return {"status": "Load generation started"}

@app.get("/load-status")
def load_status():
    global benchmark_status
    match benchmark_status:
        case "running":
            return {
                "status": "Load generation is running"
            }
        case "not started":
            return {
                "status": "Ready to run benchmark"
            }
        case "finished":
            return {
                "status": "Load generation finished"
            }

@app.get("/load-result")
def load_result():
    global benchmark_result
    if benchmark_result is not None:
        return benchmark_result
    else:
        raise HTTPException(status_code=400, detail="Benchmark result is not available yet")
    