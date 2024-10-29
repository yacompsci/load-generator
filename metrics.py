from typing import List
from prometheus_client import Counter, generate_latest, REGISTRY, Gauge, Summary, Histogram

class Metrics:
    def __init__(self, labelnames: List[str]):
        self.labels = None
        self.total_request_count = Counter(
            name="load_genenerator:total_request_count",
            documentation="Number of requests sent",
            labelnames=labelnames,
        )
        self.failed_request_count = Counter(
            name="load_genenerator:failed_request_count",
            documentation="Number of failed requests",
            labelnames=labelnames,
        )
        self.successful_request_count = Counter(
            name="load_genenerator:successful_request_count",
            documentation="Number of successful requests",
            labelnames=labelnames,
        )
        
        self.bench_duration = Gauge(
            name="load_genenerator:bench_duration",
            documentation="Benchmark duration in seconds",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.input_tokens = Counter(
            name="load_genenerator:input_tokens",
            documentation="Total number of input tokens",
            labelnames=labelnames,
        )
        self.generated_tokens = Counter(
            name="load_genenerator:generated_count",
            documentation="Total number of generated tokens",
            labelnames=labelnames,
        )

        self.request_throughput = Gauge(
            name="load_genenerator:request_throughput",
            documentation="Request throughput (requests per second)",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.output_token_throughput = Gauge(
            name="load_genenerator:output_token_throughput",
            documentation="Output token throughput (tokens per second)",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.total_token_throughput = Gauge(
            name="load_genenerator:total_token_throughput",
            documentation="Total token throughput (tokens per second)",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.ttft = Histogram(
            name="load_generator:ttft",
            documentation="Histogram of time to first token in seconds",
            labelnames=labelnames,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0,
            ],
        )
        self.tpot = Histogram(
            name="load_generator:tpot",
            documentation="Histogram of time per output token in seconds",
            labelnames=labelnames,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0,
            ],
        )
        self.itl = Histogram(
            name="load_generator:itl",
            documentation="Histogram of inter-token latency in seconds",
            labelnames=labelnames,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0,
            ],
        )

        self.ttft_mean = Gauge(
            name="load_genenerator:ttft_mean",
            documentation="ttft_mean",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.ttft_median = Gauge(
            name="load_genenerator:ttft_median",
            documentation="ttft_median",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.ttft_p99 = Gauge(
            name="load_genenerator:ttft_p99",
            documentation="ttft_p99",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.tpot_mean = Gauge(
            name="load_genenerator:tpot_mean",
            documentation="tpot_mean",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.tpot_median = Gauge(
            name="load_genenerator:tpot_median",
            documentation="tpot_median",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.tpot_p99 = Gauge(
            name="load_genenerator:tpot_p99",
            documentation="tpot_p99",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.itl_mean = Gauge(
            name="load_genenerator:itl_mean",
            documentation="itl_mean",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.itl_median = Gauge(
            name="load_genenerator:itl_median",
            documentation="itl_median",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.itl_p99 = Gauge(
            name="load_genenerator:itl_p99",
            documentation="itl_p99",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

    def intitialize_metrics(self):
        assert self.labels is not None, "Provide labels to instantiale metrics"

        # intialize prometheus metrics
        self.total_request_count.labels(
            **self.labels
        )
        self.failed_request_count.labels(
            **self.labels
        )
        self.successful_request_count.labels(
            **self.labels
        )

        self.bench_duration.labels(
            **self.labels
        )
        self.input_tokens.labels(
            **self.labels
        )
        self.generated_tokens.labels(
            **self.labels
        )

        self.request_throughput.labels(
            **self.labels
        )
        self.output_token_throughput.labels(
            **self.labels
        )
        self.total_token_throughput.labels(
            **self.labels
        )

        self.ttft.labels(
            **self.labels
        )
        self.tpot.labels(
            **self.labels
        )
        self.itl.labels(
            **self.labels
        )

        self.ttft_mean.labels(
            **self.labels
        )
        self.ttft_median.labels(
            **self.labels
        )
        self.ttft_p99.labels(
            **self.labels
        )

        self.tpot_mean.labels(
            **self.labels
        )
        self.tpot_median.labels(
            **self.labels
        )
        self.tpot_p99.labels(
            **self.labels
        )

        self.itl_mean.labels(
            **self.labels
        )
        self.itl_median.labels(
            **self.labels
        )
        self.itl_p99.labels(
            **self.labels
        )
