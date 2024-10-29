import os
import re
from typing import Dict, Any
import json
import requests
import socket
import boto3
from botocore.exceptions import ClientError

def sanitize_label(label_name: str) -> str:
    sanitized_label = re.sub(r'[^a-zA-Z0-9_]', '_', label_name)
    if sanitized_label[0].isdigit():
        sanitized_label = f"_{sanitized_label}"

    return sanitized_label

def save_result(result_dict: Dict[str, Any], file_name: str):
    s3_client = boto3.client(
        's3',
        endpoint_url=os.environ.get("AWS_S3_ENDPOINT"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION")
    )

    directory = os.environ.get("BENCHMARK_RESULTS_DIRECTORY", "debug-results")
    if not directory.endswith('/'):
        directory += '/'
    
    bucket_name = os.environ.get("BENCHMARK_RESULTS_BUCKET", "inference-benchmark-results")
    s3_key = f"{directory}{file_name}"
    json_data = json.dumps(result_dict)

    try:
        s3_client.put_object(
            Bucket=bucket_name, 
            Key=s3_key, 
            Body=json_data, 
            ContentType='application/json'
        )
        print(f"File uploaded to s3://{bucket_name}/{s3_key}")
    except ClientError as e:
        print(f"Failed to upload file: {e}")

def get_instance_name():
    k8s_host = os.getenv("KUBERNETES_SERVICE_HOST")
    k8s_port = os.getenv("KUBERNETES_SERVICE_PORT")
    namespace = os.getenv("JOB_NAMESPACE", "test-job")
    pod_name = socket.gethostname()
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    
    with open(token_path, 'r') as file:
        token = file.read()

    url = f"https://{k8s_host}:{k8s_port}/api/v1/namespaces/{namespace}/pods/{pod_name}"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers, verify=False)

    if response.status_code == 200:
        pod_info = response.json()
        node_name = pod_info["spec"]["nodeName"]
        print(f"Pod is running on node: {node_name}")
        
        return node_name
    else:
        raise Exception(f"Failed to retrieve pod details: {response.status_code}")

def get_specs():
    config_path = "/etc/config/config.json"

    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            try:
                specs = json.load(file)
                return specs
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    else:
        print(f"Specs file does not exist at path: {config_path}")
    
    return None
