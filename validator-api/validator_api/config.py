import os
import json
from typing import List


def parse_proxies(proxy_list: List[str]) -> List[str]:
    transformed_proxies = []
    for proxy in proxy_list:
        proxy_ip, proxy_port, proxy_user, proxy_pass = proxy.split(':')
        transformed_proxies.append(f"http://{proxy_user}:{proxy_pass}@{proxy_ip}:{proxy_port}")
    return transformed_proxies


PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ["HF_REPO"]
REPO_TYPE = "dataset"
TOPICS_LIST = json.loads(os.environ["TOPICS_LIST"])
PROXY_LIST = parse_proxies(json.loads(os.environ["PROXY_LIST"]))
IS_PROD = os.environ.get("IS_PROD", "false").lower() == "true"
CHECK_PROBABILITY = float(os.environ.get("CHECK_PROBABILITY", 0.1))
UPLOAD_BATCH_SIZE = int(os.environ.get("UPLOAD_BATCH_SIZE", 1024))
