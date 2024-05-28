import os
import json
from typing import List


def parse_proxies(proxy_list: List[str]) -> List[str]:
    transformed_proxies = []
    for proxy in proxy_list:
        proxy_ip, proxy_port, proxy_user, proxy_pass = proxy.split(':')
        transformed_proxies.append(f"http://{proxy_user}:{proxy_pass}@{proxy_ip}:{proxy_port}")
    return transformed_proxies


NETWORK = os.environ["NETWORK"]
NETUID = int(os.environ["NETUID"])

ENABLE_COMMUNE = True if os.environ["ENABLE_COMMUNE"] == "True" else False
print("Running with ENABLE_COMMUNE:", ENABLE_COMMUNE)
COMMUNE_NETWORK = os.environ["COMMUNE_NETWORK"]
COMMUNE_NETUID = int(os.environ["COMMUNE_NETUID"])

API_KEY_NAME = "OMEGA_MM_API_KEY"
API_KEYS = json.loads(os.environ["API_KEYS"])

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

DB_CONFIG = {
    'user': os.environ["DBUSER"],
    'password': os.environ["DBPASS"],
    'host': os.environ["DBHOST"],
    'database': os.environ["DBNAME"]
}