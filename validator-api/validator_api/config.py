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

# Omega Focus Constants
FOCUS_DB_HOST = os.environ["FOCUS_DB_HOST"]
FOCUS_DB_NAME = os.environ["FOCUS_DB_NAME"]
FOCUS_DB_USER = os.environ["FOCUS_DB_USER"]
FOCUS_DB_PASSWORD = os.environ["FOCUS_DB_PASSWORD"]
DB_STRING_LENGTH = 200
ENCRYPTION_KEY = os.environ["ENCRYPTION_KEY"]

BT_TESTNET = "test"
BT_MAINNET = "finney"
assert NETWORK in [BT_TESTNET, BT_MAINNET], "SUBTENSOR_NETWORK must be either test or finney"
TAO_REFRESH_INTERVAL_MINUTES = int(os.getenv('TAO_REFRESH_INTERVAL_MINUTES', 10))
FV_EMISSIONS_PCT = float(os.getenv('FV_EMISSIONS_PCT', 0.2))

FOCUS_BACKEND_API_URL = os.environ["FOCUS_BACKEND_API_URL"]
FOCUS_API_KEYS = json.loads(os.environ["FOCUS_API_KEYS"])
GOOGLE_AI_API_KEY = os.environ["GOOGLE_AI_API_KEY"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_S3_REGION = os.environ["AWS_S3_REGION"]
AWS_S3_BUCKET_NAME = os.environ["AWS_S3_BUCKET_NAME"]

MAX_FOCUS_POINTS = int(os.getenv('MAX_FOCUS_POINTS', 1000))