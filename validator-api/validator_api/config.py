import os
from dotenv import load_dotenv
import json
from typing import List
import boto3
from omega import constants

load_dotenv(override=True)

def get_secret(secret_name, region_name):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
    )

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )

    # For a list of exceptions thrown, see
    # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']

    return secret

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
PINECONE_AUDIO_INDEX = os.environ["PINECONE_AUDIO_INDEX"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ["HF_REPO"]
HF_AUDIO_REPO = os.environ["HF_AUDIO_REPO"]
REPO_TYPE = "dataset"
TOPICS_LIST = json.loads(os.environ["TOPICS_LIST"])
PROXY_LIST = parse_proxies(json.loads(os.environ["PROXY_LIST"]))
IS_PROD = os.environ.get("IS_PROD", "false").lower() == "true"
CHECK_PROBABILITY = float(os.environ.get("CHECK_PROBABILITY", 0.1))
UPLOAD_BATCH_SIZE = int(os.environ.get("UPLOAD_BATCH_SIZE", 1024))
UPLOAD_AUDIO_BATCH_SIZE = int(os.environ.get("UPLOAD_AUDIO_BATCH_SIZE", 256))

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
FOCUS_DB_PORT = os.getenv("FOCUS_DB_PORT", 5432)
DB_STRING_LENGTH = 200
DB_STRING_LENGTH_LONG = 500
ENCRYPTION_KEY = os.environ["ENCRYPTION_KEY"]

BT_TESTNET = "test"
BT_MAINNET = "finney"
assert NETWORK in [BT_TESTNET, BT_MAINNET], "SUBTENSOR_NETWORK must be either test or finney"
TAO_REFRESH_INTERVAL_MINUTES = int(os.getenv('TAO_REFRESH_INTERVAL_MINUTES', 10))

FOCUS_REWARDS_PERCENT = float(os.getenv('FOCUS_REWARDS_PERCENT', constants.FOCUS_REWARDS_PERCENT))
FOCUS_API_KEYS = json.loads(os.environ["FOCUS_API_KEYS"])
GOOGLE_AI_API_KEY = os.environ["GOOGLE_AI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_S3_REGION = os.environ["AWS_S3_REGION"]
AWS_S3_BUCKET_NAME = os.environ["AWS_S3_BUCKET_NAME"]

MAX_FOCUS_POINTS_PER_HOUR = int(os.getenv("MAX_FOCUS_POINTS_PER_HOUR", 80))  # $80 / hour
FIXED_TAO_USD_ESTIMATE = float(os.getenv("FIXED_TAO_USD_ESTIMATE", 300.0))
FIXED_ALPHA_TAO_ESTIMATE = float(os.getenv("FIXED_ALPHA_TAO_ESTIMATE", 0.001))  # 1 alpha to tao, changes over time, you can find this with `btcli subnet list`
FIXED_TAO_ALPHA_ESTIMATE = 1 / FIXED_ALPHA_TAO_ESTIMATE
BOOSTED_TASKS_PERCENTAGE = float(os.getenv("BOOSTED_TASKS_PERCENTAGE", 0.7))

GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

with open(GOOGLE_APPLICATION_CREDENTIALS, "w") as f:
    f.write(get_secret("prod/gcp_service_user", region_name=AWS_S3_REGION))

SENTRY_DSN = os.getenv("SENTRY_DSN")
IMPORT_SCORE = os.getenv("IMPORT_SCORE", "true").lower() == "true"