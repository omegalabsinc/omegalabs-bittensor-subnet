import os
from dotenv import load_dotenv
import json
from typing import List
import base64
import tempfile
from omega import constants
import boto3
load_dotenv(override=True)


def get_bool_env(key, default: bool=False):
    return os.getenv(key, str(default)).lower() == "true"

def get_secret(secret_name, region_name):
    """Retrieve secret from AWS Secrets Manager"""
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
    )
    
    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    
    secret = get_secret_value_response['SecretString']
    return secret

def setup_gcp_credentials():
    """
    Setup GCP credentials from AWS Secrets Manager based on environment.
    Creates a temporary file for Google Cloud libraries to use.
    """
    # Determine environment (dev or prod)
    is_prod = get_bool_env('IS_PROD', False)
    environment = 'prod' if is_prod else 'dev'
    
    # Map to correct secret name
    secret_names = {
        'dev': 'omega-labs/gcp/dev-service-account',
        'prod': 'omega-labs/gcp/prod-service-account'
    }
    
    secret_name = secret_names[environment]
    region = os.getenv('AWS_S3_REGION', 'us-east-2')
    
    # Get GCP credentials from Secrets Manager
    gcp_creds_json = get_secret(secret_name, region_name=region)
    
    # Create temporary file for credentials
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.write(gcp_creds_json)
    temp_file.close()
    
    # Set the credentials file path for Google Cloud libraries
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
    
    return temp_file.name


def parse_proxies(proxy_list: List[str]) -> List[str]:
    transformed_proxies = []
    for proxy in proxy_list:
        proxy_ip, proxy_port, proxy_user, proxy_pass = proxy.split(":")
        transformed_proxies.append(
            f"http://{proxy_user}:{proxy_pass}@{proxy_ip}:{proxy_port}"
        )
    return transformed_proxies


def robust_json_loads(json_str: str) -> List[str]:
    return json.loads(json_str.replace('\\"', '"'))


PORT = int(os.environ.get("PORT", 8002))
NETWORK = os.environ["NETWORK"]
print(f"Running with NETWORK={NETWORK}")
NETUID = int(os.environ["NETUID"])
STAKE_HOTKEY = os.environ["STAKE_HOTKEY"]

ENABLE_COMMUNE = True if os.environ["ENABLE_COMMUNE"] == "True" else False
print("Running with ENABLE_COMMUNE:", ENABLE_COMMUNE)
COMMUNE_NETWORK = os.environ["COMMUNE_NETWORK"]
COMMUNE_NETUID = int(os.environ["COMMUNE_NETUID"])

API_KEY_NAME = "OMEGA_MM_API_KEY"
API_KEYS_STR = os.environ["API_KEYS"]
API_KEYS = robust_json_loads(API_KEYS_STR)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
PINECONE_AUDIO_INDEX = os.environ["PINECONE_AUDIO_INDEX"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ["HF_REPO"]
HF_AUDIO_REPO = os.environ["HF_AUDIO_REPO"]
REPO_TYPE = "dataset"
TOPICS_LIST = robust_json_loads(os.environ["TOPICS_LIST"])
PROXY_LIST = parse_proxies(robust_json_loads(os.environ["PROXY_LIST"]))
IS_PROD = os.environ.get("IS_PROD", "false").lower() == "true"
CHECK_PROBABILITY = float(os.environ.get("CHECK_PROBABILITY", 0.1))
UPLOAD_BATCH_SIZE = int(os.environ.get("UPLOAD_BATCH_SIZE", 1024))
UPLOAD_AUDIO_BATCH_SIZE = int(os.environ.get("UPLOAD_AUDIO_BATCH_SIZE", 256))

DB_CONFIG = {
    "user": os.environ["DBUSER"],
    "password": os.environ["DBPASS"],
    "host": os.environ["DBHOST"],
    "database": os.environ["DBNAME"],
}

# Omega Focus Constants
FOCUS_DB_HOST = os.environ["FOCUS_DB_HOST"]
FOCUS_DB_NAME = os.environ["FOCUS_DB_NAME"]
FOCUS_DB_USER = os.environ["FOCUS_DB_USER"]
FOCUS_DB_PASSWORD = os.environ["FOCUS_DB_PASSWORD"]
FOCUS_DB_PORT = int(os.getenv("FOCUS_DB_PORT", 5432))
FOCUS_DB_POOL_SIZE = int(os.getenv("FOCUS_DB_POOL_SIZE", 20))
FOCUS_DB_MAX_OVERFLOW = int(os.getenv("FOCUS_DB_MAX_OVERFLOW", 30))
DB_STRING_LENGTH = 200
DB_STRING_LENGTH_LONG = 500
ENCRYPTION_KEY = os.environ["ENCRYPTION_KEY"]

BT_TESTNET = "test"
BT_MAINNET = "finney"
assert NETWORK in [BT_TESTNET, BT_MAINNET], (
    "SUBTENSOR_NETWORK must be either test or finney"
)
TAO_REFRESH_INTERVAL_MINUTES = int(os.getenv("TAO_REFRESH_INTERVAL_MINUTES", 10))

FOCUS_REWARDS_PERCENT = float(
    os.getenv("FOCUS_REWARDS_PERCENT", constants.FOCUS_REWARDS_PERCENT)
)
FOCUS_API_KEYS = robust_json_loads(os.environ["FOCUS_API_KEYS"])
FOCUS_API_URL = os.environ["FOCUS_API_URL"]
GOOGLE_AI_API_KEY = os.environ["GOOGLE_AI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

MAX_FOCUS_POINTS_PER_HOUR = int(
    os.getenv("MAX_FOCUS_POINTS_PER_HOUR", 80)
)  # $80 / hour
FIXED_TAO_USD_ESTIMATE = float(os.getenv("FIXED_TAO_USD_ESTIMATE", 300.0))
FIXED_ALPHA_TAO_ESTIMATE = float(
    os.getenv("FIXED_ALPHA_TAO_ESTIMATE", 0.0208)
)  # 1 alpha to tao, changes over time, you can find this with `btcli subnet list`
FIXED_TAO_ALPHA_ESTIMATE = 1 / FIXED_ALPHA_TAO_ESTIMATE
FIXED_ALPHA_USD_ESTIMATE = FIXED_ALPHA_TAO_ESTIMATE * FIXED_TAO_USD_ESTIMATE
BOOSTED_TASKS_PERCENTAGE = float(os.getenv("BOOSTED_TASKS_PERCENTAGE", 0.7))

GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")
GOOGLE_CLOUD_BUCKET_NAME = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")

# Setup GCP credentials from base64 environment variable
GOOGLE_APPLICATION_CREDENTIALS = setup_gcp_credentials()

SENTRY_DSN = os.getenv("SENTRY_DSN")
IMPORT_SCORE = os.getenv("IMPORT_SCORE", "true").lower() == "true"

# Subnet Videos Configuration
SUBNET_VIDEOS_WALLET_COLDKEY = os.getenv("SUBNET_VIDEOS_WALLET_COLDKEY")
SUBNET_VIDEOS_TAO_REWARD = float(os.getenv("SUBNET_VIDEOS_TAO_REWARD", "0.1"))
SUBNET_VIDEOS_COUNT = int(os.getenv("SUBNET_VIDEOS_COUNT", "5"))
