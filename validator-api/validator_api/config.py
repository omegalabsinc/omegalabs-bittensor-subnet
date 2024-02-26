import os
import json

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = "salmanshahid/omega-mm"
REPO_TYPE = "dataset"
TOPICS_LIST = json.loads(os.environ["TOPICS_LIST"])
