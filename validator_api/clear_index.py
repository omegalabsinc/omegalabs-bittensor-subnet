from validator_api.validator_api import config
from pinecone import Pinecone

PINECONE_INDEX = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
PINECONE_INDEX.delete(delete_all=True)
