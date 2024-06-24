from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from pprint import pprint

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch.exceptions import BadRequestError

from documents import DOCUMENTS


def get_embeddings(input: List[dict]):
    document_names = list(map(lambda elem: elem["name"], input))

    result = openai_client.embeddings.create(
        input=document_names, model="text-embedding-3-small").data

    return list(map(lambda embedding: embedding.embedding, result))


def create_es_index():
    try:
        client.indices.create(index="vehicles-index", mappings={
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "similarity": "cosine"
                },
                "name": {
                    "type": "text"
                },
                "type": {
                    "type": "text"
                }
            }
        })
    except BadRequestError as e:
        print(e.message)


def ingest_documents(embedded_documents: List[List[float]]):
    actions = [
        {
            "_index": "vehicles-index",
            "_id": j,
            "_source": {
                "vector": embedded_documents[j],
                "name": DOCUMENTS[j].get("name"),
                "type": DOCUMENTS[j].get("type")
            }
        }
        for j in range(0, len(DOCUMENTS))
    ]

    helpers.bulk(client=client, actions=actions)


load_dotenv()
openai_client = OpenAI()
client = Elasticsearch(
    "https://elastic:elastic@localhost:9200",  # Elasticsearch endpoint
    verify_certs=False
)
QUERY = "avión"


create_es_index()
embedded_documents = get_embeddings(DOCUMENTS)
ingest_documents(embedded_documents)

client.indices.refresh(index="vehicles-index")

resp = client.knn_search(index="vehicles-index", knn={
    "field": "vector",
    "query_vector": get_embeddings([list(filter(lambda elem: elem["name"] == QUERY, DOCUMENTS))][0])[0],
    "k": 4,
    "num_candidates": 100
# },
#     filter={
#         "term": {
#             "type": "tierra"
#         }
},
    source=["name"])

pprint(resp._body)
