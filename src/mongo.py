from typing import List
from openai import OpenAI

from documents import DOCUMENTS
import pymongo
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel
from dotenv import load_dotenv
import os


def get_embeddings(input: List[dict]):
    document_names = list(map(lambda elem: elem["name"], input))

    result = openai_client.embeddings.create(
        input=document_names, model="text-embedding-3-small").data

    return list(map(lambda embedding: embedding.embedding, result))


def create_search_index(collection):
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": 1536,
                    "path": "vector",
                    "similarity": "cosine"
                },
                {
                    "type": "filter",
                    "path": "name"
                },
                {
                    "type": "filter",
                    "path": "type"
                },
            ]
        },
        name="vehicles",
        type="vectorSearch",
    )

    try:
        collection.create_search_indexes(models=[search_index_model])
    except OperationFailure as e:
        print(e._message)


def ingest_data(collection):
    actions = [
        {
            "vector": embedded_documents[j],
            "name": DOCUMENTS[j].get("name"),
            "type": DOCUMENTS[j].get("type"),
        }
        for j in range(0, len(DOCUMENTS))
    ]

    collection.insert_many(actions)


def query_data(embedded_query, collection):
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vehicles',
                'path': 'vector',
                'queryVector': embedded_query,
                'numCandidates': 150,
                'limit': 4,
                # 'filter': {
                #     '$and': [
                #         {
                #             'type': {
                #                 '$in': ['mar']
                #             }
                #         }
                #     ]
                # }
            }
        }, {
            '$project': {
                '_id': 0,
                'vector': 1,
                'name': 1,
                'type': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                }
            }
        }
    ]

    result = collection.aggregate(pipeline)
    for i in result:
        print(i)
        print()
        print()


load_dotenv()
openai_client = OpenAI()
embedded_documents = get_embeddings(DOCUMENTS)

MONGODB_PASSWORD = os.environ.get("MONGODB_PASSWORD")
client = pymongo.MongoClient(
    f"mongouri")
database = client["semanticsearch"]
collection = database["vehicles"]

QUERY = "avión"

create_search_index(collection)
ingest_data(collection)


query_data(get_embeddings(
    [list(filter(lambda elem: elem["name"] == QUERY, DOCUMENTS))][0])[0], collection)
