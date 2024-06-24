from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from pprint import pprint

from elasticsearch import (Elasticsearch, helpers)
from elasticsearch.exceptions import BadRequestError


def get_embeddings(input: str):
    result = openai_client.embeddings.create(
        input=input, model="text-embedding-3-small").data

    return result[0].embedding


def create_es_index():
    try:
        client.indices.create(index="images-index", mappings={
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "similarity": "cosine"
                },
                "content": {
                    "type": "text"
                },
                "url": {
                    "type": "text"
                }
            }
        })
    except BadRequestError as e:
        print(e.message)


def ingest_documents(documents: List[dict]):
    actions = [
        {
            "_index": "images-index",
            "_source": {
                "vector": doc["embedding"],
                "content": doc["content"],
                "url": doc["url"]
            }
        }
        for doc in documents
    ]

    helpers.bulk(client=client, actions=actions)


def get_image_description(image: str) -> str:
    return llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who perfectly describes images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    }
                ]
            }
        ]
    )["choices"][0]["message"]["content"]


def prepare_documents(images):
    documents = list(map(lambda image: {
        "content": get_image_description(image),
        "url": image
    }, images))

    for doc in documents:
        doc["embedding"] = get_embeddings(doc["content"])

    return documents


load_dotenv()

images = [
    "file:///home/ubuntu/documentos/semanticsearch/files/elephant.jpg",
    "https://media.istockphoto.com/id/1367357589/es/foto/cielo-rojo-en-forma-de-coraz%C3%B3n-al-atardecer-hermoso-paisaje-con-flores-me-encanta-el-fondo-con.jpg?s=2048x2048&w=is&k=20&c=jWjaUWV4hEC8DU-ilYk1cDhAJThW19LNkZ3WAZATCKY=",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Gull_portrait_ca_usa.jpg/800px-Gull_portrait_ca_usa.jpg?20101128165003",
    "https://i.pinimg.com/originals/d1/ae/fb/d1aefbcdb1316b65f8d1d528138f36ac.jpg"
]

chat_handler = Llava15ChatHandler(
    clip_model_path="/home/ubuntu/documentos/llama.cpp/models/mmproj-ggml_llava-v1.5-13b.gguf")
llm = Llama(
    model_path="/home/ubuntu/documentos/llama.cpp/models/ggml_llava-v1.5-13b.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,  # n_ctx should be increased to accommodate the image embedding
)

openai_client = OpenAI()
client = Elasticsearch(
    "https://elastic:elastic@localhost:9200",  # Elasticsearch endpoint
    verify_certs=False
)

# Create Elasticsearch index with mappings
create_es_index()

# Take images array and convert it to dictionary with keys: content, url and embedding
documents = prepare_documents(images)

# Load documents into Elasticsearch
ingest_documents(documents)

client.indices.refresh(index="images-index")

# Query Elasticsearch
resp = client.knn_search(index="images-index", knn={
    "field": "vector",
    "query_vector": get_embeddings("dumbo en el cielo"),
    "k": 1,
    "num_candidates": 100},
    source=["content", "url"])

pprint(resp.body)
