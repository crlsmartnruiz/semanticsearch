from documents import DOCUMENTS
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


def get_embeddings(input: str):

    result = openai_client.embeddings.create(
        input=[input], model="text-embedding-3-small").data

    return list(map(lambda embedding: embedding.embedding, result))


load_dotenv()
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
openai_client = OpenAI()

collection = None

try:
    collection = chroma_client.get_or_create_collection(
        name="my_collection", metadata={"hnsw:space": "cosine"})
except Exception as e:
    print(e)

processed_documents = list(map(lambda doc: {
    "doc": doc["name"],
    "embeddings": get_embeddings(doc["name"]),
    "metadatas": [{"type": doc["type"]}],
    "id": doc["name"]
}, DOCUMENTS))

for doc in processed_documents:
    collection.add(
        documents=[doc["doc"]],
        embeddings=doc["embeddings"],
        metadatas=doc["metadatas"],
        ids=[doc["id"]]
    )

QUERY = "avión"

print(collection.query(
    query_embeddings=get_embeddings(
        list(filter(lambda elem: elem["name"] == QUERY, DOCUMENTS))[0]["name"])[0],
    n_results=4,
    # where={"type": "mar"}
))
