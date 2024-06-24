from typing import List
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader


from dotenv import load_dotenv

load_dotenv()


def ingest_data(file_path: str):
    loader = TextLoader(file_path)
    pages = loader.load_and_split()

    elastic_vector_search.add_documents(pages)


embedding = OpenAIEmbeddings()
elastic_vector_search = ElasticsearchStore(
    es_url="https://localhost:9200",
    index_name="elastic-rag-index",
    embedding=embedding,
    es_user="elastic",
    es_password="elastic",
    es_params={
        "verify_certs": False
    }
)

ingest_data("../files/monitorizacion.txt")
ingest_data("../files/infraestructura.txt")
ingest_data("../files/devops.txt")


nearest_docs = elastic_vector_search.similarity_search_with_relevance_scores(
    "¿Qué tipos de logs existen?")
for doc in nearest_docs:
    print(doc)
    print()

retriever = elastic_vector_search.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", verbose=True)


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "¿Qué tipos de logs existen?"})

print()
print()
print()
print(response["answer"])
print()
print()
print()


for document in response["context"]:
    print(document)
    print()
