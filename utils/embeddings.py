from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def load_embeddings(vector_store, query):
    docs = vector_store.similarity_search(query, k=5)
    return docs