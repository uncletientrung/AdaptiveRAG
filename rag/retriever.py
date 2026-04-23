from langchain_community.vectorstores import FAISS
from typing import Optional

def get_retriever(vectorstore: FAISS, k: int = 3, fetch_k: int = 20,  filter_metadata=None): # fetch_k là số vector để chọn top k
    return vectorstore.as_retriever( # Gán FAISS thành retriever
        search_type="similarity", # Tìm các vector gần query nhất
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
        }
    )
# as_retriever() của FAISS trong LangChain KHÔNG hỗ trợ filter_metadata