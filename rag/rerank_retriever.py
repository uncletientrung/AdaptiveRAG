from langchain.schema import BaseRetriever
from typing import List
from langchain_core.documents import Document
from pydantic import Field

class RerankRetriever(BaseRetriever):
    base_retriever: BaseRetriever = Field(...)
    reranker: object = Field(...)
    top_k: int = 3
    fetch_k: int = 20

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Bước 1: retrieve nhiều doc
        docs = self.base_retriever.get_relevant_documents(query)

        # Bước 2: rerank
        reranked_docs = self.reranker.rerank(query, docs, self.top_k)

        return reranked_docs