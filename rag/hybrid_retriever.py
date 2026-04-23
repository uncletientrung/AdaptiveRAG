from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rag.retriever import get_retriever
from langchain.schema import BaseRetriever
from typing import Any, Optional

class FilteredVectorRetriever(BaseRetriever): # Custom lại Retriever với vector
    vectorstore: Any
    k: int
    fetch_k: int
    filter_metadata: Optional[dict] = None

    def _get_relevant_documents(self, query, *, run_manager=None):
        docs = self.vectorstore.similarity_search(query, k=self.fetch_k)
        if self.filter_metadata:
            docs = [
                d for d in docs
                if d.metadata.get("file_name") == self.filter_metadata.get("file_name")
            ]

        return docs[:self.k]


def create_hybrid_retriever(vectorstore, chunks, top_k= 3, fetch_k=20, bm25_weight=0.35, vector_weight=0.65, filter_metadata =None):
    if filter_metadata:
        chunks = [ doc for doc in chunks
                    if doc.metadata.get("file_name") == filter_metadata.get("file_name")]
    if not chunks: 
        chunks = []

    bm25_retriever = BM25Retriever.from_documents(chunks) # keyword search
    bm25_retriever.k = fetch_k
    
    # vector_retriever = get_retriever(vectorstore, k=top_k, fetch_k=fetch_k) # semantic search
    vector_retriever = FilteredVectorRetriever(
        vectorstore=vectorstore,
        k=fetch_k,
        fetch_k=fetch_k,
        filter_metadata=filter_metadata
    )

    combine_Search = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )

    return combine_Search   