from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rag.retriever import get_retriever
from langchain.schema import BaseRetriever
from typing import Any, Optional

class FilteredVectorRetriever(BaseRetriever):
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
    
class HybridRerankRetriever(BaseRetriever):
    ensemble_retriever: Any
    reranker: Optional[Any] = None
    top_k: int = 3

    def _get_relevant_documents(self, query, *, run_manager=None):
        docs = self.ensemble_retriever.get_relevant_documents(query)
        docs = docs[:30] # Giới hạn lại
        # dedupe nhẹ
        seen, uniq = set(), []
        for d in docs:
            key = (d.metadata.get("file_name"), d.metadata.get("chunk_index"), d.page_content[:100])
            if key not in seen:
                seen.add(key)
                uniq.append(d)

        if self.reranker:
            return self.reranker.rerank(query, uniq, top_k=self.top_k)
        return uniq[:self.top_k]


def create_hybrid_retriever(vectorstore, chunks, top_k= 3, fetch_k=20, bm25_weight=0.35, vector_weight=0.65, filter_metadata =None):
    if filter_metadata:
        chunks = [ doc for doc in chunks
                    if doc.metadata.get("file_name") == filter_metadata.get("file_name")]
    if not chunks: 
        chunks = []

    bm25_retriever = BM25Retriever.from_documents(chunks) # keyword search
    bm25_retriever.k = top_k
    
    # vector_retriever = get_retriever(vectorstore, k=top_k, fetch_k=fetch_k) # semantic search
    vector_retriever = FilteredVectorRetriever(
        vectorstore=vectorstore,
        k=top_k,
        fetch_k=fetch_k,
        filter_metadata=filter_metadata
    )

    combine_Search = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )

    return combine_Search   