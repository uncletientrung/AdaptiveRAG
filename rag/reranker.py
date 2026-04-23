from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=3):
        pairs = [(query, doc.page_content) for doc in documents]

        scores = self.model.predict(pairs,batch_size=16) #Xử lý 16 cặp (query, doc) cùng lúc

        # Gắn score vào doc
        scored_docs = list(zip(documents, scores))

        # Sort giảm dần
        ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        # Lấy top_k
        return [doc for doc, _ in ranked_docs[:top_k]]