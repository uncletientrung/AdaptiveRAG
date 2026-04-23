from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, question, documents, top_k=3):
        list_Cap_Question_PageContent = [(question, doc.page_content) for doc in documents]
        scores = self.model.predict(list_Cap_Question_PageContent,batch_size=16) # Sử dụng model để tính điểm 
                    # batch_size=16 là xử lý tính điểm 16 cái cùng lúc, Trả về danh sách điểm VD: [0.2, 0.9, 0.5, 0.1, ...]
        scored_docs = list(zip(documents, scores)) # Tạo list gán từng document với điểm của nó
        ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)   # Sort giảm dần dựa trên score
        top_docs = []
        for doc, score in ranked_docs[:top_k]:
            top_docs.append(doc)
        return top_docs # Trả về doc với topk