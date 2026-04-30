from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy
from rag.embedding import get_embedding_model
embeddings = get_embedding_model() # Trả về embedding model

def ragas_score(llm, question, answer, contexts):
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[doc.page_content for doc in contexts]], # List src được nối vào
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings
    )

    return {
        "faithfulness": float(result["faithfulness"][0]), # Điểm tin tưởng (ngược với halluciation)
        "relevance": float(result["answer_relevancy"][0])  # Điểm liên quan với question
    }