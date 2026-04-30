import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def _tokenize_vi(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt: lowercase + tách từ."""
    return re.findall(r'[\wÀ-ỹ]+', text.lower())


def _ngram_jaccard(text_a: str, text_b: str, n: int = 2) -> float:
    """Jaccard similarity trên n-gram giữa 2 đoạn văn."""
    def get_ngrams(tokens, n):
        return set(zip(*[tokens[i:] for i in range(n)]))

    tok_a = _tokenize_vi(text_a)
    tok_b = _tokenize_vi(text_b)
    ng_a = get_ngrams(tok_a, n)
    ng_b = get_ngrams(tok_b, n)

    if not ng_a or not ng_b:
        return 0.0
    return len(ng_a & ng_b) / len(ng_a | ng_b)


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Cosine similarity dùng TF-IDF (không cần embedding model)."""
    try:
        vec = TfidfVectorizer(
            analyzer='char_wb',   # char n-gram → tốt cho tiếng Việt
            ngram_range=(2, 4),
            min_df=1,
        )
        matrix = vec.fit_transform([text_a, text_b])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except Exception:
        return 0.0


def _faithfulness_score(answer: str, contexts: list) -> float:
    """
    Faithfulness: câu trả lời có bám sát context không?
    → Đo bằng max similarity của answer với từng context chunk.
    Dùng trung bình có trọng số: 60% TF-IDF cosine + 40% bigram Jaccard.
    """
    if not contexts:
        return 0.0

    context_texts = [doc.page_content for doc in contexts]
    scores = []
    for ctx in context_texts:
        cos = _tfidf_cosine(answer, ctx)
        jac = _ngram_jaccard(answer, ctx, n=2)
        scores.append(0.6 * cos + 0.4 * jac)

    # Lấy top-2 chunk liên quan nhất, tránh bị kéo xuống bởi chunk không liên quan
    top2 = sorted(scores, reverse=True)[:2]
    return float(np.mean(top2))


def _relevance_score(question: str, answer: str) -> float:
    """
    Relevance: câu trả lời có trả lời đúng câu hỏi không?
    → Đo similarity giữa question và answer.
    """
    cos = _tfidf_cosine(question, answer)
    jac = _ngram_jaccard(question, answer, n=1)  # unigram cho câu hỏi ngắn
    return 0.7 * cos + 0.3 * jac


def _coverage_score(answer: str, contexts: list) -> float:
    """
    Coverage (Completeness một phần): answer có cover nhiều chunk không?
    → Tỉ lệ chunk có similarity đáng kể với answer.
    Tránh trường hợp answer chỉ bám 1 chunk, bỏ qua phần còn lại.
    """
    if not contexts:
        return 0.0

    threshold = 0.08   # ngưỡng similarity tối thiểu để tính là "có liên quan"
    covered = sum(
        1 for doc in contexts
        if _tfidf_cosine(answer, doc.page_content) >= threshold
    )
    return covered / len(contexts)


def self_rag_evaluate(question: str, answer: str, contexts: list) -> float:
    """
    Đánh giá chất lượng câu trả lời RAG bằng metric xác định (không dùng LLM).

    Trả về điểm tổng hợp [0.0, 1.0]:
        - 0.0 → 0.4 : câu trả lời tệ (không liên quan / hallucinate nặng)
        - 0.4 → 0.6 : chấp nhận được
        - 0.6 → 0.8 : tốt
        - 0.8 → 1.0 : rất tốt, bám sát context và trả lời đúng câu hỏi

    Không nhận `llm` làm tham số — loại bỏ vòng lặp LLM đánh giá LLM.
    """
    if not answer or not answer.strip():
        return 0.0

    faith = _faithfulness_score(answer, contexts)   # 50% — quan trọng nhất
    relev = _relevance_score(question, answer)       # 35%
    cover = _coverage_score(answer, contexts)        # 15%

    score = 0.50 * faith + 0.35 * relev + 0.15 * cover
    return round(min(max(score, 0.0), 1.0), 4)