from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from document_processing.pdf_loader import load_and_split_pdf
from document_processing.docx_loader import load_and_split_docx
from vector_store.faiss_store import create_faiss_vectorstore
from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.promt import VIETNAMESE_PROMPT
from rag.hybrid_retriever import create_hybrid_retriever
import os
import json
from rag.reranker import CrossEncoderReranker
from rag.rerank_retriever import RerankRetriever

def query_rewriter(llm, query): # Viết lại câu hỏi
    prompt = f"""
        Bạn là hệ thống tối ưu truy vấn tìm kiếm tài liệu.

        Nhiệm vụ:
        Viết lại câu hỏi người dùng để phù hợp hơn cho việc tìm kiếm trong tài liệu.

        Quy tắc:
        - Viết bằng tiếng Việt
        - Không đổi nghĩa
        - Làm rõ ý hơn
        - Ngắn gọn
        - Chỉ trả về câu hỏi đã viết lại

        Câu hỏi: {query}

        Trả lời:
    """
    response = llm.invoke(prompt)
    return get_llm_text(response).strip()

def confidence_scorer(): # Điểm tin cậy
    pass
def build_rag_pipeline(
        list_file_path, chunk_size=1000, chunk_overlap=200,
        top_k=3, fetch_k=15, temperature=0.7,
        filter_metadata=None # Dùng khi nếu user chọn filter cho multi file
    ):
    
    all_chunks = [] # Gộp hết các chunk trong các file thành 1 (cái này chứa các obj Document)
    all_documents = []
    for file_path in list_file_path:
        file_format = os.path.splitext(file_path)[1].lower()
        if file_format == ".pdf":
            chunks, documents = load_and_split_pdf(file_path, chunk_size, chunk_overlap)  # Load pdf và cắt chunk
        elif file_format == ".docx":
            chunks, documents = load_and_split_docx(file_path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks) 
        all_documents.extend(documents)

    vectorstore = create_faiss_vectorstore(all_chunks) # Tạo FAISS obj
    # retriever = get_retriever(vectorstore, top_k, fetch_k, filter_metadata) # Gán FAISS sang retriever
    # hybrid_retriever = create_hybrid_retriever(   # Sử dụng bi-encoder
    #     vectorstore=vectorstore,
    #     chunks=all_chunks,
    #     top_k=top_k,
    #     fetch_k=fetch_k,
    #     bm25_weight=0.35,   
    #     vector_weight=0.65
    # )
    base_retriever = create_hybrid_retriever(
        vectorstore=vectorstore,
        chunks=chunks,
        top_k=fetch_k, 
        fetch_k=fetch_k,
        bm25_weight=0.35,  vector_weight=0.65 
    )

    reranker = CrossEncoderReranker()
    hybrid_retriever = RerankRetriever(
        base_retriever=base_retriever,
        reranker=reranker,
        top_k=top_k,
        fetch_k=fetch_k
    )

    llm = get_llm(temperature=temperature) # Tạo Ollama
    chatMemory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True, # Định dạng trả về true thì là Obj ([ HumanMessage("Hi"), AIMessage("Hello")]), false thì "Human: Hi\nAI: Hello"
        output_key="answer" # Lấy nội dung answer để lưu vào memory
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=chatMemory,
        chain_type="stuff", # Nối các chunk lại vảo promt
        combine_docs_chain_kwargs={"prompt": VIETNAMESE_PROMPT},
        return_source_documents=True,
        verbose=False  # debug memory
    )
    # qa_chain = RetrievalQA.from_chain_type( 
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="stuff", # Nối các chunk lại vảo promt
    #     chain_type_kwargs={"prompt": VIETNAMESE_PROMPT},  # Dùng prompt tiếng Việt
    #     return_source_documents=True,  # Hiển thị nguồn
    # )
    qa_chain.memory.clear() # reset memory mỗi lần 
    return (
        qa_chain,
        vectorstore,
        all_chunks,
        all_documents,
    )

def multi_hop_reasoning(rag_chain, llm, query: str, so_buoc_lap=2): # Lặp lại câu trả lời để lấy final answer
    current_query = query
    final_answer = ""
    all_source = []
    for step in range(so_buoc_lap):
        result = rag_chain.invoke({"question": current_query})
        answer = result["answer"] # Lấy câu trả lời
        sources = result["source_documents"]
        all_source.extend(sources)

        prompt_multi_hop = f"""
            Dựa trên câu hỏi và câu trả lời sau, hãy xác định:

            - Nếu cần thêm thông tin → viết lại câu hỏi rõ hơn
            - Nếu đã đủ → trả về đúng từ: DONE

            Câu hỏi: {current_query}
            Câu trả lời: {answer}

            Trả lời:
        """

        new_query = get_llm_text(llm.invoke(prompt_multi_hop)).strip() # Viết lại câu hỏi
        if new_query == "DONE":
            final_answer = answer
            break
        current_query = new_query
        final_answer = answer

    return final_answer, all_source




def self_rag_evaluate(llm, question, answer, contexts): # Tự đánh giá câu trả lời
    context_text = "\n\n".join([c.page_content for c in contexts])
    prompt = f"""
        Bạn là hệ thống kiểm tra độ chính xác của câu trả lời AI.

        Nhiệm vụ:
        Đánh giá câu trả lời dựa trên NGỮ CẢNH được cung cấp.

        QUY TẮC:
        - Chỉ dựa vào Context
        - Không suy đoán
        - Trả về JSON hợp lệ

        FORMAT:
        {{
            "supported": true/false,
            "confidence": 0.0 - 1.0,
            "problem": "không có / thiếu thông tin / sai nội dung",
            "improved_answer": "câu trả lời cải thiện"
        }}

        CÂU HỎI:
        {question}

        CÂU TRẢ LỜI:
        {answer}

        NGỮ CẢNH:
        {context_text}

        Trả về JSON:
    """

    response = llm.invoke(prompt)
    text = get_llm_text(response)
    try:
        return json.loads(text)
    except:
        return {
            "supported": False,
            "confidence": 0.36666,
            "problem": "parse_error",
            "improved_answer": answer
        }

def get_llm_text(response): # Kiểm tra xem có hàm content không
    if hasattr(response, "content"):
        return response.content
    return response