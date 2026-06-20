from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from datetime import datetime
from document_processing.pdf_loader import load_and_split_pdf


def ocr_and_pdf_loader(file_path, chunk_size=1000, chunk_overlap=200):
    pdf_chunks, pdf_documents = load_and_split_pdf(file_path, chunk_size, chunk_overlap)
    page_docs = pdf_documents  
    images = convert_from_path(file_path) # Danh sách ảnh, mỗi trang cắt thành 1 ảnh
    final_docs = []
    file_name = os.path.basename(file_path)
    for i, page_doc in enumerate(page_docs):
        text = page_doc.page_content.strip()
        if len(text) > 100:  # Kiểm tra xem trang đó có text không
            final_docs.append(
                Document( page_content=text,
                    metadata={
                        "page": i,
                        "file_name": file_name,
                        "source": "pdf",
                        "loader": "pdf"
                    }
                )
            )
        else: # Nếu text trang đó < 100 
            ocr_text = pytesseract.image_to_string(images[i], lang="vie+eng")  # Lấy text từ ảnh trang đó
            if ocr_text.strip():
                final_docs.append(
                    Document(
                        page_content=ocr_text,
                        metadata={
                            "page": i,
                            "file_name": file_name,
                            "source": "ocr",
                            "loader": "ocr"
                        }
                    )
                )

    splitter = RecursiveCharacterTextSplitter( # Định nghĩa
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(final_docs) # Cắt chunk
    upload_time = datetime.now().isoformat()

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "file_name": file_name,            
            "file_type": "pdf",              
            "chunk_index": i,                 
            "upload_time": upload_time,       
        })

    return chunks, final_docs