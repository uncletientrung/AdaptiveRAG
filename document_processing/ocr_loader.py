from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from datetime import datetime

def ocr_pdf(file_path, chunk_size: int = 1000, chunk_overlap: int = 200):
    images = convert_from_path(file_path)

    all_text = []
    documents = []
    file_name = os.path.basename(file_path)
    upload_time = datetime.now().isoformat()

    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="vie+eng")
        if text.strip():
            all_text.append(text)
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": i,
                        "file_name": file_path,
                        "source": "ocr"
                    }
                )
            )
    
    text_splitter = RecursiveCharacterTextSplitter( # Cắt chunk
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "file_name": file_name,              # tên file
            "file_type": "pdf",               # loại file
            "chunk_index": i,                 # thứ tự chunk
            "upload_time": upload_time,       # thời gian upload
        })

    return chunks, documents