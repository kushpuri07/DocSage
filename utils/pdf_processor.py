from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks