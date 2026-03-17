# DocSage

DocSage is an AI-powered document question answering application. Users can upload any PDF document and ask questions about it in natural language. The application uses Retrieval Augmented Generation (RAG) to find relevant sections of the document and generate accurate answers using a large language model.

## Features

- Upload any PDF document
- Ask questions in natural language
- AI-powered answers based on document content
- View the relevant document chunks used to generate the answer
- Handles large documents by splitting into searchable chunks

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq API (LLaMA 3.1)
- FAISS
- HuggingFace Sentence Transformers
- PyPDF2

## How It Works

1. User uploads a PDF document
2. The app extracts all text from the PDF using PyPDF2
3. The text is split into chunks of 500 characters with 50 character overlap using LangChain
4. Each chunk is converted into a numerical embedding using HuggingFace sentence transformers
5. All embeddings are stored in a FAISS vector store for fast similarity search
6. When the user asks a question, the question is also converted to an embedding
7. FAISS finds the top 5 most similar chunks to the question
8. The relevant chunks and the question are sent to Groq LLaMA 3.1 via LangChain
9. The model generates an answer based on the document content
10. The answer and relevant chunks are displayed to the user

## Project Structure

    DocSage/
    ├── app.py
    ├── utils/
    │   ├── __init__.py
    │   ├── pdf_processor.py
    │   ├── embeddings.py
    │   └── qa_chain.py
    ├── .env
    ├── .gitignore
    ├── requirements.txt
    └── README.md

## Getting Started

### Prerequisites

Make sure you have Python 3.8 or above installed on your system. You will also need a free Groq API key from console.groq.com.

### Installation

1. Clone the repository

        git clone https://github.com/yourusername/DocSage.git

2. Navigate to the project directory

        cd DocSage

3. Create a virtual environment

        python3 -m venv venv
        source venv/bin/activate

4. Install dependencies

        pip install -r requirements.txt

5. Create a .env file and add your Groq API key

        GROQ_API_KEY=your_groq_api_key_here

6. Run the application

        streamlit run app.py

7. Open your browser and go to http://localhost:8501

## Usage

1. Upload a PDF file using the file uploader
2. Wait for the document to be processed
3. Type your question in the text input
4. Click Get Answer
5. View the answer and expand the relevant chunks section to see which parts of the document were used

## What is RAG

Retrieval Augmented Generation is a technique that combines information retrieval with text generation. Instead of relying solely on the model's training data, RAG retrieves relevant information from a provided document and uses it as context for generating answers. This allows the model to answer questions about documents it has never seen before.
