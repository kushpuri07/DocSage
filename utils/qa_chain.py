from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

def get_qa_chain():
    prompt_template = """
    You are a helpful assistant. Answer the question using the provided context.
    If the context contains relevant information, use it to answer.
    If the context does not contain enough information, answer from your general knowledge but mention it.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )
    
    chain = prompt | model
    return chain

def get_answer(chain, docs, question):
    context = "\n\n".join([doc.page_content for doc in docs])
    response = chain.invoke({
        "context": context,
        "question": question
    })
    return response.content