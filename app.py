import streamlit as st
from utils.pdf_processor import extract_text, split_text
from utils.embeddings import create_embeddings, load_embeddings
from utils.qa_chain import get_qa_chain, get_answer

st.set_page_config(
    page_title="DocSage",
    page_icon="",
    layout="wide"
)

st.title("DocSage")
st.subheader("Upload a PDF and ask anything about it")
st.divider()

uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

if uploaded_file is not None:
    with st.spinner("Reading and processing your document..."):
        raw_text = extract_text(uploaded_file)
        
        if not raw_text.strip():
            st.error("Could not extract text from this PDF. Please try another file.")
        else:
            chunks = split_text(raw_text)
            vector_store = create_embeddings(chunks)
            chain = get_qa_chain()
            
            st.success(f"Document processed successfully! {len(chunks)} chunks created.")
            st.divider()
            
            st.header("Ask a Question")
            question = st.text_input("Type your question about the document:")
            
            if st.button("Get Answer"):
                if question.strip() == "":
                    st.warning("Please enter a question first.")
                else:
                    with st.spinner("Finding answer..."):
                        docs = load_embeddings(vector_store, question)
                        answer = get_answer(chain, docs, question)
                    
                    st.subheader("Answer")
                    st.write(answer)
                    
                    with st.expander("View relevant document chunks"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)
                            st.divider()
else:
    st.info("Upload a PDF file to get started!")
    st.markdown("""
    ### What DocSage does:
    - **Reads** your PDF document
    - **Understands** the content using AI embeddings
    - **Answers** your questions based on the document
    - **Shows** the relevant sections it used to answer
    """)