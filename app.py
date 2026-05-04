import streamlit as st
from utils.pdf_processor import extract_text, split_text
from utils.embeddings import create_embeddings
from utils.agent import run_agent

st.set_page_config(
    page_title="DocSage",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

* { font-family: 'DM Sans', sans-serif; }

html, body, [class*="css"] {
    background-color: #080810;
    color: #DDDDE8;
}

.stApp {
    background: #080810;
}

.hero {
    text-align: center;
    padding: 3.5rem 0 2rem 0;
}

.hero-badge {
    display: inline-block;
    background: transparent;
    border: 1px solid rgba(180, 170, 255, 0.25);
    color: #A89FFF;
    padding: 0.3rem 1.1rem;
    border-radius: 20px;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    font-weight: 500;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 5rem;
    font-weight: 400;
    color: #EEEEF5;
    margin: 0;
    line-height: 1;
    letter-spacing: -3px;
}

.hero-title span {
    font-style: italic;
    background: linear-gradient(135deg, #A89FFF, #FF8FC8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 0.85rem;
    color: #55556A;
    margin-top: 0.8rem;
    font-weight: 300;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,159,255,0.2), transparent);
    margin: 2rem 0;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(168,159,255,0.25);
    border-radius: 16px;
    padding: 1rem;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(168,159,255,0.5);
    background: rgba(168,159,255,0.03);
}

[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #DDDDE8 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(168,159,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(168,159,255,0.08) !important;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #A89FFF, #FF8FC8) !important;
    color: #080810 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}

[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(168,159,255,0.3) !important;
}

.stat-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}

.stat-number {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #A89FFF;
    letter-spacing: -1px;
}

.stat-label {
    font-size: 0.65rem;
    color: #55556A;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.3rem;
}

.answer-box {
    background: rgba(168,159,255,0.06);
    border: 1px solid rgba(168,159,255,0.15);
    border-radius: 16px;
    padding: 1.75rem;
    margin-top: 1rem;
    line-height: 1.9;
    font-size: 0.95rem;
    color: #DDDDE8;
}

.answer-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #A89FFF;
    margin-bottom: 0.85rem;
}

.retry-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,143,200,0.08);
    border: 1px solid rgba(255,143,200,0.18);
    color: #FF8FC8;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.72rem;
    margin-top: 0.75rem;
    letter-spacing: 0.05em;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin-top: 1.5rem;
}

.feature-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

.feature-card:hover {
    border-color: rgba(168,159,255,0.25);
    background: rgba(168,159,255,0.04);
    transform: translateY(-2px);
}

.feature-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    font-weight: 400;
    color: #EEEEF5;
    margin-bottom: 0.4rem;
    letter-spacing: -0.3px;
}

.feature-desc {
    font-size: 0.78rem;
    color: #55556A;
    line-height: 1.6;
    font-weight: 300;
}

.welcome-hint {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
    color: #3A3A50;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-badge">Agentic RAG System</div>
    <h1 class="hero-title">Doc<span>Sage </span></h1>
    <p class="hero-sub">Intelligent document intelligence powered by AI agents</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'], label_visibility="collapsed")

if uploaded_file is not None:
    with st.spinner("Processing your document..."):
        raw_text = extract_text(uploaded_file)

        if not raw_text.strip():
            st.error("Could not extract text from this PDF. Please try another file.")
        else:
            chunks = split_text(raw_text)
            vector_store = create_embeddings(chunks)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(chunks)}</div>
            <div class="stat-label">Chunks Created</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(raw_text):,}</div>
            <div class="stat-label">Characters Processed</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">3</div>
            <div class="stat-label">Max Agent Retries</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    question = st.text_input("", placeholder="Ask anything about your document...")

    if st.button("Ask DocSage"):
        if question.strip() == "":
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Agent is thinking..."):
                answer, docs, retry_count = run_agent(vector_store, question)

            st.markdown(f"""
            <div class="answer-box">
                <div class="answer-label">Answer</div>
                {answer}
            </div>
            """, unsafe_allow_html=True)

            if retry_count > 0:
                st.markdown(f"""
                <div class="retry-badge">
                    Agent refined search {retry_count} time(s)
                </div>
                """, unsafe_allow_html=True)

            with st.expander("View relevant document chunks"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.divider()

else:
    st.markdown('<div class="welcome-hint">Drop your PDF above to get started</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-title">Agentic Reasoning</div>
            <div class="feature-desc">Agent analyzes your question and decides the best search strategy before retrieving anything</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">Self Reflection</div>
            <div class="feature-desc">Checks its own answer quality and retries with better queries if the answer is not good enough</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">Multi-hop Retrieval</div>
            <div class="feature-desc">Searches multiple times across different parts of your document for complex questions</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">Powered by LLaMA 3.1</div>
            <div class="feature-desc">Groq inference for fast, accurate answers grounded entirely in your document</div>
        </div>
    </div>
    """, unsafe_allow_html=True)