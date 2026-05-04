from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

# ─── STATE ───────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    search_query: str
    documents: List[Document]
    answer: str
    reflection: str
    retry_count: int
    vector_store: object

# ─── LLM ─────────────────────────────────────────────────
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

# ─── NODE 1: ANALYZE QUESTION ────────────────────────────
def analyze_question(state: AgentState) -> AgentState:
    llm = get_llm()
    
    prompt = PromptTemplate(
        template="""You are an expert at analyzing questions.
        Given the question below, generate the best possible search query 
        to find relevant information in a document.
        Return ONLY the search query, nothing else.
        
        Question: {question}
        
        Search Query:""",
        input_variables=["question"]
    )
    
    chain = prompt | llm
    result = chain.invoke({"question": state["question"]})
    search_query = result.content.strip()
    
    return {**state, "search_query": search_query}

# ─── NODE 2: RETRIEVE ─────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    vector_store = state["vector_store"]
    docs = vector_store.similarity_search(state["search_query"], k=5)
    return {**state, "documents": docs}

# ─── NODE 3: GENERATE ─────────────────────────────────────
def generate(state: AgentState) -> AgentState:
    llm = get_llm()
    
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer the question based on the context below.
        If the answer is not in the context, say "I could not find the answer in the document."
        Do not make up answers.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    chain = prompt | llm
    result = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    
    return {**state, "answer": result.content.strip()}

# ─── NODE 4: REFLECT ──────────────────────────────────────
def reflect(state: AgentState) -> AgentState:
    llm = get_llm()
    
    prompt = PromptTemplate(
        template="""You are a quality checker. Evaluate if the answer properly addresses the question.
        
        Question: {question}
        Answer: {answer}
        
        Is this answer complete and accurate? Reply with only YES or NO.""",
        input_variables=["question", "answer"]
    )
    
    chain = prompt | llm
    result = chain.invoke({
        "question": state["question"],
        "answer": state["answer"]
    })
    
    reflection = result.content.strip().upper()
    return {**state, "reflection": reflection}

# ─── NODE 5: REWRITE QUERY ────────────────────────────────
def rewrite_query(state: AgentState) -> AgentState:
    llm = get_llm()
    
    prompt = PromptTemplate(
        template="""The previous search did not return a good answer.
        Generate a different, better search query for the same question.
        Return ONLY the search query, nothing else.
        
        Original Question: {question}
        Previous Search Query: {search_query}
        Previous Answer: {answer}
        
        Better Search Query:""",
        input_variables=["question", "search_query", "answer"]
    )
    
    chain = prompt | llm
    result = chain.invoke({
        "question": state["question"],
        "search_query": state["search_query"],
        "answer": state["answer"]
    })
    
    new_query = result.content.strip()
    retry_count = state.get("retry_count", 0) + 1
    
    return {**state, "search_query": new_query, "retry_count": retry_count}

# ─── CONDITIONAL EDGE ─────────────────────────────────────
def should_retry(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    reflection = state.get("reflection", "NO")
    
    if reflection == "YES" or retry_count >= 3:
        return "end"
    else:
        return "retry"

# ─── BUILD GRAPH ──────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)
    
    graph.add_node("analyze_question", analyze_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("reflect", reflect)
    graph.add_node("rewrite_query", rewrite_query)
    
    graph.set_entry_point("analyze_question")
    graph.add_edge("analyze_question", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "reflect")
    
    graph.add_conditional_edges(
        "reflect",
        should_retry,
        {
            "end": END,
            "retry": "rewrite_query"
        }
    )
    
    graph.add_edge("rewrite_query", "retrieve")
    
    return graph.compile()

# ─── RUN AGENT ────────────────────────────────────────────
def run_agent(vector_store, question):
    agent = build_agent()
    
    initial_state = {
        "question": question,
        "search_query": question,
        "documents": [],
        "answer": "",
        "reflection": "",
        "retry_count": 0,
        "vector_store": vector_store
    }
    
    result = agent.invoke(initial_state)
    return result["answer"], result["documents"], result["retry_count"]