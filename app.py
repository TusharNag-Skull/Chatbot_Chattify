


import streamlit as st
import pandas as pd
from rag_chatbot import build_qa_chain
import time

# Page configuration
st.set_page_config(
    page_title="📚 RAG Q&A Chatbot", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Input section styling */
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Answer section styling */
    .answer-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .answer-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Source styling */
    .source-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .source-header {
        color: #495057;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .source-content {
        color: #6c757d;
        font-size: 0.9rem;
        line-height: 1.6;
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    
    /* Spinner styling */
    .stSpinner {
        text-align: center;
        color: #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    /* Warning styling */
    .custom-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Success styling */
    .custom-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Loan Approval Prediction Chatbot - Chatify</h1>
    <p class="header-subtitle">Powered by RAG (Retrieval-Augmented Generation)</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Input section
    st.markdown("""
    <div class="input-container">
        <h3 style="color: #2c3e50; margin-bottom: 1rem;">💬 Ask Your Question</h3>
        <p style="color: #6c757d; margin-bottom: 1rem;">Enter your question about the documents below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Build QA chain
    @st.cache_resource
    def get_qa_chain():
        return build_qa_chain()
    
    qa_chain = get_qa_chain()
    
    # Query input
    query = st.text_input(
        "", 
        placeholder="E.g., How many loan applications were approved?",
        help="Type your question here and press Enter"
    )
    
    # Example questions
    st.markdown("**💡 Example Questions:**")
    example_questions = [
        "How many loan applications were approved?",
        "What is the average loan amount?",
        "Show me the distribution by region",
        "What are the key insights from the data?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"📝 {question}", key=f"example_{i}"):
                query = question
                st.rerun()

def display_as_table(text_result):
    """Enhanced table display function"""
    try:
        rows = [row.strip() for row in text_result.strip().split('\n') if row.strip()]
        structured_data = [r.split(",") for r in rows if "," in r]
        
        if not structured_data:
            # Just display the plain text response without warning
            st.markdown(f"""
            <div style="
                padding: 1.5rem;
                border-radius: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            ">
                {text_result}
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Create DataFrame
        df = pd.DataFrame(structured_data)
        
        # Display table
        st.dataframe(
            df, 
            use_container_width=True,
            hide_index=True
        )
        
        # Add download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="qa_results.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error processing result: {str(e)}")
        st.text(text_result)

# Process query
if query:
    with st.spinner("🔍 Analyzing your question and searching through documents..."):
        # Add a slight delay for better UX
        time.sleep(0.5)
        
        try:
            result = qa_chain({"query": query})
            
            # Answer section
            st.markdown("""
            <div class="answer-container">
                <div class="answer-header">
                    🗣️ Answer
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            display_as_table(result["result"])
            
            # Sources section
            if "source_documents" in result and result["source_documents"]:
                st.markdown("### 📚 Source Documents")
                
                for i, doc in enumerate(result["source_documents"]):
                    with st.expander(f"📄 Source {i+1} - Click to expand"):
                        st.markdown(f"""
                        <div class="source-container">
                            <div class="source-header">Document Content:</div>
                            <div class="source-content">
                                {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add metadata if available
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.markdown("**📋 Metadata:**")
                            for key, value in doc.metadata.items():
                                st.write(f"• **{key}:** {value}")
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": query,
                "answer": result["result"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please try rephrasing your question or check if the documents are properly loaded.")

# Sidebar with additional features
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    if st.button("ℹ️ About This App"):
        st.info("""
        This RAG-powered chatbot can answer questions about your documents using:
        
        • **Retrieval-Augmented Generation (RAG)**
        • **Natural Language Processing**
        • **Document Embedding & Search**
        
        Simply ask questions in natural language!
        """)
    
    # Show chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Recent Questions")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:30]}..."):
                st.write(f"**Asked:** {chat['timestamp']}")
                st.write(f"**Question:** {chat['question']}")
                st.write(f"**Answer:** {chat['answer'][:200]}...")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <p>Created By Tushar Nag | Chatbot</p>
</div>
""", unsafe_allow_html=True)