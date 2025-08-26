import streamlit as st
import os
import tempfile
from pathlib import Path
import base64
from datetime import datetime
import json
from typing import List, Dict, Any
import pandas as pd

# Import our RAG agent components
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.error("❌ python-dotenv package not found. Please install it with: pip install python-dotenv")
    st.error("💡 Make sure to activate your virtual environment first: source venv/bin/activate")
    st.stop()

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
except ImportError as e:
    st.error(f"❌ Required packages not found: {str(e)}")
    st.error("Please install required packages with: pip install -r requirements.txt")
    st.stop()

import csv

# Page configuration
st.set_page_config(
    page_title="Medical Document Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Chat container */
    .chat-container {
        background: transparent;
        padding: 20px;
        margin: 20px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Message styling - more like ChatGPT */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 25px;
        border-radius: 25px 25px 5px 25px;
        margin: 20px 0;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        font-size: 16px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    
    .ai-message {
        background: #ffffff;
        color: #333;
        padding: 20px 25px;
        border-radius: 25px 25px 25px 5px;
        margin: 20px 0;
        max-width: 85%;
        margin-right: auto;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        font-size: 16px;
        line-height: 1.6;
        word-wrap: break-word;
    }
    
    /* Input styling - more like ChatGPT */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e9ecef;
        padding: 18px 25px;
        font-size: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        background: #ffffff;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.25);
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #999;
        font-style: italic;
    }
    
    /* Button styling - more like ChatGPT */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 35px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    .stButton > button:active, .stFormSubmitButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* File upload styling */
    .stFileUploader > div {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* History item styling */
    .history-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    /* Status indicators */
    .status-success {
        color: #28a745;
        font-weight: 600;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: 600;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: 600;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .chat-container {
            margin: 10px;
            padding: 15px;
        }
        
        .user-message, .ai-message {
            max-width: 90%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'current_conversation' not in st.session_state:
    st.session_state.current_conversation = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

class MedicalDocumentAssistant:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.model_provider = "Unknown"
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize LLM and embeddings with fallback logic."""
        # Try OpenAI first
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    api_key=openai_api_key
                )
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=openai_api_key
                )
                self.model_provider = "OpenAI"
                return
            except Exception as e:
                st.warning(f"OpenAI connection failed: {str(e)}")
        
        # Fallback to Mistral AI
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if mistral_api_key:
            try:
                from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
                self.llm = ChatMistralAI(
                    model="mistral-large-latest",
                    temperature=0,
                    api_key=mistral_api_key
                )
                self.embeddings = MistralAIEmbeddings(
                    model="mistral-embed",
                    api_key=mistral_api_key
                )
                self.model_provider = "Mistral AI"
                return
            except Exception as e:
                st.error(f"Mistral AI connection failed: {str(e)}")
        
        st.error("No working API connections found. Please check your API keys.")
    
    def load_documents_from_file(self, uploaded_file):
        """Load documents from uploaded file."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
            elif file_extension == ".docx":
                loader = Docx2txtLoader(tmp_file_path)
                documents = loader.load()
            elif file_extension in [".txt", ".soap"]:
                loader = TextLoader(tmp_file_path, encoding="utf-8")
                documents = loader.load()
            elif file_extension == ".csv":
                with open(tmp_file_path, newline="", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    lines = []
                    for row in reader:
                        lines.append(", ".join(row))
                    content = "\n".join(lines)
                    documents = [Document(page_content=content, metadata={"source": uploaded_file.name})]
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            return documents
            
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return None
    
    def create_vectorstore(self, documents):
        """Create or update vector store with documents."""
        try:
            if not documents:
                return None
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Create vector store
            persist_directory = "chroma_store"
            collection_name = "medical_documents"
            
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)
            
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            
            return vectorstore
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None
    
    def get_response(self, query: str, vectorstore) -> str:
        """Get response from the RAG system."""
        try:
            # Handle small-talk/identity queries without requiring documents
            normalized = query.strip().lower()
            small_talk_triggers = [
                "who are you",
                "what is your name",
                "your name",
                "who am i talking to",
                "introduce yourself",
                "what can you do",
                "help",
                "hi",
                "hello",
            ]
            if any(trigger in normalized for trigger in small_talk_triggers):
                return (
                    "I’m your Medical Document Assistant — an AI that analyzes PDFs, DOCX, TXT, CSV, and SOAP notes. "
                    "Upload one or more files, and I’ll retrieve relevant passages and answer your questions using a retrieval-augmented approach."
                )

            if not vectorstore:
                return "No documents loaded. Please upload some documents first."
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Retrieve relevant documents
            docs = retriever.invoke(query)
            
            if not docs:
                return "I couldn't find relevant information in the uploaded documents."
            
            # Create context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create system prompt
            system_prompt = f"""You are a Medical Document Assistant. Answer questions based on the provided context from medical documents.

CONTEXT:
{context}

IMPORTANT: Only answer based on the information provided in the context. If the answer is not in the context, say "I don't have enough information to answer this question based on the uploaded documents."

Answer the following question: {query}"""
            
            # Get response from LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error getting response: {str(e)}"

def main():
    # Check if all required packages are available
    try:
        import streamlit
        import langchain
        import langchain_openai
        import langchain_mistralai
        import langchain_community
        import langchain_chroma
        import chromadb
        print("✅ All required packages are available")
    except ImportError as e:
        st.error(f"❌ Missing required package: {str(e)}")
        st.error("Please run: python install_deps.py")
        st.stop()
    
    # Initialize the assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = MedicalDocumentAssistant()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: white; margin: 0;">🏥 Medical Assistant</h2>
            <p style="color: #bdc3c7; margin: 5px 0;">AI-Powered Document Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model status
        st.markdown("### 🤖 Model Status")
        if st.session_state.assistant.model_provider != "Unknown":
            st.success(f"✅ {st.session_state.assistant.model_provider}")
        else:
            st.error("❌ No model available")
        
        # File upload section
        st.markdown("### 📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose medical documents",
            type=['pdf', 'docx', 'txt', 'csv', 'soap'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, TXT, CSV, or SOAP files"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    all_documents = []
                    for file in uploaded_files:
                        docs = st.session_state.assistant.load_documents_from_file(file)
                        if docs:
                            all_documents.extend(docs)
                    
                    if all_documents:
                        st.session_state.vectorstore = st.session_state.assistant.create_vectorstore(all_documents)
                        st.session_state.uploaded_files = [f.name for f in uploaded_files]
                        st.success(f"✅ Processed {len(uploaded_files)} document(s)")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process documents")
        
        # Show uploaded files
        if st.session_state.uploaded_files:
            st.markdown("### 📋 Loaded Documents")
            for file_name in st.session_state.uploaded_files:
                st.markdown(f"📄 {file_name}")
        
        # Conversation history
        if st.session_state.conversation_history:
            st.markdown("### 💬 Recent Conversations")
            for i, conv in enumerate(st.session_state.conversation_history[-5:]):
                if st.button(f"📝 {conv['title'][:30]}...", key=f"hist_{i}"):
                    st.session_state.current_conversation = conv['messages']
                    st.rerun()
        
        # Clear conversation button
        if st.session_state.current_conversation:
            if st.button("🗑️ Clear Current Chat", type="secondary"):
                st.session_state.current_conversation = []
                st.rerun()
    
    # Main chat area with better layout
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px;">
        <h1 style="color: white; margin: 0; font-size: 3rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🏥 Medical Document Assistant</h1>
        <p style="color: white; font-size: 1.2rem; margin: 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">Ask questions about your medical documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container with better spacing
    if not st.session_state.current_conversation:
        # Welcome message when no conversation
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; background: rgba(255, 255, 255, 0.95); border-radius: 20px; margin: 20px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
            <h2 style="color: #333; margin-bottom: 20px;">👋 Welcome to Medical Document Assistant</h2>
            <p style="color: #666; font-size: 1.1rem; line-height: 1.6;">
                Upload your medical documents in the sidebar and start asking questions!<br>
                I can help you analyze PDFs, Word documents, text files, and more.
            </p>
            <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 15px; border-left: 4px solid #667eea;">
                <h3 style="color: #333; margin: 0 0 15px 0;">💡 Example Questions:</h3>
                <ul style="text-align: left; color: #666; line-height: 1.8;">
                    <li>What is the patient's diagnosis?</li>
                    <li>What are the patient's vital signs?</li>
                    <li>What medications were prescribed?</li>
                    <li>What is the treatment plan?</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Chat messages with better layout
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages with better spacing
        for i, message in enumerate(st.session_state.current_conversation):
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message">{message["content"]}</div>', unsafe_allow_html=True)
            
            # Add spacing between message pairs
            if i < len(st.session_state.current_conversation) - 1:
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Input area with better styling
    st.markdown('<div style="margin: 40px 20px; padding: 20px; background: rgba(255, 255, 255, 0.95); border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">', unsafe_allow_html=True)
    
    # Use a form to handle input submission properly
    with st.form(key="chat_form", clear_on_submit=True):
        # Query input with better layout
        col1, col2 = st.columns([5, 1])
        with col1:
            user_query = st.text_input(
                "Ask a question about your medical documents...",
                key="user_input",
                placeholder="e.g., What is the patient's diagnosis?",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process query
    if send_button and user_query and user_query.strip():
            if not st.session_state.vectorstore:
                st.error("❌ Please upload and process some documents first!")
            else:
                # Check if this is a duplicate question (check last 4 messages)
                recent_messages = st.session_state.current_conversation[-4:] if len(st.session_state.current_conversation) >= 4 else st.session_state.current_conversation
                is_duplicate = any(
                    msg["role"] == "user" and msg["content"] == user_query 
                    for msg in recent_messages
                )
                
                if is_duplicate:
                    st.warning("⚠️ You recently asked the same question. Please ask a different question.")
                else:
                    # Add user message to chat
                    st.session_state.current_conversation.append({
                        "role": "user",
                        "content": user_query,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Get AI response
                    with st.spinner("🤔 Thinking..."):
                        ai_response = st.session_state.assistant.get_response(
                            user_query, 
                            st.session_state.vectorstore
                        )
                    
                    # Add AI response to chat
                    st.session_state.current_conversation.append({
                        "role": "assistant",
                        "content": ai_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Save to conversation history
                    if len(st.session_state.current_conversation) >= 2:
                        conversation_title = user_query[:50] + "..." if len(user_query) > 50 else user_query
                        st.session_state.conversation_history.append({
                            "title": conversation_title,
                            "messages": st.session_state.current_conversation.copy(),
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    # Clear input and rerun
                    st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: white; opacity: 0.7;">
        <p>Powered by LangChain & Streamlit | Medical Document Analysis AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
