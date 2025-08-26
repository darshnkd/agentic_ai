# Medical Document Assistant - RAG Agent

An intelligent Retrieval-Augmented Generation (RAG) agent tailored for medical documents with automatic fallback between OpenAI and Mistral AI APIs.

## 🚀 Features

- **🔄 Dual API Support**: Automatically uses OpenAI API if available, falls back to Mistral AI
- **📚 Document Support**: PDF, DOCX, SOAP notes, and CSV files
- **🏥 Medical Focus**: Optimized for clinical and biomedical document processing
- **🧠 ChromaDB Integration**: Persistent vector storage for document retrieval
- **⚡ Smart Fallback**: Seamless switching between providers for reliability

## 📁 Project Structure

```
AgentDocAssist/
├── README.md                    # This file - main documentation
├── LangGraph/
│   ├── Agents/
│   │   ├── RAG_Agent.py        # Main RAG agent with fallback logic
│   │   ├── streamlit_app.py    # Beautiful ChatGPT-like web UI
│   │   ├── SETUP.md            # Detailed setup instructions
│   │   ├── setup_env.py        # Setup and testing script
│   │   ├── Medical_Document_Input.pdf  # Example document
│   │   ├── chroma_store/       # Vector database storage
│   │   └── venv/               # Virtual environment
│   └── requirements.txt         # Python dependencies
└── .venv/                       # Root virtual environment
```

## 🛠️ Quick Start

### 1. Install Dependencies

```bash
# Navigate to the project directory
cd "AI Agents/LangGraph/AgentDocAssist"

# Install dependencies
pip install -r LangGraph/requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the `LangGraph/Agents/` folder:

```bash
cd LangGraph/Agents

# Create .env file with your API keys
echo "OPENAI_API_KEY=sk-your-openai-api-key-here" > .env
echo "MISTRAL_API_KEY=mist-your-mistral-api-key-here" >> .env
```

**Note**: You can set either one or both. If both are set, the system will try OpenAI first, then fall back to Mistral AI.

### 3. Run Setup Script

```bash
python setup_env.py
```

This script will test your API connections and validate your setup.

### 4. Choose Your Interface

#### Option A: Command Line Interface
```bash
python RAG_Agent.py
```

#### Option B: Beautiful Streamlit Web UI (Recommended)
```bash
cd LangGraph/Agents
streamlit run streamlit_app.py
```

The Streamlit app provides a ChatGPT-like interface with:
- 📁 Document upload and processing
- 💬 Chat interface for asking questions
- 📋 Conversation history
- 🎨 Modern, responsive design
- 📱 Mobile-friendly interface

## 🔑 API Key Sources

- **OpenAI**: https://platform.openai.com/api-keys
- **Mistral AI**: https://console.mistral.ai/

## 🤖 Models Used

### OpenAI (Primary)
- **LLM**: gpt-4o-mini (configurable to gpt-4o or gpt-3.5-turbo)
- **Embeddings**: text-embedding-3-small (configurable to text-embedding-3-large)

### Mistral AI (Fallback)
- **LLM**: mistral-large-latest
- **Embeddings**: mistral-embed

## 🔄 How the Fallback System Works

1. **Primary Choice**: The system first attempts to connect to OpenAI API
2. **Automatic Fallback**: If OpenAI fails, it automatically switches to Mistral AI
3. **Seamless Operation**: Users don't need to manually switch - it happens automatically
4. **Error Handling**: Clear error messages indicate which provider is being used

## 📖 Usage

1. Place your medical documents in the `LangGraph/Agents/` folder
2. Update the `doc_path` variable in `RAG_Agent.py` if needed
3. Run the agent and ask questions about your documents
4. The system will automatically retrieve relevant information and provide answers

## 🚨 Troubleshooting

### Common Issues

1. **No API Keys**: Ensure at least one API key is set in your `.env` file
2. **Authentication Errors**: Verify your API keys are correct and active
3. **Rate Limits**: Wait a moment and try again
4. **Quota Exceeded**: Check your account limits with the respective provider

### Testing Connections

Run the setup script to test your API connections:

```bash
cd LangGraph/Agents
python setup_env.py
```

This will verify both providers and show which one(s) are working.

## 🔮 Future Integration

The system is designed to easily integrate additional AI providers in the future. The fallback logic can be extended to support other services like:
- Anthropic Claude
- Google Gemini
- Local models (Ollama, etc.)

## 📚 Detailed Setup

For comprehensive setup instructions, troubleshooting, and advanced configuration, see:
- **`LangGraph/Agents/SETUP.md`** - Detailed setup guide
- **`LangGraph/Agents/setup_env.py`** - Interactive setup script

## ⚡ Requirements

Core dependencies:
- langgraph
- langchain
- langchain_openai
- langchain_mistralai
- langchain_community
- chromadb
- langchain_chroma
- python-docx
- PyMuPDF
- pypdf

Install all dependencies with:
```bash
pip install -r LangGraph/requirements.txt
```

---

**🎯 Goal**: Create a clean, maintainable RAG agent that can seamlessly switch between AI providers for maximum reliability and future flexibility.
