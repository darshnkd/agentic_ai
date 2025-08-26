# Detailed Setup Guide for Medical Document Assistant

## 🎯 Overview

This guide provides comprehensive setup instructions for the Medical Document Assistant RAG agent with dual API support (OpenAI + Mistral AI fallback).

## 🔑 Environment Variables Required

You need to create a `.env` file in the `LangGraph/Agents/` folder with your API keys:

```bash
# Create .env file in the Agents folder
cd LangGraph/Agents

# Option 1: OpenAI only (recommended for future integration)
echo "OPENAI_API_KEY=sk-your-openai-api-key-here" > .env

# Option 2: Mistral AI only (current working setup)
echo "MISTRAL_API_KEY=mist-your-mistral-api-key-here" > .env

# Option 3: Both (recommended for fallback capability)
echo "OPENAI_API_KEY=sk-your-openai-api-key-here" > .env
echo "MISTRAL_API_KEY=mist-your-mistral-api-key-here" >> .env
```

## 🔑 Getting Your API Keys

### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-`)

### Mistral AI API Key
1. Go to [Mistral AI Console](https://console.mistral.ai/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `mist-`)

## 🚀 Complete Setup Process

### Step 1: Install Dependencies
```bash
# Navigate to the project root
cd "AI Agents/LangGraph/AgentDocAssist"

# Option A: Install using requirements.txt
pip install -r LangGraph/requirements.txt

# Option B: Use the automatic installer (recommended)
cd LangGraph/Agents
python install_deps.py
```

### Step 2: Create Environment File
```bash
# Navigate to Agents folder
cd LangGraph/Agents

# Create .env file with your preferred API key(s)
# For testing with Mistral AI (current setup):
echo "MISTRAL_API_KEY=mist-your-actual-api-key-here" > .env
```

### Step 3: Run Setup Script
```bash
python setup_env.py
```

This script will:
- Test your API connections
- Validate your setup
- Ensure all dependencies are available
- Show which provider(s) are working

### Step 4: Test the Agent

#### Option A: Command Line Interface
```bash
python RAG_Agent.py
```

#### Option B: Beautiful Streamlit Web UI (Recommended)
```bash
streamlit run streamlit_app.py
```

The Streamlit app will open in your browser with a ChatGPT-like interface!

## 🧪 Testing Your Setup

### Test API Connection
```bash
cd LangGraph/Agents
python setup_env.py
```

Expected output:
```
🚀 Medical Document Assistant Setup
========================================
🔍 Testing imports...
✅ langchain-openai imported successfully
✅ langchain-mistralai imported successfully
✅ langgraph imported successfully
✅ langchain-chroma imported successfully

🔧 Creating .env file...
✅ .env file created at: /path/to/.env

🧪 Testing API connections...
✅ Mistral AI API connection successful!
🎉 Multiple API connections successful! The system will have fallback capability.

🎉 Everything is ready! You can now run the RAG agent.
```

### Test the RAG Agent

#### Command Line Version
```bash
python RAG_Agent.py
```

#### Streamlit Web UI
```bash
streamlit run streamlit_app.py
```

Expected output:
```
🚀 Using Mistral AI for LLM and embeddings
🔍 Testing Mistral AI API connection...
✅ API connection successful!
Test response: Hello! I'm here to help you with any questions or tasks you might have...

=== MEDICAL DOCUMENT ASSISTANT AGENT===
🚀 Starting the agent...

What is your question: 
```

## 🚨 Troubleshooting Common Issues

### 1. "No API Keys Found" Error
**Problem**: No API keys in `.env` file
**Solution**: 
```bash
cd LangGraph/Agents
echo "MISTRAL_API_KEY=mist-your-actual-key" > .env
```

### 2. "Authentication Error" 
**Problem**: Invalid or expired API key
**Solution**: 
- Verify your API key is correct
- Check if your account is active
- Ensure billing is set up

### 3. "Rate Limit Exceeded"
**Problem**: API quota exceeded
**Solution**: 
- Wait a few minutes and try again
- Check your account limits
- Consider upgrading your plan

### 4. "Import Error" for langchain packages
**Problem**: Missing dependencies
**Solution**: 
```bash
pip install -r ../requirements.txt
```

### 5. "File Not Found" Error
**Problem**: Missing medical document
**Solution**: 
- Ensure `Medical_Document_Input.pdf` exists in the Agents folder
- Or update `doc_path` variable in `RAG_Agent.py`

## 🔧 Advanced Configuration

### Customizing Models
Edit `RAG_Agent.py` to change models:

```python
# OpenAI models
llm = ChatOpenAI(
    model="gpt-4o",  # Change to gpt-4o or gpt-3.5-turbo
    temperature=0
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"  # Change to text-embedding-3-small
)

# Mistral AI models
llm = ChatMistralAI(
    model="mistral-medium",  # Change to mistral-small or mistral-large
    temperature=0
)
```

### Customizing Document Path
```python
# In RAG_Agent.py, change this line:
doc_path = os.path.join(SCRIPT_DIR, "your_document.pdf")
```

### Customizing Chunking
```python
# In RAG_Agent.py, modify these parameters:
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,    # Increase for longer chunks
    chunk_overlap=300   # Increase for more overlap
)
```

## 📊 Performance Monitoring

### Check Vector Store
```bash
# The system creates a ChromaDB store in chroma_store/
ls -la chroma_store/
```

### Monitor API Usage
- Check your provider's dashboard for usage statistics
- Monitor rate limits and quotas
- Track response times and quality

## 🔮 Future Enhancements

The system is designed for easy extension:

1. **Add New Providers**: Integrate Anthropic Claude, Google Gemini, etc.
2. **Local Models**: Add support for Ollama, local LLMs
3. **Advanced RAG**: Implement hybrid search, re-ranking
4. **Multi-Modal**: Support for images, audio, video
5. **Web Interface**: Add a simple web UI for document uploads

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run setup script** to diagnose problems
3. **Check error messages** for specific guidance
4. **Verify API keys** are correct and active
5. **Test with simple queries** first

## ✅ Success Checklist

- [ ] Dependencies installed
- [ ] `.env` file created with valid API key
- [ ] Setup script runs successfully
- [ ] API connection test passes
- [ ] RAG agent starts without errors
- [ ] Can ask questions and get responses
- [ ] Document retrieval works properly

---

**🎯 Goal**: Get you up and running with a fully functional medical document assistant that can seamlessly switch between AI providers for maximum reliability.
