"""
Medical Document Assistant

An intelligent Retrieval-Augmented Generation (RAG) agent tailored for medical documents.
It supports ingesting and retrieving from common clinical formats including PDF, DOCX,
SOAP notes (treated as plain text), and CSV.
"""

from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.documents import Document
import csv

# Resolve script directory and load .env from the Agents folder explicitly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(SCRIPT_DIR, ".env"))

# Initialize LLM and embeddings with fallback logic
def initialize_models():
    """Initialize LLM and embeddings with OpenAI as primary, Mistral AI as fallback."""
    llm = None
    embeddings = None
    model_provider = "Unknown"
    
    # Try OpenAI first
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            print("🔍 Testing OpenAI API connection...")
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # You can change this to gpt-4o or gpt-3.5-turbo
                temperature=0,
                api_key=openai_api_key
            )
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",  # You can change this to text-embedding-3-large
                api_key=openai_api_key
            )
            
            # Test the connection
            test_messages = [HumanMessage(content="Hello, this is a test message.")]
            response = llm.invoke(test_messages)
            print("✅ OpenAI API connection successful!")
            model_provider = "OpenAI"
            
        except Exception as e:
            print(f"❌ OpenAI API connection failed: {str(e)}")
            print("🔄 Falling back to Mistral AI...")
            llm = None
            embeddings = None
    
    # Fallback to Mistral AI
    if not llm or not embeddings:
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if mistral_api_key:
            try:
                print("🔍 Testing Mistral AI API connection...")
                llm = ChatMistralAI(
                    model="mistral-large-latest",
                    temperature=0,
                    api_key=mistral_api_key
                )
                embeddings = MistralAIEmbeddings(
                    model="mistral-embed",
                    api_key=mistral_api_key
                )
                
                # Test the connection
                test_messages = [HumanMessage(content="Hello, this is a test message.")]
                response = llm.invoke(test_messages)
                print("✅ Mistral AI API connection successful!")
                model_provider = "Mistral AI"
                
            except Exception as e:
                print(f"❌ Mistral AI API connection failed: {str(e)}")
                raise Exception("Both OpenAI and Mistral AI API connections failed. Please check your API keys and try again.")
        else:
            raise Exception("No API keys found. Please set either OPENAI_API_KEY or MISTRAL_API_KEY in your .env file.")
    
    return llm, embeddings, model_provider

# Initialize models with fallback
try:
    llm, embeddings, model_provider = initialize_models()
    print(f"🚀 Using {model_provider} for LLM and embeddings")
except Exception as e:
    print(f"❌ Failed to initialize models: {e}")
    print("Please check your .env file and API keys.")
    exit(1)


doc_path = os.path.join(SCRIPT_DIR, "Medical_Document_Input.pdf")  # Default relative to this file
print(f"DEBUG: Resolved document path: {doc_path}")


def load_documents_from_path(file_path: str):
    """Return a list of LangChain Documents loaded from the given file path.

    Supported formats: .pdf, .docx, .soap (as text), .csv
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    if ext == ".docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()
    if ext == ".soap":
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    if ext == ".csv":
        lines = []
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                lines.append(", ".join(row))
        content = "\n".join(lines)
        return [Document(page_content=content, metadata={"source": os.path.basename(file_path)})]

    raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .docx, .soap, .csv")

# Safety measure I have put for debugging purposes :)
if not os.path.exists(doc_path):
    raise FileNotFoundError(f"Input file not found: {doc_path}")

# Checks if the document is there
try:
    pages = load_documents_from_path(doc_path)
    print(f"Document has been loaded and has {len(pages)} chunk(s)")
except Exception as e:
    print(f"Error loading document: {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

persist_directory = os.path.join(SCRIPT_DIR, "chroma_store")
collection_name = "medical_documents"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    Retrieve grounded context from the ingested medical document(s)
    to answer clinical and biomedical questions.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the provided medical document(s)."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

# Debug: Check if tools are properly defined
print(f"DEBUG: Available tools: {[tool.name for tool in tools]}")

# Bind tools the standard way (works with mistral-large for native tool calls)
try:
    llm = llm.bind_tools(tools)
    print("DEBUG: Tools bound via bind_tools")
except Exception as e2:
    print(f"DEBUG: bind_tools failed: {e2}")
    print("DEBUG: Will rely on deterministic fallback retrieval")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are a Medical Document Assistant. Answer questions strictly based on the ingested medical documents
(PDF, DOCX, SOAP/txt, CSV). 

IMPORTANT: You MUST use the retriever_tool to search the medical documents before answering any questions.
The retriever_tool is available and will return relevant information from your knowledge base.

Example usage:
- When asked about patient information, use retriever_tool with the query
- When asked about diseases, use retriever_tool to search the documents
- Always call the tool first, then answer based on the retrieved information

If you cannot find the information using the retriever_tool, say "I do not have sufficient information in the provided documents."
"""


tools_dict = {tool.name: tool for tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    print(f"DEBUG: Calling LLM with {len(messages)} messages")
    try:
        message = llm.invoke(messages)
        print(f"DEBUG: LLM response has tool_calls: {hasattr(message, 'tool_calls')}")
        if hasattr(message, 'tool_calls'):
            print(f"DEBUG: Number of tool calls: {len(message.tool_calls)}")
        return {'messages': [message]}
    except Exception as e:
        # Handle provider errors (e.g., rate limit 429) by deterministic RAG fallback
        print(f"DEBUG: LLM invoke error: {e}. Falling back to deterministic retrieval.")
        # Get last human query
        user_query = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_query = m.content
                break
        if not user_query:
            return {'messages': [AIMessage(content="I encountered an error and could not identify the question.")]} 
        try:
            retrieved_context = retriever_tool.invoke(user_query)
        except Exception:
            retrieved_context = ""
        if retrieved_context and "no relevant information" not in retrieved_context.lower():
            fallback_answer = answer_with_context(user_query, retrieved_context)
        else:
            fallback_answer = "I do not have sufficient information in the provided documents."
        return {'messages': [AIMessage(content=fallback_answer)]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


def answer_with_context(query: str, context: str) -> str:
    """Deterministic RAG answering: force the model to answer using provided context only."""
    qa_system_prompt = (
        "You are a medical RAG assistant. Answer ONLY using the provided CONTEXT. "
        "If the answer is not present in the CONTEXT, say: 'I do not have sufficient information in the provided documents.' "
        "Respond concisely and extract explicit fields when present (e.g., Patient Name, Age, Disease)."
    )
    messages = [
        SystemMessage(content=qa_system_prompt),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer succinctly."),
    ]
    try:
        resp = llm.invoke(messages)
        return resp.content
    except Exception as e:
        print(f"DEBUG: Error in answer_with_context: {str(e)}")
        print(f"DEBUG: Error type: {type(e).__name__}")
        # Check for specific error types
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            if model_provider == "OpenAI":
                return "Authentication error: Please check your OpenAI API key in the .env file."
            else:
                return "Authentication error: Please check your Mistral AI API key in the .env file."
        elif "rate_limit" in str(e).lower() or "429" in str(e):
            return "Rate limit exceeded: Please wait a moment before trying again."
        elif "quota" in str(e).lower():
            if model_provider == "OpenAI":
                return "API quota exceeded: Please check your OpenAI account limits."
            else:
                return "API quota exceeded: Please check your Mistral AI account limits."
        else:
            return f"Error while processing your request: {str(e)}"


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def test_api_connection():
    """Test the API connection before running the main agent."""
    print(f"Testing {model_provider} API connection...")
    try:
        # Simple test message
        test_messages = [HumanMessage(content="Hello, this is a test message.")]
        response = llm.invoke(test_messages)
        print("✅ API connection successful!")
        print(f"Test response: {response.content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Provide specific guidance based on error type
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            print(f"\n🔑 AUTHENTICATION ERROR:")
            if model_provider == "OpenAI":
                print("Please check your OPENAI_API_KEY in the .env file")
            else:
                print("Please check your MISTRAL_API_KEY in the .env file")
        elif "rate_limit" in str(e).lower() or "429" in str(e):
            print("\n⏱️ RATE LIMIT ERROR:")
            print("Please wait a moment and try again")
        elif "quota" in str(e).lower():
            print("\n💰 QUOTA ERROR:")
            if model_provider == "OpenAI":
                print("Please check your OpenAI account limits")
            else:
                print("Please check your Mistral AI account limits")
        else:
            print("\n❓ UNKNOWN ERROR:")
            print("Please check your internet connection and try again")
        
        return False


def running_agent():
    print("\n=== MEDICAL DOCUMENT ASSISTANT AGENT===")
    
    # Test API connection first
    if not test_api_connection():
        print("\n❌ Cannot proceed without a working API connection.")
        print("Please fix the issue above and try again.")
        return
    
    print("\n🚀 Starting the agent...")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        # First, try via the graph (tool calling if supported by model)
        result = rag_agent.invoke({"messages": messages})

        # Fallback: if the model didn't call tools, or returned generic message,
        # do deterministic RAG: retrieve context -> answer with context only.
        try:
            retrieved_context = retriever_tool.invoke(user_input)
        except Exception as _e:
            retrieved_context = ""

        final_answer = result['messages'][-1].content
        if hasattr(result['messages'][-1], 'tool_calls') and len(getattr(result['messages'][-1], 'tool_calls', [])) == 0:
            print("DEBUG: LLM emitted 0 tool calls -> using fallback retrieval")
        if retrieved_context and "no relevant information" not in retrieved_context.lower():
            final_answer = answer_with_context(user_input, retrieved_context)

        print("\n=== ANSWER ===")
        print(final_answer)


running_agent()