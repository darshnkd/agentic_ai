#!/usr/bin/env python3
"""
Helper script to set up environment variables and test API connection
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with the required environment variables."""
    env_path = Path(__file__).parent / ".env"
    
    if env_path.exists():
        print(f"✅ .env file already exists at: {env_path}")
        return True
    
    print("🔧 Creating .env file...")
    print("You can add either OpenAI API key (recommended) or Mistral AI API key.")
    print("\nFor OpenAI API key:")
    print("1. Go to https://platform.openai.com/api-keys")
    print("2. Sign up or log in")
    print("3. Create a new API key")
    print("4. Copy the key")
    
    print("\nFor Mistral AI API key:")
    print("1. Go to https://console.mistral.ai/")
    print("2. Sign up or log in")
    print("3. Navigate to API Keys section")
    print("4. Create a new API key")
    print("5. Copy the key")
    
    print("\n" + "="*50)
    print("RECOMMENDED: Start with OpenAI API key")
    print("The system will automatically fall back to Mistral AI if OpenAI fails")
    print("="*50)
    
    openai_api_key = input("\nEnter your OpenAI API key (or press Enter to skip): ").strip()
    mistral_api_key = input("Enter your Mistral AI API key (or press Enter to skip): ").strip()
    
    if not openai_api_key and not mistral_api_key:
        print("❌ No API keys provided. Please run the script again.")
        return False
    
    # Validate API keys
    if openai_api_key and not openai_api_key.startswith("sk-"):
        print("⚠️  Warning: OpenAI API key doesn't start with 'sk-'. Please verify it's correct.")
    
    if mistral_api_key and not mistral_api_key.startswith("mist-"):
        print("⚠️  Warning: Mistral AI API key doesn't start with 'mist-'. Please verify it's correct.")
    
    # Create .env file
    env_content = ""
    if openai_api_key:
        env_content += f"OPENAI_API_KEY={openai_api_key}\n"
    if mistral_api_key:
        env_content += f"MISTRAL_API_KEY={mistral_api_key}\n"
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print(f"✅ .env file created at: {env_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        print("✅ langchain-openai imported successfully")
    except ImportError as e:
        print(f"❌ langchain-openai import failed: {e}")
        print("Please install it with: pip install langchain-openai")
        return False
    
    try:
        from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        print("✅ langchain-mistralai imported successfully")
    except ImportError as e:
        print(f"❌ langchain-mistralai import failed: {e}")
        print("Please install it with: pip install langchain-mistralai")
        return False
    
    try:
        from langgraph.graph import StateGraph
        print("✅ langgraph imported successfully")
    except ImportError as e:
        print(f"❌ langgraph import failed: {e}")
        print("Please install it with: pip install langgraph")
        return False
    
    try:
        from langchain_chroma import Chroma
        print("✅ langchain-chroma imported successfully")
    except ImportError as e:
        print(f"❌ langchain-chroma import failed: {e}")
        print("Please install it with: pip install langchain-chroma")
        return False
    
    return True

def test_api_connections():
    """Test API connections for both providers."""
    print("\n🧪 Testing API connections...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    
    if not openai_api_key and not mistral_api_key:
        print("❌ No API keys found in .env file")
        return False
    
    success_count = 0
    
    # Test OpenAI
    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
            response = llm.invoke("Hello, this is a test message.")
            print("✅ OpenAI API connection successful!")
            success_count += 1
        except Exception as e:
            print(f"❌ OpenAI API connection failed: {str(e)}")
    
    # Test Mistral AI
    if mistral_api_key:
        try:
            from langchain_mistralai import ChatMistralAI
            llm = ChatMistralAI(model="mistral-large-latest", api_key=mistral_api_key)
            response = llm.invoke("Hello, this is a test message.")
            print("✅ Mistral AI API connection successful!")
            success_count += 1
        except Exception as e:
            print(f"❌ Mistral AI API connection failed: {str(e)}")
    
    if success_count == 0:
        print("❌ All API connections failed. Please check your API keys.")
        return False
    elif success_count == 1:
        print("⚠️  Only one API connection successful. The system will work but without fallback.")
    else:
        print("🎉 Multiple API connections successful! The system will have fallback capability.")
    
    return True

def main():
    print("🚀 Medical Document Assistant Setup")
    print("=" * 40)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Setup failed due to missing dependencies.")
        print("Please install the required packages and try again.")
        return
    
    # Create .env file
    if not create_env_file():
        print("\n❌ Setup failed.")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\nYou can now run the RAG agent with:")
    print("python RAG_Agent.py")
    
    # Test the setup
    if not test_api_connections():
        print("\n❌ Setup testing failed.")
        return
    
    print("\n🎉 Everything is ready! You can now run the RAG agent.")
    print("\n📝 Note: The system will automatically use OpenAI if available, otherwise fall back to Mistral AI.")

if __name__ == "__main__":
    main()
