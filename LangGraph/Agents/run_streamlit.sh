#!/bin/bash

# Medical Document Assistant - Streamlit Launcher
echo "🚀 Launching Medical Document Assistant..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python setup_env.py first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if python-dotenv is installed
if ! python -c "import dotenv" 2>/dev/null; then
    echo "❌ python-dotenv not found. Installing..."
    pip install python-dotenv
fi

# Launch Streamlit
echo "🌐 Starting Streamlit app..."
echo "📱 Open your browser to: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app.py
