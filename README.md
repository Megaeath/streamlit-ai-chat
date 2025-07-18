# My Custom AI Chat App

A secure, single-user Streamlit app for chatting with Gemini and ChatGPT, featuring provider selection, usage tracking, and login authentication.

## Features
- ChatGPT-style UI
- Switch between Gemini and ChatGPT
- Simulated free-tier usage tracking
- Secure login (username/password)
- All backend API calls handled server-side

## Setup
1. Clone the repo
2. Create a `.env` file from the example below and fill in your credentials and API keys
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Example `.env` file
```env
# Streamlit login credentials
STREAMLIT_USERNAME=your_username
STREAMLIT_PASSWORD=your_password

# API keys for providers
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional, for Claude
```

## Deployment
- Designed for Streamlit Community Cloud and local PC

## Requirements
- Python 3.9+
- OpenAI and Gemini API keys 