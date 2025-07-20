# My Custom AI Chat App

A secure, single-user Streamlit app for chatting with Gemini, ChatGPT, Groq (Llama, Mixtral, Gemma, and more), featuring provider selection, usage tracking, and login authentication.

## Features
- ChatGPT-style UI
- Switch between Gemini, ChatGPT, Groq (Llama, Mixtral, Gemma, etc.), and Claude
- Simulated free-tier usage tracking per model
- Secure login (username/password)
- All backend API calls handled server-side
- **Beautiful markdown and code rendering** in chat (code blocks, lists, bold, etc.)
- **Export chat** as .txt or .json
- **Logout** button for security

## Supported Groq Models (Free)
- Llama 3 70B, Llama 3 8B, Llama 3.1 8B Instant, Llama 3.3 70B Versatile
- Llama 2 70B, Llama 2 13B, Llama 2 7B
- Mixtral 8x7B, Mistral Saba 24B
- Gemma 7B, Gemma2 9B IT
- Allam 2 7B, Compound Beta, Compound Beta Mini
- DeepSeek R1 Distill Llama 70B
- Meta Llama 4 Maverick 17B 128E Instruct, Meta Llama 4 Scout 17B 16E Instruct
- Meta Llama Guard 4 12B, Meta Llama Prompt Guard 2 22M, Meta Llama Prompt Guard 2 86M
- MoonshotAI Kimi K2 Instruct, Qwen Qwen3 32B

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
GROQ_API_KEY=your_groq_api_key            # For Groq models
```

## Deployment
- Designed for Streamlit Community Cloud and local PC

## Requirements
- Python 3.9+
- OpenAI, Gemini, Groq API keys (free tier supported) 