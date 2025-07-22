import streamlit as st
import os
import json
from datetime import datetime, date, timedelta
import html
import re

# Authentication
USERNAME = st.secrets["STREAMLIT_USERNAME"]
PASSWORD = st.secrets["STREAMLIT_PASSWORD"]

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except:
        return {"providers": {}}

CONFIG = load_config()

# Initialize AI clients (with error handling)
OPENAI_AVAILABLE = False
GEMINI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False
GROQ_AVAILABLE = False
OPENROUTER_AVAILABLE = False

try:
    import openai
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if OPENAI_API_KEY:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_AVAILABLE = True
except:
    pass

try:
    import google.generativeai as genai
    if st.secrets.get("GEMINI_API_KEY"):
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        GEMINI_AVAILABLE = True
except:
    pass

try:
    import anthropic
    if st.secrets.get("ANTHROPIC_API_KEY"):
        anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        ANTHROPIC_AVAILABLE = True
except:
    pass

try:
    import groq
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    if GROQ_API_KEY:
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
        GROQ_AVAILABLE = True
except Exception as e:
    GROQ_AVAILABLE = False

try:
    # OpenRouter uses the OpenAI SDK
    import openai
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    if OPENROUTER_API_KEY:
        openrouter_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        OPENROUTER_AVAILABLE = True
except:
    pass

# Usage tracking
USAGE_FILE = 'usage_tracker.json'
LAST_RESET_FILE = 'last_reset.json'

def load_usage():
    """Load usage data from file"""
    try:
        with open(USAGE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_usage(usage_data):
    """Save usage data to file"""
    with open(USAGE_FILE, 'w') as f:
        json.dump(usage_data, f)

def load_last_reset():
    """Load last reset timestamp"""
    try:
        with open(LAST_RESET_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_last_reset(reset_data):
    """Save last reset timestamp"""
    with open(LAST_RESET_FILE, 'w') as f:
        json.dump(reset_data, f)

def check_and_reset_daily_usage():
    """Check if 24 hours have passed and reset usage if needed"""
    current_time = datetime.now()
    last_reset_data = load_last_reset()
    usage_data = load_usage()
    
    # Check if we need to reset (24 hours passed)
    should_reset = False
    
    if not last_reset_data:
        # First time running, set initial reset time
        should_reset = True
    else:
        last_reset_str = last_reset_data.get('last_reset')
        if last_reset_str:
            last_reset = datetime.fromisoformat(last_reset_str)
            time_diff = current_time - last_reset
            if time_diff.total_seconds() >= 24 * 3600:  # 24 hours in seconds
                should_reset = True
    
    if should_reset:
        # Reset all usage to 0
        for provider in usage_data:
            for model_name in usage_data[provider]:
                usage_data[provider][model_name] = {}
        
        # Save reset usage data
        save_usage(usage_data)
        
        # Update last reset timestamp
        last_reset_data['last_reset'] = current_time.isoformat()
        save_last_reset(last_reset_data)
        
        return True
    
    return False

def get_next_reset_time():
    """Get the next reset time (24 hours from last reset)"""
    last_reset_data = load_last_reset()
    
    if not last_reset_data or 'last_reset' not in last_reset_data:
        # If no reset data, next reset is in 24 hours from now
        next_reset = datetime.now() + timedelta(hours=24)
        return next_reset
    
    last_reset_str = last_reset_data.get('last_reset')
    if last_reset_str:
        last_reset = datetime.fromisoformat(last_reset_str)
        next_reset = last_reset + timedelta(hours=24)
        return next_reset
    
    return datetime.now() + timedelta(hours=24)

def get_usage_for_model(provider, model_name):
    """Get current usage for a specific model"""
    usage_data = load_usage()
    today = date.today().isoformat()
    
    if provider not in usage_data:
        usage_data[provider] = {}
    
    if model_name not in usage_data[provider]:
        usage_data[provider][model_name] = {}
    
    if today not in usage_data[provider][model_name]:
        usage_data[provider][model_name][today] = 0
    
    return usage_data[provider][model_name][today], usage_data

def increment_usage(provider, model_name, token_count=1):
    """Increment usage for a specific model (tokens or requests)"""
    current_usage, usage_data = get_usage_for_model(provider, model_name)
    today = date.today().isoformat()
    usage_data[provider][model_name][today] = current_usage + token_count
    save_usage(usage_data)

def get_available_models():
    """Get list of available models from config"""
    available_models = []
    for provider, models in CONFIG['providers'].items():
        for model_name, model_config in models.items():
            if model_config.get('enabled', True):
                available_models.append({
                    'provider': provider,
                    'model_name': model_name,
                    'display_name': f"{provider} - {model_name}",
                    'config': model_config
                })
    return available_models

def get_ai_response(selected_model, message, chat_history):
    """Get response from selected AI model and return (text, token_count)"""
    try:
        provider = selected_model['provider']
        model_config = selected_model['config']
        model_id = model_config['model']
        
        if provider == 'Gemini' and GEMINI_AVAILABLE:
            # Build context from chat history
            context = ""
            for msg in chat_history[-5:] if chat_history else []:
                if msg['role'] == 'user':
                    context += f"User: {msg['content']}\n"
                else:
                    context += f"Assistant: {msg['content']}\n"
            
            # Generate response
            gemini_model = genai.GenerativeModel(model_id)
            response = gemini_model.generate_content(f"{context}User: {message}")
            
            # Extract token count from Gemini response
            token_count = 0
            if hasattr(response, 'usage_metadata'):
                token_count = response.usage_metadata.prompt_token_count + response.usage_metadata.candidates_token_count
            
            return response.text, token_count
            
        elif provider == 'OpenAI' and OPENAI_AVAILABLE:
            # Build messages for OpenAI
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for msg in chat_history[-5:] if chat_history else []:
                messages.append({"role": msg['role'], "content": msg['content']})
            messages.append({"role": "user", "content": message})
            
            # Generate response
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=500
            )
            
            # Extract token count from OpenAI response
            token_count = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            return response.choices[0].message.content, token_count
        elif provider == 'OpenRouter' and OPENROUTER_AVAILABLE:
            # OpenRouter logic
            messages = []
            for msg in chat_history[-5:] if chat_history else []:
                role = 'user' if msg['role'] == 'user' else 'assistant'
                messages.append({"role": role, "content": msg['content']})
            messages.append({"role": "user", "content": message})
            
            # Generate response
            response = openrouter_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=500
            )
            
            # Extract token count from OpenRouter response (same as OpenAI)
            token_count = 0
            if hasattr(response, 'usage') and response.usage is not None:
                token_count = response.usage.total_tokens
            
            return response.choices[0].message.content, token_count
            
        elif provider == 'Anthropic' and ANTHROPIC_AVAILABLE:
            # Build messages for Anthropic
            messages = []
            for msg in chat_history[-5:] if chat_history else []:
                messages.append({"role": msg['role'], "content": msg['content']})
            messages.append({"role": "user", "content": message})
            
            # Generate response
            response = anthropic_client.messages.create(
                model=model_id,
                messages=messages,
                max_tokens=500
            )
            
            # Extract token count from Anthropic response
            token_count = 0
            if hasattr(response, 'usage'):
                token_count = response.usage.input_tokens + response.usage.output_tokens
            
            return response.content[0].text, token_count
            
        elif provider == 'Groq' and GROQ_AVAILABLE:
            # Groq logic
            messages = []
            for msg in chat_history[-5:] if chat_history else []:
                role = 'user' if msg['role'] == 'user' else 'assistant'
                messages.append({"role": role, "content": msg['content']})
            messages.append({"role": "user", "content": message})
            
            # Generate response
            response = groq_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=500
            )
            
            # Extract token count from Groq response
            token_count = 0
            if hasattr(response, 'usage'):
                token_count = response.usage.prompt_tokens + response.usage.completion_tokens
            
            return response.choices[0].message.content, token_count
            
        else:
            return f"API not available for {provider}. Please check your API keys in secrets.toml file.", 0
            
    except Exception as e:
        return f"Error: {str(e)}", 0

def check_usage_limit(selected_model):
    """Check if usage limit is reached for the selected model"""
    provider = selected_model['provider']
    model_name = selected_model['model_name']
    model_config = selected_model['config']
    
    current_usage, _ = get_usage_for_model(provider, model_name)
    limit = model_config['limit']
    limit_type = model_config['limit_type']
    
    if limit_type == 'per_day':
        return current_usage < limit
    elif limit_type == 'per_month':
        # For monthly limits, check if we're in a new month
        today = date.today()
        first_day = today.replace(day=1)
        days_in_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1) - first_day
        return current_usage < limit
    elif limit_type == 'per_hour':
        # For hourly limits, check current hour
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        # This would need more complex tracking for hourly limits
        return current_usage < limit
    
    return True

# --- Session State Initialization ---
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = 0
if 'input_disabled' not in st.session_state:
    st.session_state['input_disabled'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'selected_model_name' not in st.session_state:
    st.session_state['selected_model_name'] = None

# --- Login Function ---
def login():
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title('🔐 Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login', use_container_width=True):
        if username == USERNAME and password == PASSWORD:
            # Check and reset daily usage if 24 hours have passed
            if check_and_reset_daily_usage():
                st.success('✅ Daily usage reset! All token usage has been reset to 0.')
            
            st.session_state['authenticated'] = True
            st.success('Logged in!')
        else:
            st.error('Invalid credentials')
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if not st.session_state['authenticated']:
    login()

# Main App
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-container {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.usage-card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.model-selector {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🤖 My Custom AI Chat App</h1></div>', unsafe_allow_html=True)

# Model Selector
st.markdown('<div class="model-selector">', unsafe_allow_html=True)
st.subheader('🔧 Model Configuration')

available_models = get_available_models()
if available_models:
    model_options = [model['display_name'] for model in available_models]
    prev_model = st.session_state.get('selected_model_name')
    selected_model_name = st.selectbox('Choose AI Model:', model_options, key='model_select')
    st.session_state['selected_model_name'] = selected_model_name
    # If the model changed, reset input_disabled and input_key
    if prev_model != selected_model_name:
        st.session_state['input_disabled'] = False
        st.session_state['input_key'] += 1
    # Get selected model config
    selected_model = next(model for model in available_models if model['display_name'] == selected_model_name)
    # Show model details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Provider:** {selected_model['provider']}")
    with col2:
        st.write(f"**Model:** {selected_model['model_name']}")
    with col3:
        st.write(f"**Limit:** {selected_model['config']['limit']} {selected_model['config']['limit_type']}")
    # Show API availability
    provider = selected_model['provider']
    if provider == 'Gemini':
        if GEMINI_AVAILABLE:
            st.success('✅ Gemini API available')
        else:
            st.error('❌ Gemini API not available - check GEMINI_API_KEY in secrets.toml')
    elif provider == 'OpenAI':
        if OPENAI_AVAILABLE:
            st.success('✅ OpenAI API available')
        else:
            st.error('❌ OpenAI API not available - check OPENAI_API_KEY in secrets.toml')
    elif provider == 'Anthropic':
        if ANTHROPIC_AVAILABLE:
            st.success('✅ Anthropic API available')
        else:
            st.error('❌ Anthropic API not available - check ANTHROPIC_API_KEY in secrets.toml')
    elif provider == 'Groq':
        if GROQ_AVAILABLE:
            st.success('✅ Groq API available')
        else:
            st.error('❌ Groq API not available - check GROQ_API_KEY in secrets.toml')
    elif provider == 'OpenRouter':
        if OPENROUTER_AVAILABLE:
            st.success('✅ OpenRouter API available')
        else:
            st.error('❌ OpenRouter API not available - check OPENROUTER_API_KEY in secrets.toml')
else:
    st.error('No models configured. Please check config.json')
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# --- Usage Tracker Implementation ---
st.markdown('<div class="usage-card">', unsafe_allow_html=True)
st.subheader('📊 Usage Tracker')

current_usage, _ = get_usage_for_model(selected_model['provider'], selected_model['model_name'])
limit = selected_model['config']['limit']
limit_unit = selected_model['config'].get('limit_unit', 'requests')
usage_percentage = (current_usage / limit) * 100

# Progress bar with color coding
if usage_percentage < 70:
    color = "green"
elif usage_percentage < 90:
    color = "orange"
else:
    color = "red"

st.progress(usage_percentage / 100)
st.write(f"**{selected_model['display_name']}:** {current_usage:,}/{limit:,} {limit_unit} used ({usage_percentage:.1f}%)")

# Show next reset time
next_reset = get_next_reset_time()
time_until_reset = next_reset - datetime.now()
hours_until_reset = time_until_reset.total_seconds() / 3600

if hours_until_reset > 0:
    st.info(f"🔄 Next daily reset in {hours_until_reset:.1f} hours ({next_reset.strftime('%Y-%m-%d %H:%M')})")
else:
    st.success("✅ Daily reset available! Login again to reset usage.")

if usage_percentage >= 90:
    st.warning(f"⚠️ You're close to your {selected_model['model_name']} {limit_unit} limit!")
elif usage_percentage >= 70:
    st.info(f"ℹ️ You've used {usage_percentage:.1f}% of your {selected_model['model_name']} {limit_unit} limit")

st.markdown('</div>', unsafe_allow_html=True)

# --- Enhanced Chat UI ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.subheader('💬 Chat')

# Display chat history as enhanced bubbles
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state['chat_history']):
        content = msg["content"]
        is_code_block = bool(re.match(r"^```[a-zA-Z0-9]*\\n", content)) or content.strip().startswith('```')
        if msg['role'] == 'user':
            st.markdown("""
            <div style="text-align: right; margin-bottom: 12px;">
            """, unsafe_allow_html=True)
            if is_code_block:
                # Remove triple backticks and optional language
                code = re.sub(r"^```[a-zA-Z0-9]*\\n|```$", "", content.strip(), flags=re.MULTILINE)
                st.code(code)
            else:
                st.markdown(content)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: left; margin-bottom: 12px;">
            """, unsafe_allow_html=True)
            if is_code_block:
                code = re.sub(r"^```[a-zA-Z0-9]*\\n|```$", "", content.strip(), flags=re.MULTILINE)
                st.code(code)
            else:
                st.markdown(content)
            st.markdown("</div>", unsafe_allow_html=True)

# Enhanced user input
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input('Type your message:', key=f'user_input_{st.session_state["input_key"]}', placeholder="Ask me anything...", disabled=st.session_state['input_disabled'])
with col2:
    send_button = st.button('Send', key='send_btn', use_container_width=True, disabled=st.session_state['input_disabled'])

# Autofocus the input if enabled
if not st.session_state['input_disabled']:
    st.markdown(f"""
    <script>
    setTimeout(function() {{
        var input = window.parent.document.querySelector('input[data-testid="stTextInput"]');
        if(input) {{ input.focus(); }}
    }}, 100);
    </script>
    """, unsafe_allow_html=True)

send_triggered = False
if (send_button or (user_input and user_input.strip() and not send_button)) and not st.session_state['input_disabled']:
    send_triggered = True

if send_triggered and user_input.strip():
    st.session_state['input_disabled'] = True
    if not check_usage_limit(selected_model):
        st.error(f"❌ You've reached your {selected_model['model_name']} limit. Please try another model or wait until the limit resets.")
        st.session_state['input_disabled'] = False
    else:
        # Add user message
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        # Get real AI response
        with st.spinner('🤔 Thinking...'):
            ai_response, token_count = get_ai_response(selected_model, user_input, st.session_state['chat_history'])
        # Increment usage with token count
        increment_usage(selected_model['provider'], selected_model['model_name'], token_count)
        # Enhanced streaming effect
        displayed = ''
        response_placeholder = st.empty()
        for char in ai_response:
            displayed += char
            response_placeholder.markdown(f"""
            <div style=\"text-align: left; margin-bottom: 12px;\">
                <span style=\"display: inline-block; background: white; color: #333; \
                padding: 10px 16px; border-radius: 18px; max-width: 70%; \
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e0e0e0;\">{html.escape(displayed)}</span>
            </div>
            """, unsafe_allow_html=True)
            import time
            time.sleep(0.02)
        st.session_state['chat_history'].append({'role': 'ai', 'content': ai_response})
        # Clear input by incrementing the key
        st.session_state['input_key'] += 1
        st.session_state['input_disabled'] = False
        st.rerun() 

# Block all UI with overlay if input is disabled (AI is thinking)
if st.session_state.get('input_disabled', False):
    st.markdown('''
    <style>
    .block-ui-overlay {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(255,255,255,0.7);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .block-ui-spinner {
        border: 8px solid #f3f3f3;
        border-top: 8px solid #667eea;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    <div class="block-ui-overlay">
        <div>
            <div class="block-ui-spinner"></div>
            <div style="margin-top: 1.5rem; text-align: center; font-size: 1.2rem; color: #333;">AI is thinking...<br>Please wait</div>
        </div>
    </div>
    ''', unsafe_allow_html=True) 

# --- Sidebar: Logout and Export ---
with st.sidebar:
    st.markdown('## ⚙️ Options')
    if st.button('Logout', key='logout_btn', use_container_width=True):
        st.session_state['authenticated'] = False
        st.rerun()
    
    st.markdown('---')
    st.markdown('### 🔄 Usage Management')
    
    # Show current reset status
    next_reset = get_next_reset_time()
    time_until_reset = next_reset - datetime.now()
    hours_until_reset = time_until_reset.total_seconds() / 3600
    
    if hours_until_reset > 0:
        st.info(f"Next reset: {hours_until_reset:.1f}h")
    else:
        st.success("Reset available!")
    
    # Manual reset button (for testing)
    if st.button('🔄 Force Daily Reset', key='force_reset_btn', use_container_width=True):
        if check_and_reset_daily_usage():
            st.success('✅ Usage reset successfully!')
        else:
            st.info('ℹ️ Reset not needed yet')
        st.rerun()
    
    st.markdown('---')
    if st.session_state.get('chat_history'):
        export_format = st.selectbox('Export chat as:', ['Text (.txt)', 'JSON (.json)'], key='export_format')
        if st.button('Export Chat', key='export_btn', use_container_width=True):
            import io
            import json as pyjson
            if export_format == 'Text (.txt)':
                chat_lines = []
                for msg in st.session_state['chat_history']:
                    role = 'You' if msg['role'] == 'user' else 'AI'
                    chat_lines.append(f"{role}: {msg['content']}")
                chat_text = '\n'.join(chat_lines)
                st.download_button('Download Chat (.txt)', chat_text, file_name='chat_history.txt', mime='text/plain', use_container_width=True)
            else:
                chat_json = pyjson.dumps(st.session_state['chat_history'], indent=2, ensure_ascii=False)
                st.download_button('Download Chat (.json)', chat_json, file_name='chat_history.json', mime='application/json', use_container_width=True)