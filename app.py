import streamlit as st
import os
from dotenv import load_dotenv
import json
from datetime import datetime, date, timedelta

# Load environment variables
load_dotenv()

# Authentication
USERNAME = os.getenv('STREAMLIT_USERNAME')
PASSWORD = os.getenv('STREAMLIT_PASSWORD')

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

try:
    import openai
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if os.getenv('OPENAI_API_KEY'):
        OPENAI_AVAILABLE = True
except:
    pass

try:
    import google.generativeai as genai
    if os.getenv('GEMINI_API_KEY'):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        GEMINI_AVAILABLE = True
except:
    pass

try:
    import anthropic
    if os.getenv('ANTHROPIC_API_KEY'):
        anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        ANTHROPIC_AVAILABLE = True
except:
    pass

# Usage tracking
USAGE_FILE = 'usage_tracker.json'

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

def increment_usage(provider, model_name):
    """Increment usage for a specific model"""
    current_usage, usage_data = get_usage_for_model(provider, model_name)
    today = date.today().isoformat()
    usage_data[provider][model_name][today] = current_usage + 1
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
    """Get response from selected AI model"""
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
            return response.text
            
        elif provider == 'OpenAI' and OPENAI_AVAILABLE:
            # Build messages for OpenAI
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for msg in chat_history[-5:] if chat_history else []:
                messages.append({"role": msg['role'], "content": msg['content']})
            messages.append({"role": "user", "content": message})
            
            # Generate response
            response = openai.ChatCompletion.create(
                model=model_id,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
            
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
            return response.content[0].text
            
        else:
            return f"API not available for {provider}. Please check your API keys in .env file."
            
    except Exception as e:
        return f"Error: {str(e)}"

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
            st.error('❌ Gemini API not available - check GEMINI_API_KEY in .env')
    elif provider == 'OpenAI':
        if OPENAI_AVAILABLE:
            st.success('✅ OpenAI API available')
        else:
            st.error('❌ OpenAI API not available - check OPENAI_API_KEY in .env')
    elif provider == 'Anthropic':
        if ANTHROPIC_AVAILABLE:
            st.success('✅ Anthropic API available')
        else:
            st.error('❌ Anthropic API not available - check ANTHROPIC_API_KEY in .env')
else:
    st.error('No models configured. Please check config.json')
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# --- Usage Tracker Implementation ---
st.markdown('<div class="usage-card">', unsafe_allow_html=True)
st.subheader('📊 Usage Tracker')

current_usage, _ = get_usage_for_model(selected_model['provider'], selected_model['model_name'])
limit = selected_model['config']['limit']
usage_percentage = (current_usage / limit) * 100

# Progress bar with color coding
if usage_percentage < 70:
    color = "green"
elif usage_percentage < 90:
    color = "orange"
else:
    color = "red"

st.progress(usage_percentage / 100)
st.write(f"**{selected_model['display_name']}:** {current_usage}/{limit} requests used ({usage_percentage:.1f}%)")

if usage_percentage >= 90:
    st.warning(f"⚠️ You're close to your {selected_model['model_name']} limit!")
elif usage_percentage >= 70:
    st.info(f"ℹ️ You've used {usage_percentage:.1f}% of your {selected_model['model_name']} limit")

st.markdown('</div>', unsafe_allow_html=True)

# --- Enhanced Chat UI ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.subheader('💬 Chat')

# Display chat history as enhanced bubbles
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state['chat_history']):
        if msg['role'] == 'user':
            st.markdown(f"""
            <div style="text-align: right; margin-bottom: 12px;">
                <span style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 10px 16px; border-radius: 18px; max-width: 70%; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{msg["content"]}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: left; margin-bottom: 12px;">
                <span style="display: inline-block; background: white; color: #333; 
                padding: 10px 16px; border-radius: 18px; max-width: 70%; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e0e0e0;">{msg["content"]}</span>
            </div>
            """, unsafe_allow_html=True)

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
        # Increment usage
        increment_usage(selected_model['provider'], selected_model['model_name'])
        # Get real AI response
        with st.spinner('🤔 Thinking...'):
            ai_response = get_ai_response(selected_model, user_input, st.session_state['chat_history'])
        # Enhanced streaming effect
        displayed = ''
        response_placeholder = st.empty()
        for char in ai_response:
            displayed += char
            response_placeholder.markdown(f"""
            <div style=\"text-align: left; margin-bottom: 12px;\">
                <span style=\"display: inline-block; background: white; color: #333; \
                padding: 10px 16px; border-radius: 18px; max-width: 70%; \
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e0e0e0;\">{displayed}</span>
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