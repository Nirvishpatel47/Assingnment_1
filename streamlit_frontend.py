import streamlit as st
import asyncio
from main_backend import ChatFlowController
from RAG import RAGBot

st.set_page_config(page_title="AutoStream AI", page_icon="🤖", layout="centered")

# Minimal CSS - Clean & Simple
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp {
        background-color: #ffffff;
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        max-width: 48rem;
        padding-top: 3rem;
    }
    
    h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #202124;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* User message - right aligned */
    .user-msg {
        background-color: #f0f0f0;
        padding: 0.75rem 1rem;
        border-radius: 1.125rem;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 80%;
        color: #202124;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* AI message - left aligned */
    .ai-msg {
        background-color: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 1.125rem;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 80%;
        color: #202124;
        font-size: 0.95rem;
        line-height: 1.5;
        border: 1px solid #e5e5e5;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("AutoStream AI")

# Initialize
if 'controller' not in st.session_state:
    rag_bot = RAGBot(client_id="demo_user")
    st.session_state.controller = ChatFlowController(rag_bot)
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# Input
user_input = st.chat_input("Message AutoStream AI...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(st.session_state.controller.handle(user_input))
    loop.close()
    
    st.session_state.messages.append({"role": "agent", "content": response})
    st.rerun()
