import streamlit as st
from datetime import datetime

from frontend.utils.api_client import api_client
from frontend.components.chat_view import render_chat_messages

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean, modern typography and layout */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 950px;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    .status-online {
        background-color: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-warning {
        background-color: rgba(234, 179, 8, 0.15);
        color: #ca8a04;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    .status-offline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Sidebar conversation item styling */
    .conv-item {
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(120, 120, 120, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "current_title" not in st.session_state:
    st.session_state.current_title = "New Conversation"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_instruction" not in st.session_state:
    st.session_state.system_instruction = (
        "You are a helpful, knowledgeable, and concise AI assistant."
    )

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def reset_to_new_chat():
    """Reset session state to start a fresh conversation."""
    st.session_state.current_session_id = None
    st.session_state.current_title = "New Conversation"
    st.session_state.messages = []


def load_selected_conversation(conv_id: str):
    """Load a specific conversation and its messages from the SQLite database."""
    conv_data = api_client.get_conversation(conv_id)
    if conv_data:
        st.session_state.current_session_id = conv_data["id"]
        st.session_state.current_title = conv_data.get("title", "Saved Conversation")
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]}
            for m in conv_data.get("messages", [])
        ]
    else:
        st.error(f"Could not load conversation {conv_id}.")


# -----------------------------------------------------------------------------
# Sidebar: Status, Controls, and Conversation History
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Multimodal AI")

    # 1. Backend Connectivity & Health Check
    health = api_client.check_health()
    if health["connected"]:
        data = health["data"] or {}
        gemini_ok = data.get("gemini_configured", False)
        if gemini_ok:
            st.markdown(
                '<div class="status-badge status-online">🟢 Backend Connected · Gemini Ready</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge status-warning">⚠️ Backend Connected · Missing API Key</div>',
                unsafe_allow_html=True,
            )
            st.caption("Add `GEMINI_API_KEY` to your `.env` file to enable AI answers.")
    else:
        st.markdown(
            '<div class="status-badge status-offline">🔴 Backend Offline</div>',
            unsafe_allow_html=True,
        )
        st.caption("FastAPI is not reachable on port 8000.")

    # 2. New Chat Action
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        reset_to_new_chat()
        st.rerun()

    st.divider()

    # 3. Model Configuration Expander
    with st.expander("⚙️ Chat Settings", expanded=False):
        st.session_state.temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.5,
            value=st.session_state.temperature,
            step=0.1,
            help="Lower values make answers more focused and deterministic; higher values make them more creative.",
        )
        st.session_state.system_instruction = st.text_area(
            "System Persona",
            value=st.session_state.system_instruction,
            help="Custom guidelines for the AI assistant's persona or tone.",
            height=90,
        )

    st.markdown("#### 💬 Saved Conversations")

    # 4. Conversation History from Database
    conversations = api_client.list_conversations()
    if not conversations:
        st.caption("No previous conversations saved yet.")
    else:
        for conv in conversations:
            conv_id = conv["id"]
            title = conv.get("title", "Untitled Chat")
            is_active = conv_id == st.session_state.current_session_id

            col_btn, col_del = st.columns([0.82, 0.18])

            with col_btn:
                button_label = f"{'👉 ' if is_active else ''}{title}"
                if st.button(
                    button_label,
                    key=f"load_{conv_id}",
                    use_container_width=True,
                    help=f"ID: {conv_id[:8]}... · Click to resume",
                ):
                    load_selected_conversation(conv_id)
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"del_{conv_id}", help="Delete conversation"):
                    if api_client.delete_conversation(conv_id):
                        if st.session_state.current_session_id == conv_id:
                            reset_to_new_chat()
                        st.rerun()
                    else:
                        st.error("Failed to delete.")


# -----------------------------------------------------------------------------
# Main Chat Area
# -----------------------------------------------------------------------------

# Header
col_header, col_badge = st.columns([0.75, 0.25])
with col_header:
    st.subheader(f"💬 {st.session_state.current_title}")
with col_badge:
    model_name = "gemini-1.5-flash"
    if health["connected"] and health["data"]:
        model_name = health["data"].get("model", model_name)
    st.markdown(
        f"<div style='text-align: right; padding-top: 0.5rem;'><span style='background: rgba(120,120,120,0.1); padding: 0.3rem 0.7rem; border-radius: 8px; font-size: 0.85rem;'>⚡ {model_name}</span></div>",
        unsafe_allow_html=True,
    )

# Render Chat History
render_chat_messages(st.session_state.messages)

# Chat Input & Interaction
user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    if not user_prompt.strip():
        st.warning("Please enter a valid message.")
    elif not health["connected"]:
        st.error(
            "Cannot send message: FastAPI backend is not running. "
            "Please start the backend server (`uvicorn backend.app.main:app --port 8000`)."
        )
    else:
        # 1. Immediately append and display the user's message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_prompt)

        # 2. Display assistant container with thinking spinner
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Gemini is thinking..."):
                response = api_client.send_chat_message(
                    message=user_prompt,
                    session_id=st.session_state.current_session_id,
                    system_instruction=st.session_state.system_instruction,
                    temperature=st.session_state.temperature,
                )

        # 3. Handle response
        if response["success"]:
            resp_data = response["data"]
            reply_text = resp_data["reply"]
            new_session_id = resp_data.get("session_id")

            # Update session state with assigned session ID
            if new_session_id and not st.session_state.current_session_id:
                st.session_state.current_session_id = new_session_id
                # Derive title from user prompt
                title_snip = user_prompt.strip().split("\n")[0][:35]
                st.session_state.current_title = (
                    title_snip + "..." if len(user_prompt.strip().split("\n")[0]) > 35 else title_snip
                )

            # Append assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": reply_text})
            st.rerun()

        else:
            error_msg = response["error"]
            st.error(f"❌ {error_msg}")
