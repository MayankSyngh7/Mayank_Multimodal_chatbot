import streamlit as st
from typing import List, Dict, Any


def render_empty_state():
    """Renders a friendly welcome screen when there are no messages in the current session."""
    st.markdown(
        """
        <div style="text-align: center; padding: 2.5rem 1rem; border-radius: 12px; background: rgba(120, 120, 120, 0.05); margin-bottom: 2rem;">
            <h2 style="margin-bottom: 0.5rem;">👋 Welcome to your Multimodal AI Assistant</h2>
            <p style="color: #666; font-size: 1.05rem; margin-bottom: 1.5rem;">
                Powered by <strong>Google Gemini</strong>, <strong>FastAPI</strong>, and <strong>SQLite</strong>.
            </p>
            <div style="display: flex; gap: 0.8rem; justify-content: center; flex-wrap: wrap;">
                <span style="background: rgba(0, 120, 255, 0.1); color: #0070f3; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.9rem;">
                    💡 Ask technical questions
                </span>
                <span style="background: rgba(40, 167, 69, 0.1); color: #28a745; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.9rem;">
                    📝 Draft summaries & notes
                </span>
                <span style="background: rgba(255, 140, 0, 0.1); color: #ff8c00; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.9rem;">
                    🐍 Debug Python & code
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_messages(messages: List[Dict[str, Any]]):
    """
    Renders all chat messages from session state in chronological order.
    Handles 'user' and 'assistant' / 'model' roles.
    """
    if not messages:
        render_empty_state()
        return

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map assistant role
        ui_role = "user" if role == "user" else "assistant"
        avatar = "👤" if ui_role == "user" else "🤖"

        with st.chat_message(ui_role, avatar=avatar):
            st.markdown(content)
