import streamlit as st
from typing import List, Dict, Any, Callable, Optional


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
                    💬 Text Chat
                </span>
                <span style="background: rgba(147, 51, 234, 0.1); color: #9333ea; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.9rem;">
                    🖼️ Image Understanding
                </span>
                <span style="background: rgba(234, 88, 12, 0.1); color: #ea580c; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.9rem;">
                    🎙️ Voice Input
                </span>
                <span style="background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.9rem;">
                    🔊 Text-to-Speech
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_messages(
    messages: List[Dict[str, Any]],
    on_play_tts: Optional[Callable[[str, int], None]] = None,
):
    """
    Renders all chat messages from session state in chronological order.
    Supports user messages, AI replies, image previews, and TTS audio playback.
    """
    if not messages:
        render_empty_state()
        return

    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        ui_role = "user" if role == "user" else "assistant"
        avatar = "👤" if ui_role == "user" else "🤖"

        with st.chat_message(ui_role, avatar=avatar):
            # Check if this message has an attached image preview in session
            if "image_data" in msg and msg["image_data"]:
                st.image(msg["image_data"], caption="Uploaded Image", use_container_width=False, width=320)

            st.markdown(content)

            # For assistant responses, provide TTS audio playback
            if ui_role == "assistant" and content.strip():
                # Check if audio was already generated for this message
                audio_key = f"audio_{idx}"
                if audio_key in st.session_state and st.session_state[audio_key]:
                    st.audio(st.session_state[audio_key], format="audio/mp3")
                else:
                    col_listen, _ = st.columns([0.22, 0.78])
                    with col_listen:
                        if st.button("🔊 Listen", key=f"tts_btn_{idx}", help="Convert AI response to voice audio"):
                            if on_play_tts:
                                on_play_tts(content, idx)
