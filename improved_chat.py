"""
Improved Chat Interface Component

This demonstrates a better chat interface with proper scrolling behavior.
"""

import streamlit as st
from datetime import datetime

def render_improved_chat_interface(chroma_client, selected_collection):
    """Render an improved chat interface with better UX."""
    
    # Chat History initialisieren
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # CSS für verbessertes Chat-Layout
    st.markdown("""
    <style>
    /* Chat Container Styling */
    .chat-container {
        height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background: linear-gradient(to bottom, #fafafa, #ffffff);
        margin-bottom: 1rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Streamlit Chat Message Styling */
    .stChatMessage {
        margin-bottom: 1rem !important;
    }
    
    /* Chat Input fixiert am unteren Rand */
    .stChatInput {
        position: sticky !important;
        bottom: 0 !important;
        background-color: white !important;
        padding: 1rem 0 !important;
        border-top: 2px solid #e0e0e0 !important;
        z-index: 999 !important;
    }
    
    /* Chat Input Field Styling */
    .stChatInput > div > div > div > div {
        border-radius: 25px !important;
        border: 2px solid #667eea !important;
    }
    
    /* Auto-scroll Animation */
    .chat-scroll {
        scroll-behavior: smooth;
    }
    
    /* Message Timestamps */
    .message-timestamp {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        color: #666;
    }
    
    .typing-dots {
        display: inline-flex;
        align-items: center;
    }
    
    .typing-dots span {
        height: 8px;
        width: 8px;
        background-color: #667eea;
        border-radius: 50%;
        display: inline-block;
        margin: 0 2px;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat Header mit Statistiken
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 💬 Chat mit deiner Wissensdatenbank")
    
    with col2:
        if st.session_state.chat_history:
            msg_count = len(st.session_state.chat_history)
            user_msgs = len([m for m in st.session_state.chat_history if m["role"] == "user"])
            st.metric("Nachrichten", msg_count, delta=f"{user_msgs} Fragen")
    
    with col3:
        if st.button("🗑️ Chat löschen", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Chat Messages Container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            # Zeige alle Chat-Nachrichten
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Timestamp anzeigen
                    if "timestamp" in message:
                        timestamp = datetime.fromisoformat(message["timestamp"])
                        st.markdown(f'<div class="message-timestamp">{timestamp.strftime("%H:%M")}</div>', 
                                  unsafe_allow_html=True)
        else:
            # Welcome Message
            with st.chat_message("assistant"):
                st.markdown("""
                👋 **Willkommen!** 
                
                Ich bin dein KI-Assistent für die Wissensdatenbank **{}**. 
                
                Du kannst mir Fragen stellen wie:
                - "Was ist das Hauptthema dieser Dokumentation?"
                - "Erkläre mir [spezifisches Thema]"
                - "Gib mir eine Zusammenfassung von [Bereich]"
                
                Stelle einfach deine erste Frage! 🚀
                """.format(selected_collection))
    
    # Chat Input (bleibt am unteren Rand fixiert)
    prompt = st.chat_input("Stelle eine Frage zu deiner Wissensdatenbank...")
    
    # Chat Input Verarbeitung
    if prompt:
        # Timestamp für neue Nachricht
        current_time = datetime.now()
        
        # User Message zur History hinzufügen
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt,
            "timestamp": current_time.isoformat()
        })
        
        # User Message anzeigen
        with st.chat_message("user"):
            st.markdown(prompt)
            st.markdown(f'<div class="message-timestamp">{current_time.strftime("%H:%M")}</div>', 
                       unsafe_allow_html=True)
        
        # Typing Indicator
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("""
            <div class="typing-indicator">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span style="margin-left: 10px;">Durchsuche Wissensdatenbank...</span>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # RAG Response generieren
                response = generate_rag_response(prompt, selected_collection, chroma_client)
                
                # Typing Indicator entfernen und Antwort anzeigen
                typing_placeholder.empty()
                st.markdown(response)
                
                # Response zur History hinzufügen
                response_time = datetime.now()
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": response_time.isoformat()
                })
                
                # Timestamp anzeigen
                st.markdown(f'<div class="message-timestamp">{response_time.strftime("%H:%M")}</div>', 
                           unsafe_allow_html=True)
                
            except Exception as e:
                typing_placeholder.empty()
                error_msg = f"❌ Entschuldigung, es gab einen Fehler: {str(e)}"
                st.error(error_msg)
                
                # Error zur History hinzufügen
                error_time = datetime.now()
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "timestamp": error_time.isoformat()
                })
        
        # Auto-scroll zum Ende mit JavaScript
        st.markdown("""
        <script>
        setTimeout(function() {
            // Scroll to bottom of the page
            window.scrollTo(0, document.body.scrollHeight);
            
            // Also try to scroll chat container if it exists
            var chatElements = document.querySelectorAll('[data-testid="stChatMessageContainer"]');
            chatElements.forEach(function(element) {
                element.scrollTop = element.scrollHeight;
            });
        }, 100);
        </script>
        """, unsafe_allow_html=True)
        
        # Rerun für bessere UX
        st.rerun()
    
    # Chat Export Funktionalität
    if st.session_state.chat_history:
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📥 Chat exportieren"):
                # Chat als Markdown exportieren
                chat_export = f"# Chat Export - {selected_collection}\n\n"
                chat_export += f"Exportiert am: {datetime.now().strftime('%d.%m.%Y um %H:%M')}\n\n"
                
                for msg in st.session_state.chat_history:
                    role = "**Du**" if msg["role"] == "user" else "**Assistant**"
                    timestamp = ""
                    if "timestamp" in msg:
                        ts = datetime.fromisoformat(msg["timestamp"])
                        timestamp = f" _{ts.strftime('%H:%M')}_"
                    
                    chat_export += f"{role}{timestamp}:\n{msg['content']}\n\n---\n\n"
                
                st.download_button(
                    label="💾 Download Chat (.md)",
                    data=chat_export,
                    file_name=f"chat_{selected_collection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col2:
            # Chat Statistiken
            total_chars = sum(len(msg["content"]) for msg in st.session_state.chat_history)
            st.metric("Zeichen gesamt", f"{total_chars:,}")


def generate_rag_response(question: str, collection_name: str, chroma_client) -> str:
    """Generiere RAG-Antwort (Placeholder für Demo)."""
    import time
    import random
    
    # Simuliere Verarbeitungszeit
    time.sleep(random.uniform(1, 3))
    
    # Demo Response
    responses = [
        f"Basierend auf der Wissensdatenbank '{collection_name}' kann ich dir folgendes sagen: {question}",
        f"Interessante Frage zu '{collection_name}'! Hier ist was ich gefunden habe...",
        f"Nach der Durchsuchung von '{collection_name}' habe ich relevante Informationen zu deiner Frage gefunden.",
    ]
    
    return random.choice(responses) + f"\n\n*Diese Antwort basiert auf {random.randint(3, 15)} relevanten Dokumenten aus der Wissensdatenbank.*"


# Demo Usage
if __name__ == "__main__":
    st.set_page_config(
        page_title="Improved Chat Demo",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 Improved Chat Interface Demo")
    
    # Mock chroma client and collection
    class MockChromaClient:
        pass
    
    render_improved_chat_interface(MockChromaClient(), "Demo Collection")