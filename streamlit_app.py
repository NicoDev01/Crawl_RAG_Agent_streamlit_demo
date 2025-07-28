# HACK: SQLite3 Kompatibilität für Streamlit Community Cloud
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import asyncio
import base64
import tempfile
import time

from datetime import datetime

# ChromaDB und andere Imports
import chromadb
from crawler_client import CrawlerClient

# UX Components Import
from ux_components import URLValidator
from url_detection import detect_url_type

# Windows Event Loop Policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Streamlit Konfiguration
st.set_page_config(
    page_title="🤖 CraCha - Crawl Chat Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS für minimalistisches Design
st.markdown("""
<style>
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Standard Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    /* Hervorgehobener Submit Button */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6) !important;
    }
    
    /* Verhindere Duplikation während Processing */
    .stSpinner {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 9999 !important;
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 2rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Verhindere Overlay-Duplikation */
    .stForm {
        position: relative !important;
    }
    
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_google_cloud_credentials():
    """Setup Google Cloud Credentials aus Streamlit Secrets."""
    try:
        creds_json_b64 = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if creds_json_b64:
            creds_json = base64.b64decode(creds_json_b64).decode('utf-8')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as temp:
                temp.write(creds_json)
                temp_filename = temp.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_filename
            return True
        return False
    except Exception as e:
        st.error(f"Ein Fehler ist beim Setup von Google Cloud Credentials aufgetreten: {e}")
        return False

@st.cache_resource
def get_chroma_client():
    """Erstelle einen In-Memory ChromaDB Client mit vorgeladenem Model."""
    try:
        client = chromadb.Client()
        
        # Model durch Dummy-Operation vorladen um 79MB Download zu vermeiden
        print("🔄 Lade ChromaDB Embedding-Model vor...")
        temp_collection_name = f"model_preload_{int(time.time())}"
        temp_collection = client.create_collection(temp_collection_name)
        temp_collection.add(documents=["dummy text for model loading"], ids=["preload_1"])
        client.delete_collection(temp_collection_name)
        print("✅ Embedding-Model vorgeladen.")
        
        return client
    except Exception as e:
        st.error(f"Ein Fehler ist beim Initialisieren von ChromaDB aufgetreten: {e}")
        try:
            print("⚠️ WARNUNG: Fallback zu ChromaDB Client ohne Pre-Loading.")
            return chromadb.Client()
        except Exception as fallback_e:
            st.error(f"Ein unerwarteter Fehler ist aufgetreten: {fallback_e}")
            return None

@st.cache_resource
def get_crawler_client():
    """Erstelle den Modal.com Crawler Client."""
    try:
        base_url = st.secrets.get("MODAL_API_URL", "https://nico-gt91--crawl4ai-service")
        api_key = st.secrets.get("MODAL_API_KEY")
        
        if not api_key:
            return None
            
        return CrawlerClient(base_url=base_url, api_key=api_key)
    except Exception as e:
        st.error(f"Ein Fehler ist beim Initialisieren des Crawler Clients aufgetreten: {e}")
        return None

def main():
    # Dezenter Header mit Blog-Link
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 1.5rem;">
        <h2 style="color: #667eea; margin-bottom: 0.5rem;">🤖 CraCha - Crawl Chat Agent</h2>
        <p style="color: #666; font-size: 0.9rem; margin: 0;">Intelligente Wissensdatenbanken aus Webseiten erstellen</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Blog-Artikel Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <a href="https://seed-vanilla-a45.notion.site/CraCha-Crawl-Chat-RAG-Agent-23a3f68d746880df9b28f0db73f6f6f0" target="_blank" style="text-decoration: none;">
                <button style="
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 0.5rem 1.5rem;
                    font-weight: bold;
                    font-size: 0.9rem;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
                " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(102, 126, 234, 0.4)'" 
                   onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 2px 8px rgba(102, 126, 234, 0.3)'">
                    📖 Zum Blog-Artikel über CraCha
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialisiere Services (im Hintergrund)
    setup_google_cloud_credentials()
    chroma_client = get_chroma_client()
    crawler_client = get_crawler_client()
    
    # Prüfe kritische Services
    if not chroma_client or not crawler_client:
        st.error("🚨 Service nicht verfügbar. Bitte versuche es später erneut.")
        st.stop()
    
    # Hauptnavigation
    tab1, tab2 = st.tabs(["📚 Wissensdatenbank erstellen", "💬 Chat mit deinen Daten"])
    
    with tab1:
        create_knowledge_base(crawler_client, chroma_client)
    
    with tab2:
        chat_interface(chroma_client)

def create_knowledge_base(crawler_client, chroma_client):
    """Minimalistisches Interface für Wissensdatenbank-Erstellung."""
    
    # Wenn Processing läuft, zeige nur den Progress
    if 'processing' in st.session_state and st.session_state.processing:
        st.info("🔄 Verarbeitung läuft... Bitte warten.")
        return
    
    # Hilfe-Sektion
    with st.expander("💡 Hilfe: Unterstützte Formate und Einstellungen"):
        st.markdown("""
        **1. 📋 Unterstützte Formate:**
        
        • **🌐 Website-URLs** → Automatisches Crawling aller Unterseiten  
          `https://docs.example.com`, `https://example.com/help`
          
        • **🗺️ Sitemap-URLs** → Direktes Parsing der XML-Sitemap  
          `https://example.com/sitemap.xml`, `https://site.com/sitemap_index.xml`
          
        • **📄 Einzelseiten** → Extraktion einer spezifischen Seite  
          `https://example.com/page.html`, `https://blog.example.com/artikel`
          
        • **📚 Dokumentations-Sites** → Speziell für Docs optimiert  
          `https://docs.example.com`, `https://help.example.com`
        
        **2. ⚙️ Einstellungsmöglichkeiten:**
        
        - **Crawling-Tiefe**: Bestimmt, wie viele Link-Ebenen verfolgt werden (1-4)
        - **Seitenzahl**: Maximale Anzahl der zu verarbeitenden Seiten (1-100)
        - **Automatische Optimierung**: Passt Einstellungen für beste Performance an
        - **Demo-Limits**: Bis 20 Seiten ~5min, darüber deutlich länger
        
        **💡 Tipp: Starte mit niedrigen Werten und erhöhe sie bei Bedarf!**
        """)
    
    with st.form("knowledge_creation"):
        # Basis-Konfiguration
        st.subheader("🌐 Website eingeben")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Initialize URL validator
            if 'url_validator' not in st.session_state:
                st.session_state.url_validator = URLValidator(timeout=10, debounce_delay=0.5)
            
            url_validator = st.session_state.url_validator
            
            # URL input with real-time validation (Enter soll NICHT submitten)
            url = st.text_input(
                "Website URL: *",
                placeholder="https://docs.example.com oder https://example.com/sitemap.xml",
                help="Vollständige URL der Website oder Sitemap (Pflichtfeld)",
                key="main_url_input",
                on_change=None  # Verhindert Auto-Submit bei Enter
            )
            
            # Real-time URL validation feedback
            if url and url.strip():
                validation_result = url_validator.render_validation_feedback(
                    url, 
                    show_reachability=True, 
                    debounced=True
                )
                
                # Store validation result in session state for form submission check
                st.session_state.url_validation_result = validation_result
            else:
                st.session_state.url_validation_result = None
            
            name = st.text_input(
                "Name der Wissensdatenbank: *",
                placeholder="z.B. 'Produktdokumentation' oder 'Firmen-Wiki'",
                help="Eindeutiger Name zur Identifikation deiner Wissensdatenbank (Pflichtfeld)"
            )
        
        with col2:
            # Intelligente URL-Typ-Erkennung
            if url and url.strip():
                # URL-Validierung
                url_validation = getattr(st.session_state, 'url_validation_result', None)
                if url_validation:
                    if url_validation.is_valid:
                        # Intelligente Typ-Erkennung (nur für interne Logik)
                        detected_method = detect_url_type(url)
                        st.session_state.detected_crawling_method = detected_method
            else:
                st.session_state.detected_crawling_method = None

        
        # Crawling-Einstellungen - immer angezeigt
        st.subheader("⚙️ Crawling-Einstellungen")
        
        col3, col4 = st.columns(2)
        
        with col3:
            max_depth = st.slider(
                "Crawling-Tiefe:",
                min_value=1, max_value=4, value=1,
                help="Wie tief sollen Links verfolgt werden?"
            )
            st.caption("🔍 Bestimmt, wie viele Link-Ebenen von der Startseite aus verfolgt werden")
        
        with col4:
            max_pages = st.number_input(
                "Maximale Seitenzahl:",
                min_value=1, max_value=100, 
                value=1,
                help="Maximale Anzahl zu crawlender Seiten"
            )
            st.caption("🔢 Bestimmt die maximale Anzahl der Seiten, die gecrawlt und verarbeitet werden")
        
        # Erweiterte Einstellungen - für Demo ausgeblendet, feste Werte
        chunk_size = 1200  # Fester Wert
        auto_reduce = True  # Fester Wert
        max_concurrent = 2  # Fester Wert
        
        # Demo-Zeitschätzung mit allgemeinen Hinweisen
        st.info("⏱️ **Demo-Hinweise:** Bis zu 20 Seiten dauern ca. 5 Minuten. Ab 20 Seiten erhöht sich die Dauer deutlich!!!")
        
        # Form submission with enhanced validation
        submitted = st.form_submit_button("🚀 Wissensdatenbank erstellen", use_container_width=True)
        
        if submitted and 'processing' not in st.session_state:
            # Enhanced validation before processing
            validation_errors = []
            
            # Check required fields
            if not url or not url.strip():
                validation_errors.append("Website URL ist erforderlich")
            
            if not name or not name.strip():
                validation_errors.append("Name der Wissensdatenbank ist erforderlich")
            
            # Check URL validation result
            url_validation = getattr(st.session_state, 'url_validation_result', None)
            if url and url_validation and not url_validation.is_valid:
                validation_errors.append(f"URL ist ungültig: {url_validation.error_message}")
            
            # Show validation errors
            if validation_errors:
                st.error("❌ **Bitte korrigiere folgende Fehler:**")
                for error in validation_errors:
                    st.error(f"• {error}")
            else:
                # Show URL validation status before processing
                if url_validation and url_validation.warning_message:
                    st.warning(f"⚠️ **Warnung:** {url_validation.warning_message}")
                    st.info("Das Crawling wird trotzdem fortgesetzt...")
                
                # Set processing state to prevent duplication
                st.session_state.processing = True
                
                # Proceed with knowledge base creation
                create_knowledge_base_process(url, name, None, max_pages, chunk_size, auto_reduce, crawler_client, chroma_client, max_depth, max_concurrent)

def create_knowledge_base_process(url, name, detected_method, max_pages, chunk_size, auto_reduce, crawler_client, chroma_client, max_depth=2, max_concurrent=5):
    """Prozess der Wissensdatenbank-Erstellung."""
    
    # Leere die Seite und zeige nur Progress
    st.empty()
    
    # Progress Container - isoliert von der Form
    with st.container():
        st.markdown("---")
        
        st.markdown("---")
        
        try:
            # Import der Ingestion-Pipeline
            from insert_docs_streamlit import run_ingestion_sync, IngestionProgress
            
            # Echter Spinner mit dynamischen Updates
            spinner_container = st.empty()
            
            class SpinnerProgress(IngestionProgress):
                def __init__(self, container):
                    self.container = container
                    self.current_spinner = None
                    self.current_status = ""
                
                def update(self, step: int, message: str):
                    """Startet neuen Spinner mit aktueller Nachricht"""
                    self.current_status = message
                    
                    # Beende vorherigen Spinner
                    if self.current_spinner:
                        self.current_spinner.__exit__(None, None, None)
                    
                    # Starte neuen Spinner
                    with self.container:
                        self.current_spinner = st.spinner(message, show_time=True)
                        self.current_spinner.__enter__()
                
                def show_sub_process(self, sub_message: str):
                    """Aktualisiert Spinner mit Sub-Prozess"""
                    self.current_status = sub_message
                    
                    # Beende vorherigen Spinner
                    if self.current_spinner:
                        self.current_spinner.__exit__(None, None, None)
                    
                    # Starte neuen Spinner mit Sub-Prozess
                    with self.container:
                        self.current_spinner = st.spinner(sub_message, show_time=True)
                        self.current_spinner.__enter__()
                
                def complete(self, message: str):
                    """Beendet Spinner und zeigt Erfolg"""
                    if self.current_spinner:
                        self.current_spinner.__exit__(None, None, None)
                        self.current_spinner = None
                    
                    with self.container:
                        st.success(f"✅ {message}")
                        import time
                        time.sleep(0.5)  # Kurz anzeigen
                
                def finish(self):
                    """Beendet alle Spinner"""
                    if self.current_spinner:
                        self.current_spinner.__exit__(None, None, None)
                    self.container.empty()
            
            progress = SpinnerProgress(spinner_container)
            
            # Konfiguration basierend auf intelligenter Erkennung
            if detected_method and detected_method.method == "sitemap":
                depth = 1
                limit = None
            elif detected_method and detected_method.method == "single":
                depth = 1
                limit = 1
            else:  # website, documentation
                depth = max_depth
                limit = max_pages
            
            # Ingestion ausführen mit dynamischer Anzeige
            result = run_ingestion_sync(
                url=url,
                collection_name=name,
                crawler_client=crawler_client,
                chroma_client=chroma_client,
                chunk_size=chunk_size,
                chunk_overlap=150,
                max_depth=depth,
                max_concurrent=max_concurrent,
                limit=limit,
                progress=progress,
                auto_reduce=auto_reduce,
                max_chunks=None
            )
            
            st.success("🎉 Wissensdatenbank erfolgreich erstellt!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Dokumente", result.get('documents_crawled', 0))
            with col2:
                st.metric("📝 Text-Chunks", result.get('chunks_created', 0))
            with col3:
                st.metric("🧠 Embeddings", result.get('embeddings_generated', 0))
            
            st.info("💡 Wechsle zum 'Chat' Tab um mit deiner Wissensdatenbank zu interagieren!")
            
            # Reset processing state
            if 'processing' in st.session_state:
                del st.session_state.processing
            
        except Exception as e:
            st.error(f"❌ Fehler beim Erstellen der Wissensdatenbank: {str(e)}")
            
            if "Memory limit exceeded" in str(e):
                st.info("💡 Tipp: Versuche eine kleinere Chunk-Größe oder weniger Seiten.")
            
            # Reset processing state on error
            if 'processing' in st.session_state:
                del st.session_state.processing

def chat_interface(chroma_client):
    """Benutzerfreundliche Chat-Oberfläche."""
    
    # Verfügbare Collections laden
    try:
        collections = [c.name for c in chroma_client.list_collections()]
    except ValueError as e:
        st.error(f"Fehler beim Laden der Collections: {e}")
        collections = []
    
    if not collections:
        st.markdown("""
        <div class="feature-card">
            <h3>🤔 Keine Wissensdatenbanken gefunden</h3>
            <p>Erstelle zuerst eine Wissensdatenbank im Tab "Wissensdatenbank erstellen"</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Collection Auswahl
    st.markdown("### 🗂️ Wähle deine Wissensdatenbank")
    selected_collection = st.selectbox(
        "Wissensdatenbank:",
        collections,
        label_visibility="collapsed"
    )
    
    if selected_collection:
        # Collection Info
        collection = chroma_client.get_collection(selected_collection)
        chunk_count = collection.count()
        
        # Berechne Dokumente (FIXED: Alle Chunks berücksichtigen)
        try:
            # Für große Collections: Alle Metadaten abrufen (nur URLs)
            all_metadata = collection.get(limit=chunk_count, include=["metadatas"])
            unique_urls = set()
            for metadata in all_metadata["metadatas"]:
                if metadata and "url" in metadata:
                    unique_urls.add(metadata["url"])
            doc_count = len(unique_urls)
            print(f"📊 Collection '{selected_collection}': {chunk_count} chunks from {doc_count} unique documents")
        except Exception as e:
            print(f"⚠️ Error calculating document count: {e}")
            doc_count = "Unbekannt"
        
        # Info-Karten
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📄 **{doc_count}** Dokumente")
        with col2:
            st.info(f"📝 **{chunk_count}** Text-Chunks")
        
        st.markdown("---")
        
        # Verbessertes Chat Interface
        render_improved_chat_interface(chroma_client, selected_collection)


def render_improved_chat_interface(chroma_client, selected_collection):
    """Render an improved chat interface with better UX and auto-scrolling."""
    
    # Chat History initialisieren
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # CSS für verbessertes Chat-Layout
    st.markdown("""
    <style>
    /* Main Container */
    .main .block-container {
        padding-bottom: 120px !important;
        max-width: 100% !important;
    }
    
    /* Standard Chat Input - minimal styling */
    .stChatInput {
        /* Use default Streamlit styling */
    }
    
    /* Chat Messages Container */
    .stChatMessage {
        margin-bottom: 1rem !important;
        animation: fadeIn 0.3s ease-in !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Message Timestamps */
    .message-timestamp {
        font-size: 0.75rem;
        color: #666;
        margin-top: 0.5rem;
        text-align: right;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        color: #667eea;
        font-style: italic;
    }
    
    .typing-dots {
        display: inline-flex;
        align-items: center;
        margin-right: 10px;
    }
    
    .typing-dots span {
        height: 6px;
        width: 6px;
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
    
    /* Use default Streamlit status styling */
    
    /* Welcome Message Styling */
    .welcome-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
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
            st.metric("💬 Nachrichten", msg_count, delta=f"{user_msgs} Fragen")
    
    with col3:
        if st.button("🗑️ Chat löschen", key="clear_chat", help="Lösche den gesamten Chat-Verlauf"):
            st.session_state.chat_history = []
            st.success("✅ Chat wurde gelöscht!")
            st.rerun()
    
    # Chat Messages Container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            # Zeige alle Chat-Nachrichten
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Timestamp anzeigen wenn vorhanden
                    if "timestamp" in message:
                        try:
                            timestamp = datetime.fromisoformat(message["timestamp"])
                            st.markdown(f'<div class="message-timestamp">{timestamp.strftime("%H:%M")}</div>', 
                                      unsafe_allow_html=True)
                        except (ValueError, TypeError):
                            # Ignore errors if timestamp is malformed
                            pass
        else:
            # Welcome Message
            st.markdown(f"""
            <div class="welcome-message">
                <h3>👋 Willkommen!</h3>
                <p>Ich bin dein KI-Assistent für die Wissensdatenbank <strong>{selected_collection}</strong>.</p>
                <p>Du kannst mir Fragen stellen wie:</p>
                <ul>
                    <li>"Was ist das Hauptthema dieser Dokumentation?"</li>
                    <li>"Erkläre mir [spezifisches Thema]"</li>
                    <li>"Gib mir eine Zusammenfassung von [Bereich]"</li>
                </ul>
                <p><strong>Stelle einfach deine erste Frage! 🚀</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
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
        
        # Kein Progress Container mehr benötigt
        
        # Assistant Response
        with st.chat_message("assistant"):
            try:
                # Status Container für Progress
                with st.status("🤖 Verarbeite deine Anfrage...", expanded=True) as status:
                    st.write("🔄 Generiere Frage-Varianten...")
                    
                    # RAG Response generieren
                    from rag_agent import run_rag_agent_entrypoint, RAGDeps
                    
                    # Setup RAG Dependencies (ohne Vertex AI Reranker)
                    deps = RAGDeps(
                        chroma_client=chroma_client,
                        collection_name=selected_collection,
                        embedding_model_name="text-embedding-004",
                        embedding_provider="default",  # ChromaDB default
                        vertex_project_id=None,
                        vertex_location="us-central1",
                        use_vertex_reranker=False,  # Deaktiviert
                        vertex_reranker_model=None
                    )
                    
                    st.write("🧠 Erstelle hypothetische Antworten...")
                    st.write("🔍 Durchsuche Wissensdatenbank...")
                    
                    response = asyncio.run(run_rag_agent_entrypoint(
                        prompt, deps, "gemini-2.5-flash"
                    ))
                    
                    st.write("✅ Antwort generiert!")
                    status.update(label="✅ Fertig!", state="complete", expanded=False)
                
                # Antwort direkt anzeigen
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
                error_msg = f"❌ Entschuldigung, es gab einen Fehler: {str(e)}"
                st.error(error_msg)
                
                # Error zur History hinzufügen
                error_time = datetime.now()
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "timestamp": error_time.isoformat()
                })
        
        # Auto-scroll zum Ende
        st.markdown("""
        <script>
        setTimeout(function() {
            // Scroll to bottom of the page
            window.scrollTo({
                top: document.body.scrollHeight,
                behavior: 'smooth'
            });
        }, 200);
        </script>
        """, unsafe_allow_html=True)
        
        # Rerun für bessere UX
        st.rerun()
    
    # Chat Export Funktionalität (nur wenn Chat vorhanden)
    if st.session_state.chat_history:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Chat Export
            chat_export = f"# Chat Export - {selected_collection}\n\n"
            chat_export += f"Exportiert am: {datetime.now().strftime('%d.%m.%Y um %H:%M')}\n\n"
            
            for msg in st.session_state.chat_history:
                role = "**Du**" if msg["role"] == "user" else "**Assistant**"
                timestamp = ""
                if "timestamp" in msg:
                    try:
                        ts = datetime.fromisoformat(msg["timestamp"])
                        timestamp = f" _{ts.strftime('%H:%M')}_"
                    except:
                        pass
                
                chat_export += f"{role}{timestamp}:\n{msg['content']}\n\n---\n\n"
            
            st.download_button(
                label="📥 Chat exportieren",
                data=chat_export,
                file_name=f"chat_{selected_collection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                help="Exportiere den Chat als Markdown-Datei"
            )
        
        with col2:
            # Chat Statistiken
            total_chars = sum(len(msg["content"]) for msg in st.session_state.chat_history)
            st.metric("📊 Zeichen", f"{total_chars:,}")
        
        with col3:
            # Letzte Aktivität
            if st.session_state.chat_history:
                last_msg = st.session_state.chat_history[-1]
                if "timestamp" in last_msg:
                    try:
                        last_time = datetime.fromisoformat(last_msg["timestamp"])
                        st.caption(f"🕒 Letzte Nachricht: {last_time.strftime('%H:%M')}")
                    except:
                        pass

async def generate_rag_response_with_progress(question: str, collection_name: str, chroma_client, progress_indicator) -> str:
    """Generate RAG response with detailed progress updates."""
    from ui_components import ProgressStatus
    
    try:
        # Import des RAG Agents
        from rag_agent import run_rag_agent_entrypoint, RAGDeps
        import dotenv
        dotenv.load_dotenv()
        
        # Setup
        vertex_project_id = (
            st.secrets.get("GOOGLE_CLOUD_PROJECT") or 
            os.getenv("GOOGLE_CLOUD_PROJECT") or 
            "vertexai-408416"
        )
        vertex_location = (
            st.secrets.get("GOOGLE_CLOUD_LOCATION") or 
            os.getenv("GOOGLE_CLOUD_LOCATION") or 
            "us-central1"
        )
        
        # Gemini API Key setzen
        gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        
        # Google Cloud Credentials für lokale Entwicklung
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            json_files = [f for f in os.listdir('.') if f.startswith('vertexai-') and f.endswith('.json')]
            if json_files:
                credentials_path = json_files[0]
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        # Reranker konfigurieren
        possible_configs = [
            f"projects/{vertex_project_id}/locations/global/rankingConfigs/default_ranking_config",
            f"projects/{vertex_project_id}/locations/us-central1/rankingConfigs/default_ranking_config",
            f"projects/{vertex_project_id}/locations/global/rankingConfigs/default",
        ]
        reranker_model = possible_configs[0]
        
        # Reranker-Status bestimmen
        use_vertex_reranker_env = os.getenv("USE_VERTEX_RERANKER", "false").lower() == "true"
        use_reranker = bool(vertex_project_id) and use_vertex_reranker_env
        
        deps = RAGDeps(
            chroma_client=chroma_client,
            collection_name=collection_name,
            embedding_model_name="text-multilingual-embedding-002",
            embedding_provider="vertex_ai",
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            use_vertex_reranker=use_reranker,
            vertex_reranker_model=reranker_model
        )
        
        # Update progress: Start query variations
        progress_indicator.update_step("query_variations", ProgressStatus.RUNNING)
        await asyncio.sleep(0.1)  # Small delay for UI update
        
        # Update progress: HyDE generation
        progress_indicator.update_step("query_variations", ProgressStatus.COMPLETED)
        progress_indicator.update_step("hyde_generation", ProgressStatus.RUNNING)
        await asyncio.sleep(0.1)
        
        # Update progress: Database search
        progress_indicator.update_step("hyde_generation", ProgressStatus.COMPLETED)
        progress_indicator.update_step("database_search", ProgressStatus.RUNNING)
        await asyncio.sleep(0.1)
        
        # Update progress: Reranking
        progress_indicator.update_step("database_search", ProgressStatus.COMPLETED)
        if use_reranker:
            progress_indicator.update_step("reranking", ProgressStatus.RUNNING)
            await asyncio.sleep(0.1)
            progress_indicator.update_step("reranking", ProgressStatus.COMPLETED)
        else:
            progress_indicator.update_step("reranking", ProgressStatus.COMPLETED, "⚡ Erweiterte Filterung verwendet")
        
        # Update progress: Response generation
        progress_indicator.update_step("response_generation", ProgressStatus.RUNNING)
        
        # RAG Agent ausführen
        response = await run_rag_agent_entrypoint(
            question=question,
            deps=deps,
            llm_model="gemini-2.5-flash"
        )
        
        # Complete response generation
        progress_indicator.update_step("response_generation", ProgressStatus.COMPLETED)
        
        return response
        
    except Exception as e:
        # Update progress with error
        for step in progress_indicator.steps:
            if step.status == ProgressStatus.RUNNING:
                progress_indicator.update_step(step.name, ProgressStatus.ERROR, f"Fehler: {str(e)[:50]}...")
                break
        
        return f"❌ Fehler bei der Antwortgenerierung: {str(e)}"

def generate_rag_response(question: str, collection_name: str, chroma_client) -> str:
    """Generiere RAG-Antwort."""
    try:
        # Import des RAG Agents
        from rag_agent import run_rag_agent_entrypoint, RAGDeps
        
        # Setup - Vertex AI Konfiguration
        # Lade .env Datei explizit für lokale Entwicklung
        import dotenv
        dotenv.load_dotenv()
        
        # Versuche zuerst Streamlit Secrets, dann .env, dann Fallback
        vertex_project_id = (
            st.secrets.get("GOOGLE_CLOUD_PROJECT") or 
            os.getenv("GOOGLE_CLOUD_PROJECT") or 
            "vertexai-408416"
        )
        vertex_location = (
            st.secrets.get("GOOGLE_CLOUD_LOCATION") or 
            os.getenv("GOOGLE_CLOUD_LOCATION") or 
            "us-central1"
        )
        
        # Gemini API Key setzen
        gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        
        # Google Cloud Credentials für lokale Entwicklung
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            # Suche nach der JSON-Datei im aktuellen Verzeichnis
            json_files = [f for f in os.listdir('.') if f.startswith('vertexai-') and f.endswith('.json')]
            if json_files:
                credentials_path = json_files[0]
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                print(f"INFO: Google Cloud Credentials gesetzt: {credentials_path}")
            else:
                print("WARNING: Keine Google Cloud Credentials JSON-Datei gefunden")
        
        # Reranker konfigurieren - Alternative Konfigurationen testen
        reranker_model = None
        if vertex_project_id:
            # Verschiedene mögliche Ranking Config Pfade
            possible_configs = [
                f"projects/{vertex_project_id}/locations/global/rankingConfigs/default_ranking_config",
                f"projects/{vertex_project_id}/locations/us-central1/rankingConfigs/default_ranking_config",
                f"projects/{vertex_project_id}/locations/global/rankingConfigs/default",
            ]
            reranker_model = possible_configs[0]  # Verwende den ersten als Standard
        
        # Debug Info für Reranker-Status
        print(f"DEBUG: vertex_project_id = {vertex_project_id}")
        print(f"DEBUG: vertex_location = {vertex_location}")
        print(f"DEBUG: use_vertex_reranker = {bool(vertex_project_id)}")
        print(f"DEBUG: reranker_model = {reranker_model}")
        
        # Reranker-Status bestimmen - IAM-Berechtigung jetzt verfügbar
        use_vertex_reranker_env = os.getenv("USE_VERTEX_RERANKER", "false").lower() == "true"
        use_reranker = bool(vertex_project_id) and use_vertex_reranker_env  # Wieder aktiviert
        
        if vertex_project_id:
            print(f"INFO: Vertex AI Project gefunden: {vertex_project_id}")
            if use_vertex_reranker_env:
                print("INFO: Vertex AI Reranker aktiviert")
            else:
                print("INFO: Vertex AI Reranker deaktiviert (USE_VERTEX_RERANKER=false)")
        else:
            print("WARNING: Keine Vertex AI Project ID gefunden, verwende erweiterte Fallback-Filterung")
        
        deps = RAGDeps(
            chroma_client=chroma_client,
            collection_name=collection_name,
            embedding_model_name="text-multilingual-embedding-002",
            embedding_provider="vertex_ai",
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            use_vertex_reranker=use_reranker,  # Aktiviere basierend auf .env und Project ID
            vertex_reranker_model=reranker_model
        )
        
        # RAG Agent ausführen
        response = asyncio.run(run_rag_agent_entrypoint(
            question=question,
            deps=deps,
            llm_model="gemini-2.5-flash"
        ))
        
        return response
        
    except Exception as e:
        return f"❌ Fehler bei der Antwortgenerierung: {str(e)}"

if __name__ == "__main__":
    main()