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
from typing import Optional
from datetime import datetime

# ChromaDB und andere Imports
import chromadb
from crawler_client import CrawlerClient

# UX Components Import
from ux_components import URLValidator
from state_manager import get_state_manager

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
    except Exception:
        return False

@st.cache_resource
def get_chroma_client():
    """Erstelle einen In-Memory ChromaDB Client."""
    try:
        client = chromadb.Client()
        return client
    except Exception:
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
    except Exception:
        return None

def main():
    # Dezenter Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 1.5rem;">
        <h2 style="color: #667eea; margin-bottom: 0.5rem;">🤖 CraCha - Crawl Chat Agent</h2>
        <p style="color: #666; font-size: 0.9rem; margin: 0;">Intelligente Wissensdatenbanken aus Webseiten erstellen</p>
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
    with st.expander("💡 Hilfe: Welche Einstellungen soll ich wählen?"):
        st.markdown("""
        **🎯 Crawling-Typen erklärt:**
        
        - **🌐 Website Crawling**: Crawlt Webseiten mit konfigurierbarer Tiefe und Seitenzahl
        - **🗺️ Sitemap**: Crawlt alle URLs aus einer sitemap.xml Datei automatisch
        
        **⚙️ Parameter im Detail:**
        
        - **Crawling-Tiefe**: Technisch gesehen die Rekursionstiefe beim Verfolgen von Links (1 = keine Rekursion, 2 = eine Ebene tief, etc.)
        - **Max. Seiten**: Technische Begrenzung der zu crawlenden URLs um Ressourcen zu schonen
        - **Chunk-Größe**: Technische Textaufteilung - kleinere Chunks (800-1200) für präzise Suche, größere (1500-2000) für mehr Kontext
        - **Parallele Prozesse**: Technische Parallelisierung - höhere Werte bedeuten mehr gleichzeitige Crawling-Threads
        
        **💰 Kostentipp**: Starte mit wenigen Seiten (5-10) zum Testen, bevor du große Websites crawlst!
        
        **🔧 Empfohlene Einstellungen:**
        
        - **Einzelne Seite testen**: Tiefe=1, Seiten=1
        - **Kleine Website**: Tiefe=2, Seiten=10-20  
        - **Große Website**: Tiefe=2-3, Seiten=50+ (Vorsicht bei Kosten!)
        - **Vollständige Website**: Sitemap verwenden (automatische Erkennung)
        """)
    
    with st.form("knowledge_creation"):
        # Basis-Konfiguration
        st.subheader("🌐 Website-Konfiguration")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Initialize URL validator
            if 'url_validator' not in st.session_state:
                st.session_state.url_validator = URLValidator(timeout=10, debounce_delay=0.5)
            
            url_validator = st.session_state.url_validator
            
            # URL input with real-time validation
            url = st.text_input(
                "Website URL: *",
                placeholder="https://docs.example.com oder https://example.com/sitemap.xml",
                help="Vollständige URL der Website oder Sitemap (Pflichtfeld)",
                key="main_url_input"
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
            source_type = st.selectbox(
                "Crawling-Typ:",
                ["Website Crawling", "Sitemap"],
                help="Website Crawling = konfigurierbare Tiefe und Seitenzahl, Sitemap = automatische Erkennung aller URLs"
            )
            
            # URL Status Indicator (nur bei Eingabe)
            if url and url.strip():
                url_validation = getattr(st.session_state, 'url_validation_result', None)
                if url_validation:
                    status_indicator = url_validator.get_validation_status_indicator(url_validation)
                    
                    if url_validation.is_valid:
                        if url_validation.warning_message:
                            st.warning(f"{status_indicator} URL Status: Gültig mit Warnung")
                        else:
                            st.success(f"{status_indicator} URL Status: Gültig")
                    else:
                        st.error(f"{status_indicator} URL Status: Ungültig")

        
        # Crawling-Einstellungen für alle Typen
        st.subheader("⚙️ Crawling-Einstellungen")
        
        # Typ-spezifische Informationen (nur für Sitemap)
        if source_type == "Sitemap":
            st.info("💡 Sitemap-URLs enden meist mit '/sitemap.xml' oder '/sitemap_index.xml'")
        
        # Gemeinsame Crawling-Einstellungen für alle Typen
        col3, col4 = st.columns(2)
        
        with col3:
            if source_type == "Website Crawling":
                max_depth = st.slider(
                    "Wie tief soll gecrawlt werden?",
                    min_value=1, max_value=4, value=1,
                    help="Technisch: Maximale Rekursionstiefe beim Verfolgen von Links (1 = keine Rekursion, 2 = eine Ebene tief, etc.)"
                )
                
                # Einfache Erklärung der Tiefe
                if max_depth == 1:
                    st.caption("🎯 Nur die angegebene URL wird gecrawlt")
                elif max_depth == 2:
                    st.caption("🎯 Angegebene URL + alle direkt verlinkten Seiten")
                elif max_depth == 3:
                    st.caption("🎯 Tiefes Crawling: Folgt Links 2 Ebenen tief")
                else:
                    st.caption("🎯 Sehr tiefes Crawling: Kann sehr viele Seiten finden!")
            else:
                max_depth = 1
                st.info("Tiefe wird bei Sitemaps automatisch auf 1 gesetzt")
        
        with col4:
            if source_type == "Sitemap":
                st.info("Bei Sitemaps wird die Anzahl der Seiten automatisch erkannt")
                max_pages = None
                st.metric("Seiten-Limit", "Automatisch")
            else:
                max_pages = st.number_input(
                    "Wie viele Seiten maximal crawlen?",
                    min_value=1, max_value=100, 
                    value=1,
                    help="Technisch: Maximale Anzahl zu crawlender URLs um Ressourcen zu schonen"
                )
                
                # Einfache Erklärung der Seitenzahl
                if max_pages == 1:
                    st.caption("🎯 Nur eine Seite wird gecrawlt")
                elif max_pages <= 10:
                    st.caption("🎯 Kleine Anzahl - gut zum Testen")
                elif max_pages <= 50:
                    st.caption("🎯 Mittlere Anzahl - für normale Websites")
                else:
                    st.caption("🎯 Große Anzahl - kann teuer werden!")
        
        # Warnung bei hohen Werten (für Website Crawling)
        if source_type == "Website Crawling" and (max_depth > 2 or (max_pages and max_pages > 50)):
            st.warning("⚠️ Hohe Werte können zu langen Ladezeiten und hohen Kosten führen!")
        
        # Erweiterte Einstellungen
        with st.expander("🔧 Erweiterte Einstellungen"):
            col5, col6 = st.columns(2)
            
            with col5:
                chunk_size = st.slider(
                    "Text-Chunk-Größe:",
                    min_value=500, max_value=2500, value=1200,
                    help="Kleinere Chunks = präzisere Antworten, Größere = mehr Kontext pro Antwort"
                )
                
                # Chunk-Größe Empfehlung
                if chunk_size < 800:
                    st.info("📝 Kleine Chunks: Sehr präzise, aber möglicherweise wenig Kontext")
                elif chunk_size > 1800:
                    st.info("📚 Große Chunks: Viel Kontext, aber möglicherweise weniger präzise")
                else:
                    st.success("✅ Optimale Chunk-Größe für die meisten Anwendungen")
            
            with col6:
                auto_reduce = st.checkbox(
                    "Automatische Optimierung",
                    value=True,
                    help="Reduziert automatisch die Datenmenge bei Memory-Problemen"
                )
                
                max_concurrent = st.slider(
                    "Parallele Prozesse:",
                    min_value=1, max_value=10, value=5,
                    help="Mehr Prozesse = schneller, aber höhere Serverlast"
                )
        
        # Geschätzte Kosten/Zeit für alle Typen
        if source_type == "Website Crawling" and max_pages:
            if max_depth > 1:
                estimated_pages = min(max_pages, 10 ** (max_depth - 1) * 5)  # Grobe Schätzung
                estimated_time = estimated_pages * 2  # Grobe Schätzung: 2 Sekunden pro Seite
                st.info(f"⏱️ Geschätzte Dauer: ~{estimated_time} Sekunden für ca. {estimated_pages} Seiten")
            else:
                estimated_time = max_pages * 2
                st.info(f"⏱️ Geschätzte Dauer: ~{estimated_time} Sekunden für {max_pages} Seite(n)")
        elif source_type == "Sitemap":
            st.info("⏱️ Geschätzte Dauer: Abhängig von der Anzahl der URLs in der Sitemap")
        
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
                create_knowledge_base_process(url, name, source_type, max_pages, chunk_size, auto_reduce, crawler_client, chroma_client, max_depth, max_concurrent)

def create_knowledge_base_process(url, name, source_type, max_pages, chunk_size, auto_reduce, crawler_client, chroma_client, max_depth=2, max_concurrent=5):
    """Prozess der Wissensdatenbank-Erstellung."""
    
    # Leere die Seite und zeige nur Progress
    st.empty()
    
    # Progress Container - isoliert von der Form
    with st.container():
        st.markdown("---")
        st.info("🔄 Erstelle deine Wissensdatenbank...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import der Ingestion-Pipeline
            from insert_docs_streamlit import run_ingestion_sync, IngestionProgress
            
            # Custom Progress Tracker
            class UserProgress(IngestionProgress):
                def update(self, step: int, message: str):
                    progress = int((step / 5) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Schritt {step}/5: {message}")
            
            progress = UserProgress()
            
            # Konfiguration basierend auf Typ
            if source_type == "Sitemap":
                depth = 1
                limit = None
            elif source_type == "Website Crawling":
                depth = max_depth
                limit = max_pages
            else:
                depth = 1
                limit = 1
            
            # Ingestion ausführen
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
            
            # Erfolg anzeigen
            progress_bar.progress(100)
            status_text.empty()
            
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
            progress_bar.empty()
            status_text.empty()
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
    except:
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
        
        # Berechne Dokumente
        try:
            sample_metadata = collection.get(limit=min(chunk_count, 1000), include=["metadatas"])
            unique_urls = set()
            for metadata in sample_metadata["metadatas"]:
                if metadata and "url" in metadata:
                    unique_urls.add(metadata["url"])
            doc_count = len(unique_urls)
        except:
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
    
    # CSS für verbessertes Chat-Layout mit Auto-Scroll
    st.markdown("""
    <style>
    /* Chat Input fixiert am unteren Rand */
    .stChatInput {
        position: sticky !important;
        bottom: 0 !important;
        background-color: white !important;
        padding: 1rem 0 !important;
        border-top: 2px solid #667eea !important;
        z-index: 999 !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1) !important;
    }
    
    /* Chat Input Field Styling */
    .stChatInput > div > div > div > div {
        border-radius: 25px !important;
        border: 2px solid #667eea !important;
        box-shadow: 0 2px 5px rgba(102, 126, 234, 0.2) !important;
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
                        except:
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
        
        # Assistant Response mit Typing Indicator
        with st.chat_message("assistant"):
            # Typing Indicator
            typing_placeholder = st.empty()
            typing_placeholder.markdown("""
            <div class="typing-indicator">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span>Durchsuche Wissensdatenbank...</span>
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

def generate_rag_response(question: str, collection_name: str, chroma_client) -> str:
    """Generiere RAG-Antwort."""
    try:
        # Import des RAG Agents
        from rag_agent import run_rag_agent_entrypoint, RAGDeps
        
        # Setup
        vertex_project_id = st.secrets.get("GOOGLE_CLOUD_PROJECT")
        vertex_location = st.secrets.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Gemini API Key setzen
        gemini_key = st.secrets.get("GEMINI_API_KEY")
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        
        deps = RAGDeps(
            chroma_client=chroma_client,
            collection_name=collection_name,
            embedding_model_name="text-multilingual-embedding-002",
            embedding_provider="vertex_ai",
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            use_vertex_reranker=False,
            vertex_reranker_model=None
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