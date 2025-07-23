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

# ChromaDB und andere Imports
import chromadb
from crawler_client import CrawlerClient
from config import get_config, is_developer_mode, is_user_mode

# Windows Event Loop Policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Konfiguration laden
config = get_config()

# Streamlit Konfiguration
if is_user_mode():
    st.set_page_config(
        page_title="🤖 RAG Knowledge Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
else:
    st.set_page_config(
        page_title="RAG Agent - Developer Mode",
        page_icon="🤖",
        layout="wide"
    )

# Custom CSS für Benutzer-Modus
if config["custom_styling"]:
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .feature-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
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
        if is_developer_mode():
            st.error(f"❌ Fehler beim Laden der Google Cloud Credentials: {e}")
        return False

@st.cache_resource
def get_chroma_client():
    """Erstelle einen In-Memory ChromaDB Client."""
    try:
        client = chromadb.Client()
        if is_developer_mode():
            st.success("✅ ChromaDB In-Memory Client erfolgreich erstellt")
        return client
    except Exception as e:
        if is_developer_mode():
            st.error(f"❌ Fehler beim Erstellen des ChromaDB Clients: {e}")
        return None

@st.cache_resource
def get_crawler_client():
    """Erstelle den Modal.com Crawler Client."""
    try:
        base_url = st.secrets.get("MODAL_API_URL", "https://nico-gt91--crawl4ai-service")
        api_key = st.secrets.get("MODAL_API_KEY")
        
        if not api_key:
            if is_developer_mode():
                st.error("❌ MODAL_API_KEY nicht in Secrets gefunden")
            return None
            
        client = CrawlerClient(base_url=base_url, api_key=api_key)
        if is_developer_mode():
            st.success("✅ Modal.com Crawler Client erfolgreich erstellt")
        return client
    except Exception as e:
        if is_developer_mode():
            st.error(f"❌ Fehler beim Erstellen des Crawler Clients: {e}")
        return None

def main():
    # Header basierend auf Modus
    if is_user_mode():
        st.markdown("""
        <div class="main-header">
            <h1>🤖 RAG Knowledge Assistant</h1>
            <p>Erstelle intelligente Wissensdatenbanken aus Webseiten und chatte mit deinen Daten</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.title("🤖 RAG Agent - Developer Mode")
        st.markdown("Entwickler-Interface mit erweiterten Funktionen und Debug-Informationen")
    
    # Services initialisieren
    gcp_ok = setup_google_cloud_credentials()
    chroma_client = get_chroma_client()
    crawler_client = get_crawler_client()
    
    # Status-Anzeige für Entwickler
    if config["show_status_details"]:
        st.markdown("---")
        st.subheader("🔧 System Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Google Cloud")
            if gcp_ok:
                st.success("✅ Credentials geladen")
            else:
                st.warning("⚠️ Keine Credentials (optional)")
        
        with col2:
            st.subheader("ChromaDB")
            if chroma_client:
                st.success("✅ In-Memory Client aktiv")
            else:
                st.error("❌ Client nicht verfügbar")
        
        with col3:
            st.subheader("Modal.com Crawler")
            if crawler_client:
                st.success("✅ Crawler Client aktiv")
            else:
                st.error("❌ Crawler nicht verfügbar")
    
    # Prüfe kritische Services
    essential_services_ok = chroma_client and crawler_client
    
    if not essential_services_ok:
        if is_user_mode():
            st.error("🚨 Service nicht verfügbar. Bitte versuche es später erneut.")
        else:
            st.error("❌ Kritische Services (ChromaDB/Crawler) konnten nicht initialisiert werden")
        st.stop()
    
    if is_user_mode() and essential_services_ok:
        st.success("🎉 Alle Services bereit!")
    
    st.markdown("---")
    
    # Navigation basierend auf Modus
    if config["simplified_navigation"]:
        # Vereinfachte Navigation für Benutzer
        tab1, tab2 = st.tabs(["📚 Wissensdatenbank erstellen", "💬 Chat mit deinen Daten"])
        
        with tab1:
            create_knowledge_base_user(crawler_client, chroma_client)
        
        with tab2:
            chat_interface_user(chroma_client)
    else:
        # Vollständige Navigation für Entwickler
        tabs = ["📚 Wissensdatenbank erstellen", "🤖 RAG Chat"]
        if config["show_crawler_test"]:
            tabs.insert(0, "🕷️ Crawler Test")
        
        tab_objects = st.tabs(tabs)
        
        tab_index = 0
        if config["show_crawler_test"]:
            with tab_objects[tab_index]:
                crawler_test_interface(crawler_client)
            tab_index += 1
        
        with tab_objects[tab_index]:
            create_knowledge_base_developer(crawler_client, chroma_client)
        
        with tab_objects[tab_index + 1]:
            chat_interface_developer(chroma_client)

def create_knowledge_base_user(crawler_client, chroma_client):
    """Benutzerfreundliche Wissensdatenbank-Erstellung."""
    
    st.markdown("""
    <div class="feature-card">
        <h3>📖 Neue Wissensdatenbank erstellen</h3>
        <p>Füge eine Website-URL hinzu und erstelle eine durchsuchbare Wissensdatenbank</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("knowledge_creation"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            url = st.text_input(
                "🌐 Website URL:",
                placeholder="https://example.com",
                help="URL einer Webseite oder Sitemap"
            )
            
            name = st.text_input(
                "📝 Name der Wissensdatenbank:",
                placeholder="Meine Wissensdatenbank",
                help="Eindeutiger Name für deine Wissensdatenbank"
            )
        
        with col2:
            source_type = st.selectbox(
                "📄 Typ:",
                ["Einzelne Webseite", "Sitemap", "Mehrere Seiten"],
                help="Wähle den Typ der Quelle"
            )
            
            if source_type == "Mehrere Seiten":
                max_pages = st.number_input("Max. Seiten:", 1, 50, 10)
            else:
                max_pages = 1
        
        # Erweiterte Optionen (eingeklappt)
        with st.expander("⚙️ Erweiterte Einstellungen"):
            col3, col4 = st.columns(2)
            with col3:
                chunk_size = st.slider("Chunk-Größe:", 500, 2000, 1200, 
                                     help="Größere Chunks = mehr Kontext, kleinere = präziser")
            with col4:
                auto_reduce = st.checkbox("Automatische Optimierung", value=True,
                                        help="Optimiert automatisch für beste Performance")
        
        submitted = st.form_submit_button("🚀 Wissensdatenbank erstellen", use_container_width=True)
        
        if submitted and url and name:
            create_knowledge_base_process(url, name, source_type, max_pages, chunk_size, auto_reduce, crawler_client, chroma_client, user_mode=True)

def create_knowledge_base_developer(crawler_client, chroma_client):
    """Entwickler-Version der Wissensdatenbank-Erstellung."""
    
    st.subheader("📚 Wissensdatenbank erstellen")
    
    # Import der Ingestion-Pipeline
    from insert_docs_streamlit import run_ingestion_sync, IngestionProgress
    
    with st.form("ingestion_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ingestion_url = st.text_input(
                "URL oder Pfad:",
                value="https://de.wikipedia.org/wiki/Künstliche_Intelligenz",
                help="URL einer Webseite, Sitemap oder Textdatei"
            )
            
            collection_name_input = st.text_input(
                "Collection Name (optional):",
                help="Leer lassen für automatische Generierung"
            )
        
        with col2:
            source_type = st.selectbox(
                "Quelle-Typ:",
                ["Webseite", "Sitemap", "Textdatei"]
            )
            
            if source_type == "Webseite":
                max_depth = st.slider("Max. Tiefe:", 1, 5, 2)
                limit = st.number_input("Seiten-Limit:", 1, 100, 20)
            else:
                max_depth = 1
                limit = 50
        
        # Erweiterte Optionen
        with st.expander("🔧 Erweiterte Optionen"):
            chunk_size = st.slider("Chunk-Größe:", 500, 3000, 1500, 
                                 help="Kleinere Chunks = weniger Memory, aber möglicherweise weniger Kontext")
            chunk_overlap = st.slider("Chunk-Überlappung:", 50, 500, 150)
            max_concurrent = st.slider("Max. parallele Prozesse:", 1, 10, 5)
            
            # Memory-Management Optionen
            st.subheader("💾 Memory-Management")
            auto_reduce = st.checkbox("Auto-Reduktion bei Memory-Problemen", value=True,
                                    help="Reduziert automatisch die Anzahl der Chunks bei Memory-Problemen")
            max_chunks = st.number_input("Max. Chunks (0 = unbegrenzt):", 0, 10000, 0,
                                       help="Begrenzt die maximale Anzahl der Chunks für Memory-Management")
        
        submitted = st.form_submit_button(
            "🚀 Wissensdatenbank erstellen",
            disabled=st.session_state.get('ingestion_in_progress', False)
        )
        
        if submitted:
            st.session_state.ingestion_in_progress = True
            create_knowledge_base_process(
                ingestion_url, collection_name_input, source_type, limit, 
                chunk_size, auto_reduce, crawler_client, chroma_client, 
                user_mode=False, chunk_overlap=chunk_overlap, max_depth=max_depth, 
                max_concurrent=max_concurrent, max_chunks=max_chunks
            )

def create_knowledge_base_process(url, name, source_type, max_pages, chunk_size, auto_reduce, crawler_client, chroma_client, user_mode=True, **kwargs):
    """Prozess der Wissensdatenbank-Erstellung."""
    
    # Progress Container
    progress_container = st.container()
    
    with progress_container:
        if user_mode:
            st.info("🔄 Erstelle deine Wissensdatenbank...")
        else:
            st.info("🔄 Erstelle Wissensdatenbank... Dies kann einige Minuten dauern.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import der Ingestion-Pipeline
            from insert_docs_streamlit import run_ingestion_sync, IngestionProgress
            
            # Custom Progress Tracker
            class CustomProgress(IngestionProgress):
                def update(self, step: int, message: str):
                    progress = int((step / 5) * 100)
                    progress_bar.progress(progress)
                    if user_mode:
                        # Vereinfachte Nachrichten für Benutzer
                        user_messages = {
                            1: "Bereite vor...",
                            2: "Lade Webseiten...",
                            3: "Verarbeite Inhalte...",
                            4: "Erstelle Embeddings...",
                            5: "Speichere in Datenbank..."
                        }
                        status_text.text(user_messages.get(step, message))
                    else:
                        status_text.text(f"Schritt {step}/5: {message}")
            
            progress = CustomProgress()
            
            # Konfiguration basierend auf Typ
            if source_type in ["Sitemap"]:
                max_depth = 1
                limit = None
            elif source_type in ["Mehrere Seiten", "Webseite"]:
                max_depth = kwargs.get('max_depth', 2)
                limit = max_pages
            else:
                max_depth = 1
                limit = 1
            
            # Parameter für Entwickler-Modus
            chunk_overlap = kwargs.get('chunk_overlap', 150)
            max_concurrent = kwargs.get('max_concurrent', 5)
            max_chunks = kwargs.get('max_chunks', None)
            
            # Ingestion ausführen
            result = run_ingestion_sync(
                url=url,
                collection_name=name,
                crawler_client=crawler_client,
                chroma_client=chroma_client,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_depth=max_depth,
                max_concurrent=max_concurrent,
                limit=limit,
                progress=progress,
                auto_reduce=auto_reduce,
                max_chunks=max_chunks if max_chunks and max_chunks > 0 else None
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
            
            if user_mode:
                st.info("💡 Wechsle zum 'Chat' Tab um mit deiner Wissensdatenbank zu interagieren!")
            
            # Session State zurücksetzen
            st.session_state.ingestion_in_progress = False
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Fehler beim Erstellen der Wissensdatenbank: {str(e)}")
            
            if "Memory limit exceeded" in str(e):
                st.info("💡 Tipp: Versuche eine kleinere Chunk-Größe oder weniger Seiten.")
            
            st.session_state.ingestion_in_progress = False

def chat_interface_user(chroma_client):
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
        display_collection_info_user(selected_collection, chroma_client)
        chat_interface_common(selected_collection, chroma_client, user_mode=True)

def chat_interface_developer(chroma_client):
    """Entwickler-Version der Chat-Oberfläche."""
    
    st.subheader("🤖 RAG Chat")
    
    # Verfügbare Collections laden
    try:
        collections = [c.name for c in chroma_client.list_collections()]
    except:
        collections = []
    
    if not collections:
        st.info("ℹ️ Keine Collections gefunden. Erstelle zuerst eine Wissensdatenbank.")
        return
    
    # Collection Auswahl
    selected_collection = st.selectbox(
        "Wissensdatenbank auswählen:",
        collections
    )
    
    if selected_collection:
        display_collection_info_developer(selected_collection, chroma_client)
        chat_interface_common(selected_collection, chroma_client, user_mode=False)

def display_collection_info_user(collection_name, chroma_client):
    """Zeigt Collection-Info für Benutzer an."""
    collection = chroma_client.get_collection(collection_name)
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
        st.markdown(f"""
        <div class="metric-card">
            <h3>📄 {doc_count}</h3>
            <p>Dokumente</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 {chunk_count}</h3>
            <p>Text-Chunks</p>
        </div>
        """, unsafe_allow_html=True)

def display_collection_info_developer(collection_name, chroma_client):
    """Zeigt Collection-Info für Entwickler an."""
    collection = chroma_client.get_collection(collection_name)
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Collection", collection_name)
    with col2:
        st.metric("Dokumente", doc_count)
    with col3:
        st.metric("Chunks", chunk_count)

def chat_interface_common(selected_collection, chroma_client, user_mode=True):
    """Gemeinsame Chat-Interface-Logik."""
    
    if user_mode:
        st.markdown("---")
        st.markdown("### 💬 Chatte mit deiner Wissensdatenbank")
    else:
        st.subheader("💬 Chat")
    
    # Chat History
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat Container
    chat_container = st.container()
    
    with chat_container:
        # Zeige Chat History
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat Input
    placeholder = "Stelle eine Frage zu deiner Wissensdatenbank..." if user_mode else "Stelle eine Frage..."
    
    if prompt := st.chat_input(placeholder):
        # User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Assistant Response
        with st.chat_message("assistant"):
            spinner_text = "🤔 Durchsuche Wissensdatenbank..." if user_mode else "Suche in der Wissensdatenbank..."
            
            with st.spinner(spinner_text):
                try:
                    response = generate_rag_response(prompt, selected_collection, chroma_client, user_mode)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    if user_mode:
                        error_msg = "❌ Entschuldigung, es gab einen Fehler bei der Antwortgenerierung."
                    else:
                        error_msg = f"❌ Fehler bei der RAG-Suche: {str(e)}"
                    
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Chat Reset
    reset_text = "🗑️ Chat zurücksetzen" if user_mode else "🗑️ Chat History löschen"
    if st.button(reset_text):
        st.session_state.chat_history = []
        st.rerun()

def generate_rag_response(question: str, collection_name: str, chroma_client, user_mode=True) -> str:
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
        if user_mode:
            return "❌ Entschuldigung, ich konnte keine Antwort generieren. Bitte versuche es erneut."
        else:
            return f"❌ Fehler bei der Antwortgenerierung: {str(e)}"

def crawler_test_interface(crawler_client):
    """Crawler-Test-Interface für Entwickler."""
    
    st.subheader("🕷️ Crawler Test")
    
    with st.form("crawler_test"):
        test_url = st.text_input(
            "Test URL:",
            value="https://example.com",
            help="URL zum Testen des Crawlers"
        )
        
        test_type = st.selectbox(
            "Test-Typ:",
            ["Single URL", "Batch URLs", "Recursive", "Sitemap"]
        )
        
        submitted = st.form_submit_button("🧪 Test starten")
        
        if submitted and test_url:
            with st.spinner("Teste Crawler..."):
                try:
                    if test_type == "Single URL":
                        result = asyncio.run(crawler_client.crawl_single(test_url))
                    elif test_type == "Recursive":
                        result = asyncio.run(crawler_client.crawl_recursive(test_url, max_depth=2, limit=5))
                    elif test_type == "Sitemap":
                        result = asyncio.run(crawler_client.crawl_sitemap(test_url))
                    else:
                        result = {"error": "Test-Typ nicht implementiert"}
                    
                    st.success("✅ Crawler-Test erfolgreich!")
                    st.json(result)
                    
                except Exception as e:
                    st.error(f"❌ Crawler-Test fehlgeschlagen: {str(e)}")

if __name__ == "__main__":
    main()