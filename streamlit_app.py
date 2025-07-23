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

# Windows Event Loop Policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Streamlit Konfiguration
st.set_page_config(
    page_title="🤖 RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS für besseres Design
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
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Knowledge Assistant</h1>
        <p>Erstelle intelligente Wissensdatenbanken aus Webseiten und chatte mit deinen Daten</p>
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
    """Benutzerfreundliche Wissensdatenbank-Erstellung."""
    
    st.markdown("""
    <div class="feature-card">
        <h3>📖 Neue Wissensdatenbank erstellen</h3>
        <p>Erstelle eine durchsuchbare Wissensdatenbank aus Webseiten-Inhalten. Wähle die richtige Konfiguration für optimale Ergebnisse!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hilfe-Sektion
    with st.expander("💡 Hilfe: Welche Einstellungen soll ich wählen?"):
        st.markdown("""
        **🎯 Crawling-Typen erklärt:**
        
        - **📄 Einzelne Webseite**: Crawlt nur die angegebene URL
        - **🔗 Mehrere Seiten**: Folgt Links von der Startseite (konfigurierbare Tiefe)
        - **🗺️ Sitemap**: Crawlt alle URLs aus einer sitemap.xml Datei
        
        **⚙️ Wichtige Parameter:**
        
        - **Crawling-Tiefe**: Wie tief sollen Links verfolgt werden? (1 = nur Startseite, 2 = + verlinkte Seiten, etc.)
        - **Max. Seiten**: Begrenze die Anzahl der Seiten um Kosten und Zeit zu sparen
        - **Chunk-Größe**: Kleinere Chunks (800-1200) = präzisere Antworten, Größere Chunks (1500-2000) = mehr Kontext
        
        **💰 Tipp**: Starte mit wenigen Seiten (5-10) zum Testen, bevor du große Websites crawlst!
        """)
    
    with st.form("knowledge_creation"):
        # Basis-Konfiguration
        st.subheader("🌐 Website-Konfiguration")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            url = st.text_input(
                "Website URL:",
                placeholder="https://docs.example.com oder https://example.com/sitemap.xml",
                help="Vollständige URL der Website oder Sitemap"
            )
            
            name = st.text_input(
                "Name der Wissensdatenbank:",
                placeholder="z.B. 'Produktdokumentation' oder 'Firmen-Wiki'",
                help="Eindeutiger Name zur Identifikation deiner Wissensdatenbank"
            )
        
        with col2:
            source_type = st.selectbox(
                "Crawling-Typ:",
                ["Einzelne Webseite", "Mehrere Seiten", "Sitemap"],
                help="Bestimmt, wie die Website durchsucht wird"
            )
        
        # Dynamische Konfiguration basierend auf Typ
        st.subheader("⚙️ Crawling-Einstellungen")
        
        if source_type == "Einzelne Webseite":
            st.info("📄 Crawlt nur die angegebene URL - schnell und präzise")
            max_depth = 1
            max_pages = 1
            
        elif source_type == "Mehrere Seiten":
            st.warning("🔗 Folgt Links von der Startseite - kann viele Seiten finden!")
            
            col3, col4 = st.columns(2)
            with col3:
                max_depth = st.slider(
                    "Crawling-Tiefe:",
                    min_value=1, max_value=4, value=2,
                    help="1 = nur Startseite, 2 = + direkt verlinkte Seiten, 3 = + deren Links, etc."
                )
            with col4:
                max_pages = st.number_input(
                    "Max. Seiten:",
                    min_value=1, max_value=100, value=20,
                    help="Begrenze die Anzahl der Seiten um Zeit und Kosten zu sparen"
                )
            
            # Warnung bei hohen Werten
            if max_depth > 2 or max_pages > 50:
                st.warning("⚠️ Hohe Werte können zu langen Ladezeiten und hohen Kosten führen!")
                
        elif source_type == "Sitemap":
            st.success("🗺️ Crawlt alle URLs aus der Sitemap - Anzahl wird automatisch erkannt")
            max_depth = 1
            max_pages = None
            
            st.info("💡 Sitemap-URLs enden meist mit '/sitemap.xml' oder '/sitemap_index.xml'")
        
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
        
        # Geschätzte Kosten/Zeit
        if source_type == "Mehrere Seiten":
            estimated_time = max_pages * 2  # Grobe Schätzung: 2 Sekunden pro Seite
            st.info(f"⏱️ Geschätzte Dauer: ~{estimated_time} Sekunden für {max_pages} Seiten")
        
        submitted = st.form_submit_button("🚀 Wissensdatenbank erstellen", use_container_width=True)
        
        if submitted and url and name:
            create_knowledge_base_process(url, name, source_type, max_pages, chunk_size, auto_reduce, crawler_client, chroma_client, max_depth, max_concurrent)

def create_knowledge_base_process(url, name, source_type, max_pages, chunk_size, auto_reduce, crawler_client, chroma_client, max_depth=2, max_concurrent=5):
    """Prozess der Wissensdatenbank-Erstellung."""
    
    # Progress Container
    progress_container = st.container()
    
    with progress_container:
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
            elif source_type == "Mehrere Seiten":
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
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Fehler beim Erstellen der Wissensdatenbank: {str(e)}")
            
            if "Memory limit exceeded" in str(e):
                st.info("💡 Tipp: Versuche eine kleinere Chunk-Größe oder weniger Seiten.")

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
        
        # Chat Interface
        st.markdown("### 💬 Chatte mit deiner Wissensdatenbank")
        
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
        if prompt := st.chat_input("Stelle eine Frage zu deiner Wissensdatenbank..."):
            # User Message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Durchsuche Wissensdatenbank..."):
                    try:
                        response = generate_rag_response(prompt, selected_collection, chroma_client)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"❌ Entschuldigung, es gab einen Fehler: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        # Chat Reset
        if st.button("🗑️ Chat zurücksetzen"):
            st.session_state.chat_history = []
            st.rerun()

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