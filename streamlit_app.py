# HACK: SQLite3 Kompatibilität für Streamlit Community Cloud
# WICHTIG: Dies muss ganz am Anfang stehen, vor allen anderen Imports!
import sys

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ SQLite3 Kompatibilität für Streamlit Cloud aktiviert")
except ImportError:
    # Lokal verwenden wir das normale sqlite3
    print("ℹ️ Verwende lokales sqlite3 (normale Entwicklungsumgebung)")

import streamlit as st
import os
import asyncio
import base64
import tempfile
import json
from typing import Optional, Dict, Any

# Jetzt können wir ChromaDB importieren
import chromadb

# Andere Imports
from crawler_client import CrawlerClient

# Windows Event Loop Policy für lokale Entwicklung
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Streamlit Konfiguration
st.set_page_config(
    page_title="RAG Agent - Cloud Version",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def setup_google_cloud_credentials():
    """Setup Google Cloud Credentials aus Streamlit Secrets."""
    try:
        # Prüfe, ob Credentials als Base64 in Secrets gespeichert sind
        creds_json_b64 = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if creds_json_b64:
            # Dekodiere und speichere temporär
            creds_json = base64.b64decode(creds_json_b64).decode('utf-8')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as temp:
                temp.write(creds_json)
                temp_filename = temp.name
            
            # Setze die Umgebungsvariable
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_filename
            st.success("✅ Google Cloud Credentials erfolgreich geladen")
            return True
        else:
            st.warning("⚠️ Keine Google Cloud Credentials in Secrets gefunden")
            return False
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Google Cloud Credentials: {e}")
        return False

@st.cache_resource
def get_chroma_client():
    """Erstelle einen In-Memory ChromaDB Client für Streamlit Cloud."""
    try:
        # Verwende In-Memory Client statt persistenten Client
        client = chromadb.Client()
        st.success("✅ ChromaDB In-Memory Client erfolgreich erstellt")
        return client
    except Exception as e:
        st.error(f"❌ Fehler beim Erstellen des ChromaDB Clients: {e}")
        return None

@st.cache_resource
def get_crawler_client():
    """Erstelle den Modal.com Crawler Client."""
    try:
        # API-Konfiguration aus Streamlit Secrets
        base_url = st.secrets.get("MODAL_API_URL", "https://nico-gt91--crawl4ai-service")
        api_key = st.secrets.get("MODAL_API_KEY")
        
        if not api_key:
            st.error("❌ MODAL_API_KEY nicht in Streamlit Secrets konfiguriert")
            return None
        
        client = CrawlerClient(base_url=base_url, api_key=api_key)
        
        # Test Health Check
        health = client.health_check_sync()
        if health.get("status") == "healthy":
            st.success("✅ Modal.com Crawler Service verbunden")
            return client
        else:
            st.error("❌ Modal.com Crawler Service nicht erreichbar")
            return None
            
    except Exception as e:
        st.error(f"❌ Fehler beim Verbinden mit Modal.com Service: {e}")
        return None

def init_session_state():
    """Initialisiere Session State Variablen."""
    # Crawler Test State
    if 'crawling_in_progress' not in st.session_state:
        st.session_state.crawling_in_progress = False
    if 'crawl_result' not in st.session_state:
        st.session_state.crawl_result = None
    if 'crawl_error' not in st.session_state:
        st.session_state.crawl_error = None
    
    # Ingestion State
    if 'ingestion_in_progress' not in st.session_state:
        st.session_state.ingestion_in_progress = False
    if 'ingestion_result' not in st.session_state:
        st.session_state.ingestion_result = None
    if 'ingestion_error' not in st.session_state:
        st.session_state.ingestion_error = None

def main():
    st.title("🤖 RAG Agent - Cloud Version")
    st.markdown("---")
    
    # Initialisiere Session State
    init_session_state()
    
    # Setup Services
    st.header("🔧 Service Setup")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Google Cloud")
        gcp_ok = setup_google_cloud_credentials()
    
    with col2:
        st.subheader("ChromaDB")
        chroma_client = get_chroma_client()
        chroma_ok = chroma_client is not None
    
    with col3:
        st.subheader("Modal.com Crawler")
        crawler_client = get_crawler_client()
        crawler_ok = crawler_client is not None
    
    # Zeige Gesamtstatus
    essential_services_ok = chroma_ok and crawler_ok
    
    if essential_services_ok:
        if gcp_ok:
            st.success("🎉 Alle Services erfolgreich initialisiert!")
        else:
            st.warning("⚠️ Crawler und ChromaDB funktionieren. Google Cloud optional für lokale Tests.")
    else:
        st.error("❌ Kritische Services (ChromaDB/Crawler) konnten nicht initialisiert werden")
        st.stop()
    
    st.markdown("---")
    
    # Tabs für verschiedene Funktionen
    tab1, tab2, tab3 = st.tabs(["🕷️ Crawler Test", "📚 Wissensdatenbank erstellen", "🤖 RAG Chat"])
    
    with tab1:
        st.header("🕷️ Crawler Test")
        
        url_to_test = st.text_input(
            "URL zum Testen:",
            value="https://de.wikipedia.org/wiki/Künstliche_Intelligenz",
            help="Gib eine URL ein, um den Crawler zu testen"
        )
        
        crawl_type = st.selectbox(
            "Crawling-Typ:",
            ["Single URL", "Batch URLs", "Recursive", "Sitemap"]
        )
    
    with tab2:
        st.header("📚 Wissensdatenbank erstellen")
        
        # Import der Ingestion-Pipeline
        from insert_docs_streamlit import run_ingestion_sync, IngestionProgress
        
        with st.form("ingestion_form"):
            st.subheader("🔧 Konfiguration")
            
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
                chunk_size = st.slider("Chunk-Größe:", 500, 3000, 1500)
                chunk_overlap = st.slider("Chunk-Überlappung:", 50, 500, 150)
                max_concurrent = st.slider("Max. parallele Prozesse:", 1, 10, 5)
            
            submitted = st.form_submit_button(
                "🚀 Wissensdatenbank erstellen",
                disabled=st.session_state.get('ingestion_in_progress', False)
            )
        
        if submitted and ingestion_url:
            st.session_state.ingestion_in_progress = True
            st.session_state.ingestion_result = None
            st.session_state.ingestion_error = None
            st.rerun()
        elif submitted:
            st.warning("Bitte gib eine URL ein!")
        
        # Ingestion Progress
        if st.session_state.get('ingestion_in_progress', False):
            st.info("🔄 Erstelle Wissensdatenbank... Dies kann einige Minuten dauern.")
            
            progress = IngestionProgress()
            
            try:
                result = run_ingestion_sync(
                    url=ingestion_url,
                    collection_name=collection_name_input,
                    crawler_client=crawler_client,
                    chroma_client=chroma_client,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_depth=max_depth,
                    max_concurrent=max_concurrent,
                    limit=limit,
                    progress=progress
                )
                
                st.session_state.ingestion_result = result
                
            except Exception as e:
                st.session_state.ingestion_error = str(e)
            
            finally:
                st.session_state.ingestion_in_progress = False
                st.rerun()
        
        # Ingestion Results
        if st.session_state.get('ingestion_result'):
            result = st.session_state.ingestion_result
            st.success("✅ Wissensdatenbank erfolgreich erstellt!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Collection", result.get('collection_name', 'N/A'))
            with col2:
                st.metric("Dokumente", result.get('documents_crawled', 0))
            with col3:
                st.metric("Chunks", result.get('chunks_created', 0))
            with col4:
                st.metric("Embeddings", result.get('embeddings_generated', 0))
            
            if st.button("🗑️ Ergebnis löschen"):
                st.session_state.ingestion_result = None
                st.rerun()
        
        # Ingestion Errors
        if st.session_state.get('ingestion_error'):
            st.error(f"❌ Fehler beim Erstellen der Wissensdatenbank: {st.session_state.ingestion_error}")
            
            if st.button("🗑️ Fehler löschen"):
                st.session_state.ingestion_error = None
                st.rerun()
    
    with tab3:
        st.header("🤖 RAG Chat")
        
        # Verfügbare Collections anzeigen
        try:
            collections = chroma_client.list_collections()
            collection_names = [c.name for c in collections]
            
            if collection_names:
                selected_collection = st.selectbox(
                    "Wissensdatenbank auswählen:",
                    collection_names,
                    help="Wähle eine Wissensdatenbank für den Chat aus"
                )
                
                # Collection Info anzeigen
                if selected_collection:
                    collection = chroma_client.get_collection(selected_collection)
                    doc_count = collection.count()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Collection", selected_collection)
                    with col2:
                        st.metric("Dokumente", doc_count)
                    
                    # Chat Interface
                    st.subheader("💬 Chat")
                    
                    # Chat History initialisieren
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    # Chat History anzeigen
                    for message in st.session_state.chat_history:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    
                    # Chat Input
                    if prompt := st.chat_input("Stelle eine Frage zur Wissensdatenbank..."):
                        # User Message hinzufügen
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        
                        # RAG Response generieren
                        with st.chat_message("assistant"):
                            with st.spinner("Suche in der Wissensdatenbank..."):
                                try:
                                    # Import der Query-Funktion mit korrekten Embeddings
                                    from insert_docs_streamlit import query_collection_sync
                                    
                                    # Suche mit korrekten Embedding-Dimensionen
                                    results = query_collection_sync(
                                        collection=collection,
                                        query_text=prompt,
                                        n_results=5
                                    )
                                    
                                    if results['documents'] and results['documents'][0]:
                                        # Import der verbesserten RAG-Funktion
                                        from insert_docs_streamlit import generate_rag_response
                                        
                                        # Intelligente RAG-Antwort generieren
                                        response = generate_rag_response(
                                            query=prompt,
                                            search_results=results,
                                            collection_name=selected_collection
                                        )
                                        
                                        st.markdown(response)
                                        
                                        # Response zur Chat History hinzufügen
                                        st.session_state.chat_history.append({
                                            "role": "assistant", 
                                            "content": response
                                        })
                                        
                                    else:
                                        error_msg = "Keine relevanten Informationen in der Wissensdatenbank gefunden."
                                        st.error(error_msg)
                                        st.session_state.chat_history.append({
                                            "role": "assistant", 
                                            "content": error_msg
                                        })
                                        
                                except Exception as e:
                                    error_msg = f"Fehler bei der Suche: {str(e)}"
                                    st.error(error_msg)
                                    st.session_state.chat_history.append({
                                        "role": "assistant", 
                                        "content": error_msg
                                    })
                    
                    # Chat History löschen
                    if st.button("🗑️ Chat History löschen"):
                        st.session_state.chat_history = []
                        st.rerun()
                        
            else:
                st.info("📝 Keine Wissensdatenbanken verfügbar. Erstelle zuerst eine Wissensdatenbank im Tab 'Wissensdatenbank erstellen'.")
                
        except Exception as e:
            st.error(f"Fehler beim Laden der Collections: {e}")
    
    # Crawling Button
    if st.button("🚀 Crawling starten", disabled=st.session_state.crawling_in_progress):
        if url_to_test:
            st.session_state.crawling_in_progress = True
            st.session_state.crawl_result = None
            st.session_state.crawl_error = None
            st.rerun()
        else:
            st.warning("Bitte gib eine URL ein!")
    
    # Crawling Progress
    if st.session_state.crawling_in_progress:
        st.info("🔄 Crawling läuft...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initialisiere Crawling...")
            progress_bar.progress(10)
            
            # Führe Crawling basierend auf Typ aus
            if crawl_type == "Single URL":
                result = crawler_client.crawl_single_sync(url_to_test)
            elif crawl_type == "Batch URLs":
                # Für Demo: Verwende nur eine URL
                result = crawler_client.crawl_batch_sync([url_to_test])
            elif crawl_type == "Recursive":
                result = crawler_client.crawl_recursive_sync(url_to_test, max_depth=2, limit=10)
            elif crawl_type == "Sitemap":
                result = crawler_client.crawl_sitemap_sync(url_to_test)
            
            progress_bar.progress(100)
            status_text.text("Crawling abgeschlossen!")
            
            # Speichere Ergebnis
            st.session_state.crawl_result = result
            
        except Exception as e:
            st.session_state.crawl_error = str(e)
            progress_bar.progress(0)
            status_text.text("Crawling fehlgeschlagen!")
        
        finally:
            st.session_state.crawling_in_progress = False
            st.rerun()
    
    # Ergebnisse anzeigen
    if st.session_state.crawl_result:
        st.success("✅ Crawling erfolgreich abgeschlossen!")
        
        result = st.session_state.crawl_result
        
        # Zeige Zusammenfassung
        if crawl_type == "Single URL":
            if result.get("success"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status", "Erfolgreich ✅")
                    st.metric("URL", result.get("url", "N/A"))
                with col2:
                    markdown_len = len(result.get("markdown", ""))
                    st.metric("Markdown Länge", f"{markdown_len:,} Zeichen")
                    links_count = len(result.get("links", {}).get("internal", []))
                    st.metric("Interne Links", links_count)
            else:
                st.error(f"Crawling fehlgeschlagen: {result.get('error')}")
        
        elif crawl_type in ["Batch URLs", "Recursive", "Sitemap"]:
            results_list = result.get("results", [])
            if results_list:
                successful = len([r for r in results_list if r.get("success", False)])
                total = len(results_list)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Erfolgreich", f"{successful}/{total}")
                with col2:
                    total_chars = sum(len(r.get("markdown", "")) for r in results_list if r.get("success"))
                    st.metric("Gesamt Zeichen", f"{total_chars:,}")
        
        # Detaillierte Ergebnisse in Expander
        with st.expander("🔍 Detaillierte Ergebnisse anzeigen"):
            st.json(result)
        
        # Clear Button
        if st.button("🗑️ Ergebnisse löschen"):
            st.session_state.crawl_result = None
            st.rerun()
    
    # Fehler anzeigen
    if st.session_state.crawl_error:
        st.error(f"❌ Crawling fehlgeschlagen: {st.session_state.crawl_error}")
        
        if st.button("🗑️ Fehler löschen"):
            st.session_state.crawl_error = None
            st.rerun()
    
    # Sidebar mit Informationen
    with st.sidebar:
        st.header("ℹ️ System Info")
        
        st.subheader("🔧 Services")
        st.write("✅ Google Cloud" if gcp_ok else "❌ Google Cloud")
        st.write("✅ ChromaDB" if chroma_ok else "❌ ChromaDB")
        st.write("✅ Modal.com" if crawler_ok else "❌ Modal.com")
        
        st.subheader("📊 ChromaDB Status")
        if chroma_client:
            try:
                # Zeige verfügbare Collections (falls vorhanden)
                collections = chroma_client.list_collections()
                st.write(f"Collections: {len(collections)}")
                for collection in collections:
                    st.write(f"- {collection.name}")
            except Exception as e:
                st.write(f"Fehler: {e}")
        
        st.subheader("🌐 Modal.com Status")
        if crawler_client:
            st.write("Service: Verbunden ✅")
            st.write("Endpunkte:")
            st.write("- Single URL ✅")
            st.write("- Batch URLs ✅")
            st.write("- Recursive ✅")
            st.write("- Sitemap ✅")

if __name__ == "__main__":
    main()