"""
Test für das minimalistisches UI Design

Run this with: streamlit run test_minimalist_ui.py
"""

import streamlit as st
from ux_components import URLValidator

# Page config
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
</style>
""", unsafe_allow_html=True)

# Dezenter Header
st.markdown("""
<div style="text-align: center; padding: 1rem 0; margin-bottom: 1.5rem;">
    <h2 style="color: #667eea; margin-bottom: 0.5rem;">🤖 CraCha - Crawl Chat Agent</h2>
    <p style="color: #666; font-size: 0.9rem; margin: 0;">Intelligente Wissensdatenbanken aus Webseiten erstellen</p>
</div>
""", unsafe_allow_html=True)

# Hauptnavigation
tab1, tab2 = st.tabs(["📚 Wissensdatenbank erstellen", "💬 Chat mit deinen Daten"])

with tab1:
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
        
        # Nur für Sitemap relevante Info
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
        
        # Form submission with enhanced validation - HERVORGEHOBENER BUTTON
        submitted = st.form_submit_button("🚀 Wissensdatenbank erstellen", use_container_width=True)
        
        if submitted:
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
                
                # Simulate processing
                st.success("✅ **Formular-Validierung erfolgreich!**")
                st.balloons()
                
                # Show what would be processed
                st.markdown("### 📋 Verarbeitungsdetails:")
                st.json({
                    "url": url,
                    "name": name,
                    "source_type": source_type,
                    "max_pages": max_pages,
                    "max_depth": max_depth,
                    "chunk_size": chunk_size,
                    "auto_reduce": auto_reduce,
                    "max_concurrent": max_concurrent
                })

with tab2:
    st.markdown("""
    <div class="feature-card">
        <h3>💬 Chat Interface</h3>
        <p>Hier würde das Chat-Interface erscheinen, nachdem eine Wissensdatenbank erstellt wurde.</p>
    </div>
    """, unsafe_allow_html=True)

# Design-Verbesserungen Info
st.markdown("---")
st.markdown("### ✨ Design-Verbesserungen")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    **✅ Implementiert:**
    
    - Dezenter Header mit CraCha Branding
    - Entfernung der redundanten Feature-Card
    - Minimierte URL-Status-Anzeigen
    - Hervorgehobener Submit-Button (rot/orange)
    - Entfernung unnötiger Crawling-Infos
    - Sauberes, minimalistisches Layout
    """)

with col2:
    st.info("""
    **🎨 Button-Design:**
    
    - Standard-Buttons: Blau-violetter Gradient
    - Submit-Button: Rot-oranger Gradient mit Schatten
    - Hover-Effekte mit Transform-Animation
    - Größere Schrift und Padding für Wichtigkeit
    - Box-Shadow für visuellen Pop-Effekt
    """)

st.markdown("**🚀 Der Submit-Button ist jetzt visuell hervorgehoben und macht klar, dass hier der Prozess gestartet wird!**")