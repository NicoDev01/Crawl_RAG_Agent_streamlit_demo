"""
Test für das intelligente URL-Detection System

Run this with: streamlit run test_intelligent_ui.py
"""

import streamlit as st
from ux_components import URLValidator
from url_detection import detect_url_type, URLDetector

# Page config
st.set_page_config(
    page_title="🤖 CraCha - Intelligent URL Detection Test",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
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

# Header
st.markdown("""
<div style="text-align: center; padding: 1rem 0; margin-bottom: 1.5rem;">
    <h2 style="color: #667eea; margin-bottom: 0.5rem;">🤖 CraCha - Intelligent URL Detection</h2>
    <p style="color: #666; font-size: 0.9rem; margin: 0;">Automatische Erkennung des optimalen Crawling-Typs</p>
</div>
""", unsafe_allow_html=True)

# Test verschiedene URL-Typen
st.markdown("### 🧪 Test verschiedene URL-Typen")

test_urls = [
    ("https://docs.python.org", "Dokumentations-Website"),
    ("https://example.com/sitemap.xml", "Sitemap XML"),
    ("https://example.com/page.html", "Einzelne HTML-Seite"),
    ("https://help.streamlit.io", "Hilfe-Website"),
    ("https://github.com/streamlit/streamlit", "Standard Website"),
    ("https://docs.streamlit.io/sitemap_index.xml", "Sitemap Index"),
    ("https://example.com/document.pdf", "PDF-Dokument")
]

detector = URLDetector()

for url, description in test_urls:
    with st.expander(f"🔍 {description}: {url}"):
        method = detector.detect_crawling_method(url)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"**{method.icon} Erkannter Typ:**")
            st.info(method.method.title())
            
            st.markdown("**⚙️ Einstellungen:**")
            for key, value in method.settings.items():
                if key != "recommended_reason":
                    st.caption(f"• {key}: {value}")
        
        with col2:
            st.markdown("**📝 Beschreibung:**")
            st.write(method.description)
            
            if "recommended_reason" in method.settings:
                st.markdown("**💡 Grund:**")
                st.write(method.settings["recommended_reason"])
            
            st.markdown("**🔧 Backend-Methode:**")
            st.code(method.backend_method)

# Hauptformular mit intelligenter Erkennung
st.markdown("---")
st.markdown("### 🚀 Intelligentes Crawling-Interface")

with st.form("intelligent_crawling"):
    st.subheader("🌐 Website-Konfiguration")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # URL Validator
        if 'url_validator' not in st.session_state:
            st.session_state.url_validator = URLValidator(timeout=10, debounce_delay=0.5)
        
        url_validator = st.session_state.url_validator
        
        url = st.text_input(
            "Website URL: *",
            placeholder="https://docs.example.com oder https://example.com/sitemap.xml",
            help="Vollständige URL - der Typ wird automatisch erkannt",
            key="main_url_input"
        )
        
        # Real-time URL validation
        if url and url.strip():
            validation_result = url_validator.render_validation_feedback(
                url, 
                show_reachability=True, 
                debounced=True
            )
            st.session_state.url_validation_result = validation_result
        else:
            st.session_state.url_validation_result = None
        
        name = st.text_input(
            "Name der Wissensdatenbank: *",
            placeholder="z.B. 'Produktdokumentation' oder 'Firmen-Wiki'",
            help="Eindeutiger Name zur Identifikation"
        )
    
    with col2:
        # Intelligente URL-Typ-Erkennung
        if url and url.strip():
            url_validation = getattr(st.session_state, 'url_validation_result', None)
            if url_validation and url_validation.is_valid:
                # Intelligente Typ-Erkennung
                detected_method = detect_url_type(url)
                st.session_state.detected_crawling_method = detected_method
                
                # Zeige erkannten Typ
                st.info(f"✨ {detected_method.icon} **{detected_method.description}**")
                
                # Zeige Grund für die Erkennung
                if "recommended_reason" in detected_method.settings:
                    st.caption(f"💡 {detected_method.settings['recommended_reason']}")
            else:
                st.session_state.detected_crawling_method = None
        else:
            st.session_state.detected_crawling_method = None
    
    # Intelligente Crawling-Einstellungen
    st.subheader("⚙️ Crawling-Einstellungen")
    
    detected_method = getattr(st.session_state, 'detected_crawling_method', None)
    
    if detected_method:
        col3, col4 = st.columns(2)
        
        with col3:
            if detected_method.method in ["website", "documentation"]:
                default_depth = detected_method.settings.get("max_depth", 2)
                max_depth = st.slider(
                    "Crawling-Tiefe:",
                    min_value=1, max_value=4, value=default_depth,
                    help="Wie tief sollen Links verfolgt werden?"
                )
                
                if detected_method.method == "documentation":
                    st.caption("📚 Dokumentations-Websites profitieren von tieferem Crawling")
                else:
                    st.caption("🌐 Standard Website-Crawling")
                    
            elif detected_method.method == "sitemap":
                max_depth = 1
                st.info("🗺️ Sitemap-Crawling: Tiefe automatisch auf 1 gesetzt")
                
            else:  # single
                max_depth = 1
                st.info("📄 Einzelseite: Keine Tiefe erforderlich")
        
        with col4:
            if detected_method.method == "sitemap":
                max_pages = None
                st.metric("Seiten-Limit", "Automatisch", help="Alle URLs aus der Sitemap")
                
            elif detected_method.method == "single":
                max_pages = 1
                st.metric("Seiten-Anzahl", "1", help="Nur die angegebene Seite")
                
            else:
                default_limit = detected_method.settings.get("limit", 20)
                max_pages = st.number_input(
                    "Maximale Seitenzahl:",
                    min_value=1, max_value=100, 
                    value=default_limit,
                    help="Maximale Anzahl zu crawlender Seiten"
                )
                
                if detected_method.method == "documentation":
                    st.caption("📚 Dokumentations-Sites: Höhere Limits empfohlen")
                else:
                    st.caption("🌐 Standard-Limits für Website-Crawling")
        
        # Warnung bei hohen Werten
        if detected_method.method in ["website", "documentation"] and (max_depth > 3 or (max_pages and max_pages > 50)):
            st.warning("⚠️ Hohe Werte können zu langen Ladezeiten führen!")
        
        # Intelligente Zeitschätzung
        if detected_method.method == "sitemap":
            st.info("⏱️ Geschätzte Dauer: Abhängig von der Anzahl der URLs in der Sitemap")
        elif detected_method.method == "single":
            st.info("⏱️ Geschätzte Dauer: ~5-10 Sekunden für eine Seite")
        elif detected_method.method in ["website", "documentation"] and max_pages:
            if max_depth > 1:
                estimated_pages = min(max_pages, 10 ** (max_depth - 1) * 3)
                estimated_time = estimated_pages * 2
                st.info(f"⏱️ Geschätzte Dauer: ~{estimated_time} Sekunden für ca. {estimated_pages} Seiten")
            else:
                estimated_time = max_pages * 2
                st.info(f"⏱️ Geschätzte Dauer: ~{estimated_time} Sekunden für {max_pages} Seite(n)")
                
    else:
        st.info("💡 Gib eine URL ein, um optimale Crawling-Einstellungen zu erhalten")
        max_depth = 2
        max_pages = 10
    
    # Erweiterte Einstellungen
    with st.expander("🔧 Erweiterte Einstellungen"):
        col5, col6 = st.columns(2)
        
        with col5:
            chunk_size = st.slider(
                "Text-Chunk-Größe:",
                min_value=500, max_value=2500, value=1200,
                help="Kleinere Chunks = präzisere Antworten, Größere = mehr Kontext"
            )
            
            if chunk_size < 800:
                st.info("📝 Kleine Chunks: Sehr präzise, aber wenig Kontext")
            elif chunk_size > 1800:
                st.info("📚 Große Chunks: Viel Kontext, aber weniger präzise")
            else:
                st.success("✅ Optimale Chunk-Größe")
        
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
    
    # Submit Button
    submitted = st.form_submit_button("🚀 Wissensdatenbank erstellen", use_container_width=True)
    
    if submitted:
        validation_errors = []
        
        if not url or not url.strip():
            validation_errors.append("Website URL ist erforderlich")
        
        if not name or not name.strip():
            validation_errors.append("Name der Wissensdatenbank ist erforderlich")
        
        url_validation = getattr(st.session_state, 'url_validation_result', None)
        if url and url_validation and not url_validation.is_valid:
            validation_errors.append(f"URL ist ungültig: {url_validation.error_message}")
        
        if validation_errors:
            st.error("❌ **Bitte korrigiere folgende Fehler:**")
            for error in validation_errors:
                st.error(f"• {error}")
        else:
            st.success("✅ **Intelligente Validierung erfolgreich!**")
            st.balloons()
            
            # Zeige erkannte Konfiguration
            st.markdown("### 🎯 Erkannte Konfiguration:")
            if detected_method:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.json({
                        "url": url,
                        "name": name,
                        "detected_type": detected_method.method,
                        "backend_method": detected_method.backend_method,
                        "max_depth": max_depth,
                        "max_pages": max_pages
                    })
                
                with col_b:
                    st.markdown("**🔧 Optimale Einstellungen:**")
                    for key, value in detected_method.settings.items():
                        st.write(f"• **{key}:** {value}")

# Features Info
st.markdown("---")
st.markdown("### ✨ Intelligente Features")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    **🤖 Automatische Erkennung:**
    
    - Sitemap-URLs werden automatisch erkannt
    - Dokumentations-Sites erhalten tiefere Einstellungen
    - Einzelseiten werden direkt verarbeitet
    - Website-Typ bestimmt optimale Parameter
    - Keine manuelle Typ-Auswahl nötig
    """)

with col2:
    st.info("""
    **🎯 Intelligente Optimierung:**
    
    - Crawling-Tiefe basierend auf Website-Typ
    - Seitenlimits automatisch angepasst
    - Zeitschätzung je nach erkanntem Typ
    - Backend-Methode automatisch gewählt
    - Benutzerfreundliche Erklärungen
    """)

st.markdown("**🚀 Das System erkennt automatisch den besten Crawling-Ansatz und optimiert alle Einstellungen!**")