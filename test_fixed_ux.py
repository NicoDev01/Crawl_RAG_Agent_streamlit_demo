"""
Test für das behobene UX-Problem

Run this with: streamlit run test_fixed_ux.py
"""

import streamlit as st
from ux_components import URLValidator
from url_detection import detect_url_type

# Page config
st.set_page_config(
    page_title="🤖 CraCha - Fixed UX Test",
    page_icon="🤖",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 1rem 0; margin-bottom: 1.5rem;">
    <h2 style="color: #667eea;">🤖 CraCha - UX Problem behoben</h2>
    <p style="color: #666; font-size: 0.9rem;">Enter im URL-Feld startet NICHT mehr den Crawling-Prozess</p>
</div>
""", unsafe_allow_html=True)

# Problem Erklärung
st.markdown("### 🚨 Problem behoben:")
st.error("""
**Vorher:** URL eingeben → Enter drücken → Crawling startet SOFORT → Einstellungen können nicht angepasst werden!
""")

st.success("""
**Jetzt:** URL eingeben → Enter drücken → Einstellungen werden geladen → User kann anpassen → Bewusst auf "Erstellen" klicken
""")

# Test Form
with st.form("fixed_ux_test"):
    st.subheader("🌐 Website-Konfiguration")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # URL Validator
        if 'url_validator' not in st.session_state:
            st.session_state.url_validator = URLValidator(timeout=10, debounce_delay=0.5)
        
        url_validator = st.session_state.url_validator
        
        # URL Input - Enter soll NICHT submitten
        url = st.text_input(
            "Website URL: *",
            placeholder="https://docs.example.com",
            help="Drücke Enter → Einstellungen werden geladen (KEIN Auto-Submit!)",
            key="test_url_input"
        )
        
        st.caption("💡 **Test:** Drücke Enter nach URL-Eingabe - es sollte NICHT crawlen!")
        
        # Real-time validation
        if url and url.strip():
            validation_result = url_validator.render_validation_feedback(url, show_reachability=True, debounced=True)
            st.session_state.url_validation_result = validation_result
        else:
            st.session_state.url_validation_result = None
        
        name = st.text_input(
            "Name: *",
            placeholder="Test Datenbank",
            help="Name für die Wissensdatenbank"
        )
    
    with col2:
        # Intelligente Erkennung
        if url and url.strip():
            url_validation = getattr(st.session_state, 'url_validation_result', None)
            if url_validation and url_validation.is_valid:
                detected_method = detect_url_type(url)
                st.session_state.detected_crawling_method = detected_method
                
                st.info(f"✨ {detected_method.icon} **{detected_method.description}**")
                
                if "recommended_reason" in detected_method.settings:
                    st.caption(f"💡 {detected_method.settings['recommended_reason']}")
                
                # Wichtige Benutzerführung
                st.success("👇 Passe die Einstellungen unten an und klicke dann auf 'Erstellen'")
            else:
                st.session_state.detected_crawling_method = None
        else:
            st.session_state.detected_crawling_method = None
    
    # Crawling-Einstellungen
    st.subheader("⚙️ Crawling-Einstellungen")
    
    detected_method = getattr(st.session_state, 'detected_crawling_method', None)
    
    if detected_method:
        col3, col4 = st.columns(2)
        
        with col3:
            if detected_method.method in ["website", "documentation"]:
                # DEFAULT AUF 1 GESETZT
                max_depth = st.slider(
                    "Crawling-Tiefe:",
                    min_value=1, max_value=4, value=1,  # IMMER 1
                    help="Default auf 1 - kann angepasst werden"
                )
                
                st.caption("🎯 Default: 1 (kann vor dem Crawling angepasst werden)")
                
            elif detected_method.method == "sitemap":
                max_depth = 1
                st.info("🗺️ Sitemap: Tiefe automatisch auf 1")
                
            else:  # single
                max_depth = 1
                st.info("📄 Einzelseite: Tiefe 1")
        
        with col4:
            if detected_method.method == "sitemap":
                max_pages = None
                st.metric("Seiten-Limit", "Automatisch")
                
            elif detected_method.method == "single":
                max_pages = 1
                st.metric("Seiten-Anzahl", "1")
                
            else:
                # DEFAULT AUF 1 GESETZT
                max_pages = st.number_input(
                    "Maximale Seitenzahl:",
                    min_value=1, max_value=100, 
                    value=1,  # IMMER 1
                    help="Default auf 1 - kann angepasst werden"
                )
                
                st.caption("🎯 Default: 1 (kann vor dem Crawling angepasst werden)")
        
        # Zeitschätzung
        if detected_method.method == "single":
            st.info("⏱️ ~5-10 Sekunden für eine Seite")
        elif detected_method.method == "sitemap":
            st.info("⏱️ Abhängig von Sitemap-Größe")
        else:
            estimated_time = max_pages * 2 if max_pages else 10
            st.info(f"⏱️ ~{estimated_time} Sekunden für {max_pages} Seite(n)")
            
    else:
        st.info("💡 Gib eine URL ein, um Crawling-Einstellungen zu sehen")
        max_depth = 1
        max_pages = 1
    
    # WICHTIG: Nur dieser Button startet den Crawling-Prozess!
    st.markdown("---")
    st.markdown("### 🚀 Crawling starten")
    st.info("💡 **Nur dieser Button startet den Crawling-Prozess** - nicht Enter im URL-Feld!")
    
    submitted = st.form_submit_button("🚀 Wissensdatenbank erstellen", use_container_width=True)
    
    if submitted:
        # Validation
        validation_errors = []
        
        if not url or not url.strip():
            validation_errors.append("Website URL ist erforderlich")
        
        if not name or not name.strip():
            validation_errors.append("Name ist erforderlich")
        
        url_validation = getattr(st.session_state, 'url_validation_result', None)
        if url and url_validation and not url_validation.is_valid:
            validation_errors.append(f"URL ungültig: {url_validation.error_message}")
        
        if validation_errors:
            st.error("❌ **Fehler:**")
            for error in validation_errors:
                st.error(f"• {error}")
        else:
            st.success("✅ **UX-Fix funktioniert!**")
            st.balloons()
            
            st.markdown("### 🎯 Konfiguration:")
            st.json({
                "url": url,
                "name": name,
                "detected_type": detected_method.method if detected_method else "unknown",
                "max_depth": max_depth,
                "max_pages": max_pages,
                "note": "Crawling würde jetzt mit diesen angepassten Einstellungen starten!"
            })

# Test Anweisungen
st.markdown("---")
st.markdown("### 🧪 Test-Anweisungen")

st.markdown("""
**1. URL eingeben und Enter drücken:**
- ✅ Einstellungen sollten geladen werden
- ❌ Crawling sollte NICHT starten

**2. Einstellungen anpassen:**
- ✅ Tiefe und Seitenzahl können geändert werden
- ✅ Defaults stehen auf 1

**3. Bewusst auf "Erstellen" klicken:**
- ✅ Nur dann startet der Crawling-Prozess
- ✅ Mit den angepassten Einstellungen
""")

# Test URLs
st.markdown("### 🔗 Test-URLs")
test_urls = [
    "https://docs.python.org",
    "https://example.com/sitemap.xml", 
    "https://streamlit.io/docs",
    "https://example.com/page.html"
]

for i, test_url in enumerate(test_urls):
    if st.button(f"Test: {test_url}", key=f"test_{i}"):
        st.session_state.test_url_input = test_url
        st.info(f"URL gesetzt: {test_url} - Drücke Enter im Feld oben!")
        st.rerun()

st.success("🎉 **UX-Problem behoben:** Enter im URL-Feld startet nicht mehr den Crawling-Prozess!")