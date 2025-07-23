"""
Test script for URL Validator integration in main app

Run this with: streamlit run test_integration.py
"""

import streamlit as st
from ux_components import URLValidator
from state_manager import get_state_manager

# Page config
st.set_page_config(
    page_title="URL Validator Integration Test",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 URL Validator Integration Test")
st.markdown("Test der URL-Validierung wie sie in der Hauptapp integriert ist")
st.markdown("---")

# Simulate the main app form structure
with st.form("test_form"):
    st.subheader("🌐 Website-Konfiguration")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Initialize URL validator (same as in main app)
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
        
        # URL Status Indicator
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
            else:
                st.info("🔄 URL wird validiert...")
        else:
            st.info("⏳ Warte auf URL-Eingabe...")
    
    # Form submission with enhanced validation
    submitted = st.form_submit_button("🚀 Test Formular-Validierung", use_container_width=True)
    
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
                st.info("Das Crawling würde trotzdem fortgesetzt...")
            
            # Success message
            st.success("✅ **Formular-Validierung erfolgreich!**")
            st.balloons()
            
            # Show what would be processed
            st.markdown("### 📋 Verarbeitungsdetails:")
            st.json({
                "url": url,
                "name": name,
                "source_type": source_type,
                "url_valid": url_validation.is_valid if url_validation else False,
                "response_time": f"{url_validation.response_time:.3f}s" if url_validation and url_validation.response_time else None,
                "status_code": url_validation.status_code if url_validation else None
            })

# Debug info
st.markdown("---")
st.markdown("### 🔧 Debug-Informationen")

with st.expander("Session State", expanded=False):
    relevant_keys = [k for k in st.session_state.keys() if any(term in k.lower() for term in ['url', 'validation'])]
    if relevant_keys:
        for key in relevant_keys:
            if key == 'url_validation_result':
                result = st.session_state[key]
                if result:
                    st.write(f"**{key}:**")
                    st.json({
                        "is_valid": result.is_valid,
                        "error_message": result.error_message,
                        "warning_message": result.warning_message,
                        "status_code": result.status_code,
                        "response_time": result.response_time
                    })
                else:
                    st.write(f"**{key}:** None")
            else:
                st.write(f"**{key}:** {st.session_state[key]}")
    else:
        st.write("Keine relevanten Session State Keys gefunden")

# Performance info
st.markdown("### ⚡ Performance-Features")
st.info("""
**Implementierte Optimierungen:**

✅ **Debounced Validation** - 500ms Verzögerung verhindert zu häufige Requests  
✅ **Caching** - Validierungsergebnisse werden 5 Minuten gecacht  
✅ **Smart HTTP Requests** - HEAD-Request zuerst, dann GET als Fallback  
✅ **Detaillierte Fehlermeldungen** - Hilfreiche Tipps für häufige Probleme  
✅ **Visual Feedback** - Farbige Indikatoren (🟢🟡🔴) für sofortiges Feedback  
✅ **Form Validation** - Verhindert Submission bei ungültigen URLs  
""")

# Test URLs
st.markdown("### 🧪 Test-URLs")
test_urls = [
    "https://docs.python.org",
    "https://streamlit.io/docs",
    "https://github.com/streamlit/streamlit",
    "https://nonexistent-domain-12345.com",
    "invalid-url-without-protocol",
    "https://httpstat.us/404"
]

st.markdown("Klicke auf eine URL zum Testen:")
cols = st.columns(2)
for i, test_url in enumerate(test_urls):
    with cols[i % 2]:
        if st.button(f"Test: {test_url[:40]}...", key=f"test_url_{i}"):
            # Use a different approach to set the URL
            st.session_state["test_url_selected"] = test_url
            st.info(f"📋 Test-URL kopiert: `{test_url}`")
            st.info("👆 Kopiere diese URL in das Eingabefeld oben")