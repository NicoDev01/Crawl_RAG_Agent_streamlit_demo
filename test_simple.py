"""
Simple test for URL Validator integration

Run this with: streamlit run test_simple.py
"""

import streamlit as st
from ux_components import URLValidator

# Page config
st.set_page_config(
    page_title="Simple URL Validator Test",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 Simple URL Validator Test")
st.markdown("Test der URL-Validierung Integration")
st.markdown("---")

# Initialize URL validator
if 'url_validator' not in st.session_state:
    st.session_state.url_validator = URLValidator(timeout=10, debounce_delay=0.5)

url_validator = st.session_state.url_validator

# Simple form test
with st.form("simple_test"):
    st.subheader("🌐 URL Test")
    
    # URL input with validation
    url = st.text_input(
        "Website URL:",
        placeholder="https://docs.example.com",
        help="Gib eine URL ein zum Testen"
    )
    
    # Show validation if URL is entered
    if url and url.strip():
        st.markdown("**Validierungsergebnis:**")
        validation_result = url_validator.render_validation_feedback(
            url, 
            show_reachability=True, 
            debounced=True
        )
        
        # Show status indicator
        status_indicator = url_validator.get_validation_status_indicator(validation_result)
        st.markdown(f"**Status:** {status_indicator}")
    
    # Submit button
    submitted = st.form_submit_button("✅ Test Validierung")
    
    if submitted:
        if not url:
            st.error("❌ Bitte gib eine URL ein")
        else:
            st.success("✅ Formular erfolgreich übermittelt!")
            st.json({
                "url": url,
                "timestamp": datetime.now().isoformat()
            })

# Test URLs section (outside form to avoid session state issues)
st.markdown("---")
st.markdown("### 🧪 Test-URLs zum Kopieren")

test_urls = [
    "https://docs.python.org",
    "https://streamlit.io",
    "https://github.com",
    "https://nonexistent-domain-12345.com",
    "invalid-url",
    "https://httpstat.us/404"
]

for i, test_url in enumerate(test_urls):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.code(test_url)
    with col2:
        if st.button("📋 Kopieren", key=f"copy_{i}"):
            st.success("✅ Kopiert!")
            st.info("👆 Kopiere diese URL in das Eingabefeld oben")

# Manual validator test
st.markdown("---")
st.markdown("### 🔧 Direkter Validator Test")

manual_url = st.text_input("URL für direkten Test:", key="manual_url")

if manual_url:
    with st.spinner("Validiere..."):
        result = url_validator.validate_with_debounce(manual_url, force=True)
        
        if result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Ergebnis:**")
                if result.is_valid:
                    st.success("✅ Gültig")
                else:
                    st.error("❌ Ungültig")
                
                if result.error_message:
                    st.error(f"Fehler: {result.error_message}")
                
                if result.warning_message:
                    st.warning(f"Warnung: {result.warning_message}")
            
            with col2:
                st.markdown("**Details:**")
                st.json({
                    "status_code": result.status_code,
                    "response_time": f"{result.response_time:.3f}s" if result.response_time else None,
                    "is_valid": result.is_valid
                })

# Performance info
st.markdown("---")
st.markdown("### ⚡ Features")
st.success("""
✅ **Real-time Validierung** - Sofortiges Feedback beim Tippen  
✅ **Debouncing** - 500ms Verzögerung für bessere Performance  
✅ **Caching** - Wiederholte URLs werden aus Cache geladen  
✅ **Smart HTTP** - HEAD-Request zuerst, dann GET als Fallback  
✅ **Detaillierte Fehler** - Hilfreiche Tipps bei Problemen  
✅ **Visual Feedback** - Farbige Indikatoren für Status  
""")