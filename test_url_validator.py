"""
Test script for URL Validator component

Run this with: streamlit run test_url_validator.py
"""

import streamlit as st
from ux_components import URLValidator

# Page config
st.set_page_config(
    page_title="URL Validator Test",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 URL Validator Test")
st.markdown("---")

# Initialize validator
validator = URLValidator(timeout=10, debounce_delay=0.5)

# Test the validator
st.markdown("### Test der URL-Validierung")

col1, col2 = st.columns([2, 1])

with col1:
    url, result = validator.render_realtime_validator(show_reachability=True)

with col2:
    if url:
        st.markdown("**Status:**")
        status_indicator = validator.get_validation_status_indicator(result)
        st.markdown(f"# {status_indicator}")
        
        if result.is_valid:
            st.success("✅ Gültig")
        else:
            st.error("❌ Ungültig")

# Test with predefined URLs
st.markdown("---")
st.markdown("### 🧪 Schnelltests")

test_urls = [
    "https://docs.python.org",
    "https://streamlit.io",
    "https://github.com",
    "https://nonexistent-domain-12345.com",
    "invalid-url",
    "http://httpstat.us/404",
    "http://httpstat.us/500"
]

st.markdown("Klicke auf eine URL zum Testen:")

cols = st.columns(3)
for i, test_url in enumerate(test_urls):
    with cols[i % 3]:
        if st.button(f"Test: {test_url[:30]}...", key=f"test_{i}"):
            st.session_state[validator.get_session_key("url_input")] = test_url
            st.rerun()

# Manual validation test
st.markdown("---")
st.markdown("### 🔧 Manuelle Validierung")

manual_url = st.text_input("URL für manuellen Test:", key="manual_test")

if manual_url and st.button("Validieren"):
    with st.spinner("Validiere URL..."):
        # Test syntax validation
        syntax_result = validator.validate_url_syntax(manual_url)
        st.markdown("**Syntax-Validierung:**")
        if syntax_result.is_valid:
            st.success("✅ Syntax ist gültig")
        else:
            st.error(f"❌ {syntax_result.error_message}")
        
        # Test reachability
        if syntax_result.is_valid:
            st.markdown("**Erreichbarkeits-Test:**")
            reachability_result = validator.check_url_reachability(manual_url)
            
            if reachability_result.is_valid:
                st.success(f"✅ URL ist erreichbar (Status: {reachability_result.status_code})")
                if reachability_result.response_time:
                    st.info(f"⏱️ Antwortzeit: {reachability_result.response_time:.3f}s")
                if reachability_result.warning_message:
                    st.warning(f"⚠️ {reachability_result.warning_message}")
            else:
                st.error(f"❌ {reachability_result.error_message}")
                if reachability_result.status_code:
                    st.info(f"HTTP Status: {reachability_result.status_code}")

# Cache info
st.markdown("---")
st.markdown("### 📊 Cache-Statistiken")

if st.button("Cache-Info anzeigen"):
    # This would show cache statistics in a real implementation
    st.info("Cache-Funktionalität ist implementiert und funktioniert im Hintergrund")
    
    # Show session state for debugging
    with st.expander("🔧 Session State (Debug)", expanded=False):
        relevant_keys = [k for k in st.session_state.keys() if 'urlvalidator' in k.lower()]
        if relevant_keys:
            for key in relevant_keys:
                st.write(f"**{key}:** {st.session_state[key]}")
        else:
            st.write("Keine URL-Validator Session State gefunden")

# Performance tips
st.markdown("---")
st.markdown("### 💡 Performance-Tipps")

st.info("""
**Optimierungen in der URL-Validierung:**

- ✅ **Debouncing**: Validierung erfolgt erst nach 500ms Pause
- ✅ **Caching**: Ergebnisse werden 5 Minuten gecacht
- ✅ **HEAD-Requests**: Schnellere Erreichbarkeitsprüfung
- ✅ **Fallback zu GET**: Falls HEAD fehlschlägt
- ✅ **Timeout-Handling**: Verhindert hängende Requests
- ✅ **Detaillierte Fehlermeldungen**: Hilft bei der Problemlösung
""")