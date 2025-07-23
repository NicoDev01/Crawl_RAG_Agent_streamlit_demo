"""
Konfiguration für RAG Agent App
"""

# App-Modi
APP_MODE = "user"  # "developer" oder "user"

# UI-Konfiguration
UI_CONFIG = {
    "user": {
        "show_crawler_test": False,
        "show_status_details": False,
        "show_debug_info": False,
        "show_advanced_options": True,
        "simplified_navigation": True,
        "custom_styling": True
    },
    "developer": {
        "show_crawler_test": True,
        "show_status_details": True,
        "show_debug_info": True,
        "show_advanced_options": True,
        "simplified_navigation": False,
        "custom_styling": False
    }
}

# Feature-Flags
FEATURES = {
    "vertex_ai_embeddings": True,
    "gemini_chat": True,
    "memory_management": True,
    "auto_reduction": True,
    "debug_logging": APP_MODE == "developer"
}

def get_config():
    """Gibt die aktuelle Konfiguration zurück."""
    return UI_CONFIG.get(APP_MODE, UI_CONFIG["user"])

def is_developer_mode():
    """Prüft ob Entwickler-Modus aktiv ist."""
    return APP_MODE == "developer"

def is_user_mode():
    """Prüft ob Benutzer-Modus aktiv ist."""
    return APP_MODE == "user"