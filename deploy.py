#!/usr/bin/env python3
"""
Deployment-Script für RAG Agent App
Wechselt zwischen Benutzer- und Entwickler-Modus
"""

import shutil
import sys
import os

def deploy_user_mode():
    """Deployed die benutzerfreundliche Version."""
    print("🚀 Deploying User-Friendly Version...")
    
    # Backup der aktuellen App
    if os.path.exists("streamlit_app.py"):
        shutil.copy("streamlit_app.py", "streamlit_app_backup.py")
        print("📦 Backup der aktuellen App erstellt")
    
    # Kopiere die benutzerfreundliche Version
    shutil.copy("streamlit_app_user.py", "streamlit_app.py")
    print("✅ Benutzerfreundliche Version deployed")
    
    # Update config
    update_config("user")
    
    print("🎉 User-Mode erfolgreich aktiviert!")
    print("💡 Die App ist jetzt optimiert für Endbenutzer")

def deploy_developer_mode():
    """Deployed die Entwickler-Version."""
    print("🔧 Deploying Developer Version...")
    
    # Backup der aktuellen App
    if os.path.exists("streamlit_app.py"):
        shutil.copy("streamlit_app.py", "streamlit_app_backup.py")
        print("📦 Backup der aktuellen App erstellt")
    
    # Kopiere die adaptive Version (mit Developer-Config)
    shutil.copy("streamlit_app_adaptive.py", "streamlit_app.py")
    print("✅ Entwickler-Version deployed")
    
    # Update config
    update_config("developer")
    
    print("🎉 Developer-Mode erfolgreich aktiviert!")
    print("💡 Die App zeigt jetzt alle Debug-Informationen und erweiterte Funktionen")

def deploy_adaptive_mode():
    """Deployed die adaptive Version."""
    print("🔄 Deploying Adaptive Version...")
    
    # Backup der aktuellen App
    if os.path.exists("streamlit_app.py"):
        shutil.copy("streamlit_app.py", "streamlit_app_backup.py")
        print("📦 Backup der aktuellen App erstellt")
    
    # Kopiere die adaptive Version
    shutil.copy("streamlit_app_adaptive.py", "streamlit_app.py")
    print("✅ Adaptive Version deployed")
    
    print("🎉 Adaptive-Mode erfolgreich aktiviert!")
    print("💡 Du kannst jetzt in config.py zwischen Modi wechseln")

def update_config(mode):
    """Updated die Konfiguration."""
    config_content = f'''"""
Konfiguration für RAG Agent App
"""

# App-Modi
APP_MODE = "{mode}"  # "developer" oder "user"

# UI-Konfiguration
UI_CONFIG = {{
    "user": {{
        "show_crawler_test": False,
        "show_status_details": False,
        "show_debug_info": False,
        "show_advanced_options": True,
        "simplified_navigation": True,
        "custom_styling": True
    }},
    "developer": {{
        "show_crawler_test": True,
        "show_status_details": True,
        "show_debug_info": True,
        "show_advanced_options": True,
        "simplified_navigation": False,
        "custom_styling": False
    }}
}}

# Feature-Flags
FEATURES = {{
    "vertex_ai_embeddings": True,
    "gemini_chat": True,
    "memory_management": True,
    "auto_reduction": True,
    "debug_logging": APP_MODE == "developer"
}}

def get_config():
    """Gibt die aktuelle Konfiguration zurück."""
    return UI_CONFIG.get(APP_MODE, UI_CONFIG["user"])

def is_developer_mode():
    """Prüft ob Entwickler-Modus aktiv ist."""
    return APP_MODE == "developer"

def is_user_mode():
    """Prüft ob Benutzer-Modus aktiv ist."""
    return APP_MODE == "user"'''
    
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print(f"⚙️ Konfiguration auf '{mode}' gesetzt")

def restore_backup():
    """Stellt die Backup-Version wieder her."""
    if os.path.exists("streamlit_app_backup.py"):
        shutil.copy("streamlit_app_backup.py", "streamlit_app.py")
        print("🔄 Backup wiederhergestellt")
    else:
        print("❌ Kein Backup gefunden")

def show_help():
    """Zeigt die Hilfe an."""
    print("""
🤖 RAG Agent Deployment Script

Verwendung:
    python deploy.py [MODE]

Modi:
    user        - Benutzerfreundliche Version (empfohlen für Produktion)
    developer   - Entwickler-Version mit Debug-Informationen
    adaptive    - Adaptive Version (wechselbar über config.py)
    restore     - Stellt Backup wieder her
    
Beispiele:
    python deploy.py user       # Deployed User-Version
    python deploy.py developer  # Deployed Developer-Version
    python deploy.py restore    # Stellt Backup wieder her
    """)

def main():
    if len(sys.argv) != 2:
        show_help()
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "user":
        deploy_user_mode()
    elif mode == "developer":
        deploy_developer_mode()
    elif mode == "adaptive":
        deploy_adaptive_mode()
    elif mode == "restore":
        restore_backup()
    elif mode in ["help", "-h", "--help"]:
        show_help()
    else:
        print(f"❌ Unbekannter Modus: {mode}")
        show_help()

if __name__ == "__main__":
    main()