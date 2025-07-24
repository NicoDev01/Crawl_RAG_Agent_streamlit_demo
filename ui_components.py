"""
UI Components for enhanced chat experience.
"""

import streamlit as st
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ProgressStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ProgressStep:
    name: str
    status: ProgressStatus
    message: str
    duration: Optional[float] = None
    icon: str = "⏳"

class ChatProgressIndicator:
    """Real-time progress indicator for RAG processing."""
    
    def __init__(self, container):
        self.container = container
        self.steps = []
        self.start_time = time.time()
        
    def add_step(self, name: str, message: str, icon: str = "⏳"):
        """Add a new step to track."""
        step = ProgressStep(
            name=name,
            status=ProgressStatus.PENDING,
            message=message,
            icon=icon
        )
        self.steps.append(step)
        self._render()
        
    def update_step(self, name: str, status: ProgressStatus, message: str = None):
        """Update the status of a step."""
        for step in self.steps:
            if step.name == name:
                step.status = status
                if message:
                    step.message = message
                if status == ProgressStatus.COMPLETED:
                    step.duration = time.time() - self.start_time
                break
        self._render()
        
    def _render(self):
        """Render the progress indicator."""
        if not self.steps:
            return
            
        progress_html = """
        <div class="progress-container">
            <div style="font-weight: 600; color: #667eea; margin-bottom: 12px; font-size: 16px;">
                🤖 KI-Verarbeitung läuft...
            </div>
        """
        
        for step in self.steps:
            status_icon = self._get_status_icon(step.status)
            status_color = self._get_status_color(step.status)
            
            if step.status == ProgressStatus.RUNNING:
                class_name = "active"
            elif step.status == ProgressStatus.COMPLETED:
                class_name = "completed"
            elif step.status == ProgressStatus.ERROR:
                class_name = "error"
            else:
                class_name = "pending"
                
            duration_text = ""
            if step.duration:
                duration_text = f" ({step.duration:.1f}s)"
                
            progress_html += f"""
            <div class="progress-step {class_name}" style="color: {status_color};">
                <span class="progress-step-icon">{status_icon}</span>
                <span>{step.message}{duration_text}</span>
            </div>
            """
        
        progress_html += "</div>"
        self.container.markdown(progress_html, unsafe_allow_html=True)
    
    def _get_status_icon(self, status: ProgressStatus) -> str:
        """Get icon for status."""
        icons = {
            ProgressStatus.PENDING: "⏳",
            ProgressStatus.RUNNING: "🔄",
            ProgressStatus.COMPLETED: "✅",
            ProgressStatus.ERROR: "❌"
        }
        return icons.get(status, "⏳")
    
    def _get_status_color(self, status: ProgressStatus) -> str:
        """Get color for status."""
        colors = {
            ProgressStatus.PENDING: "#95a5a6",
            ProgressStatus.RUNNING: "#3498db", 
            ProgressStatus.COMPLETED: "#27ae60",
            ProgressStatus.ERROR: "#e74c3c"
        }
        return colors.get(status, "#95a5a6")

class StreamingResponseHandler:
    """Handle streaming text responses."""
    
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.full_text = ""
        
    async def stream_text(self, text: str, chunk_size: int = 10, delay_ms: int = 50):
        """Stream text word by word."""
        words = text.split()
        current_text = ""
        
        for i, word in enumerate(words):
            current_text += word + " "
            
            # Show text with cursor
            display_text = current_text + "▌"
            self.placeholder.markdown(display_text)
            
            # Add delay for streaming effect
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
        
        # Final text without cursor
        self.placeholder.markdown(current_text.strip())
        self.full_text = current_text.strip()
        
        return self.full_text

def create_progress_steps() -> List[Dict]:
    """Define the standard RAG processing steps."""
    return [
        {
            "name": "query_variations",
            "message": "🔄 Generiere Frage-Varianten...",
            "icon": "🔄"
        },
        {
            "name": "hyde_generation", 
            "message": "🧠 Erstelle hypothetische Antworten (HyDE)...",
            "icon": "🧠"
        },
        {
            "name": "database_search",
            "message": "🔍 Durchsuche Wissensdatenbank...",
            "icon": "🔍"
        },
        {
            "name": "reranking",
            "message": "⚡ Sortiere nach Relevanz (Vertex AI)...",
            "icon": "⚡"
        },
        {
            "name": "response_generation",
            "message": "✍️ Formuliere Antwort (Gemini 2.5)...",
            "icon": "✍️"
        }
    ]