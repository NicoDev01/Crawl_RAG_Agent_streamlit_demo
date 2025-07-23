"""
State Manager for Streamlit RAG Knowledge Assistant

This module provides centralized session state management for the application,
organizing state into logical groups and providing clean interfaces for state operations.
"""

import streamlit as st
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FormValidationState(Enum):
    """Enumeration for form field validation states."""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    UNKNOWN = "unknown"


@dataclass
class FormFieldState:
    """State for individual form fields."""
    value: Any = None
    validation_state: FormValidationState = FormValidationState.UNKNOWN
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    last_validated: Optional[datetime] = None


@dataclass
class FormState:
    """State management for form inputs and validation."""
    # URL field
    url: FormFieldState = field(default_factory=FormFieldState)
    
    # Name field
    name: FormFieldState = field(default_factory=FormFieldState)
    
    # Crawling settings
    source_type: str = "Website Crawling"
    max_depth: int = 1
    max_pages: Optional[int] = 1
    chunk_size: int = 1200
    max_concurrent: int = 5
    auto_reduce: bool = True
    
    # Form submission state
    is_submitted: bool = False
    submission_timestamp: Optional[datetime] = None
    
    # Auto-suggestions
    suggested_names: List[str] = field(default_factory=list)
    auto_filled_name: bool = False
    
    # Website analysis results
    detected_website_type: Optional[str] = None
    recommended_settings: Optional[Dict[str, Any]] = None
    sitemap_discovered: Optional[str] = None


@dataclass
class ProgressState:
    """State management for crawling progress tracking."""
    # Progress tracking
    is_active: bool = False
    current_step: int = 0
    total_steps: int = 5
    current_message: str = ""
    current_url: str = ""
    
    # Process timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Statistics
    success_count: int = 0
    error_count: int = 0
    total_processed: int = 0
    processing_speed: float = 0.0
    
    # Log entries (stored as dicts for JSON serialization)
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Results
    final_results: Optional[Dict[str, Any]] = None
    completion_status: str = "not_started"  # not_started, running, completed, failed


@dataclass
class UIState:
    """State management for UI component visibility and interactions."""
    # Component visibility
    show_advanced_settings: bool = False
    show_help_section: bool = False
    show_validation_details: bool = True
    show_progress_log: bool = False
    
    # Active tab/section
    active_tab: str = "create"  # create, chat
    selected_collection: Optional[str] = None
    
    # Chat state
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Success flow state
    show_success_animation: bool = False
    success_animation_played: bool = False
    
    # Error handling
    last_error: Optional[str] = None
    error_timestamp: Optional[datetime] = None
    
    # User preferences
    preferred_chunk_size: int = 1200
    preferred_max_concurrent: int = 5
    remember_settings: bool = True


class StateManager:
    """Centralized state manager for the Streamlit application."""
    
    def __init__(self):
        self.form_state_key = "app_form_state"
        self.progress_state_key = "app_progress_state"
        self.ui_state_key = "app_ui_state"
        
        # Initialize states if they don't exist
        self._initialize_states()
    
    def _initialize_states(self) -> None:
        """Initialize all state objects if they don't exist."""
        if self.form_state_key not in st.session_state:
            st.session_state[self.form_state_key] = FormState()
        
        if self.progress_state_key not in st.session_state:
            st.session_state[self.progress_state_key] = ProgressState()
        
        if self.ui_state_key not in st.session_state:
            st.session_state[self.ui_state_key] = UIState()
    
    # Form State Management
    @property
    def form(self) -> FormState:
        """Get the current form state."""
        return st.session_state[self.form_state_key]
    
    def update_form_field(self, field_name: str, value: Any, 
                         validation_state: FormValidationState = FormValidationState.UNKNOWN,
                         error_message: Optional[str] = None,
                         warning_message: Optional[str] = None) -> None:
        """Update a form field with validation state."""
        form_state = self.form
        
        if hasattr(form_state, field_name):
            field_state = getattr(form_state, field_name)
            if isinstance(field_state, FormFieldState):
                field_state.value = value
                field_state.validation_state = validation_state
                field_state.error_message = error_message
                field_state.warning_message = warning_message
                field_state.last_validated = datetime.now()
            else:
                # For simple fields, just set the value
                setattr(form_state, field_name, value)
        
        # Update session state
        st.session_state[self.form_state_key] = form_state
    
    def get_form_field_value(self, field_name: str, default: Any = None) -> Any:
        """Get the value of a form field."""
        form_state = self.form
        
        if hasattr(form_state, field_name):
            field_value = getattr(form_state, field_name)
            if isinstance(field_value, FormFieldState):
                return field_value.value if field_value.value is not None else default
            else:
                return field_value
        
        return default
    
    def get_form_field_validation(self, field_name: str) -> Optional[FormFieldState]:
        """Get the validation state of a form field."""
        form_state = self.form
        
        if hasattr(form_state, field_name):
            field_value = getattr(form_state, field_name)
            if isinstance(field_value, FormFieldState):
                return field_value
        
        return None
    
    def reset_form(self) -> None:
        """Reset the form state to defaults."""
        st.session_state[self.form_state_key] = FormState()
    
    def is_form_valid(self) -> bool:
        """Check if the form is in a valid state for submission."""
        form_state = self.form
        
        # Check required fields
        url_valid = (form_state.url.validation_state == FormValidationState.VALID and 
                    form_state.url.value)
        name_valid = (form_state.name.value and 
                     len(str(form_state.name.value).strip()) > 0)
        
        return url_valid and name_valid
    
    # Progress State Management
    @property
    def progress(self) -> ProgressState:
        """Get the current progress state."""
        return st.session_state[self.progress_state_key]
    
    def start_progress(self, total_steps: int = 5) -> None:
        """Start a new progress tracking session."""
        progress_state = ProgressState(
            is_active=True,
            current_step=0,
            total_steps=total_steps,
            start_time=datetime.now(),
            completion_status="running"
        )
        st.session_state[self.progress_state_key] = progress_state
    
    def update_progress(self, step: int, message: str, current_url: str = None) -> None:
        """Update the current progress step."""
        progress_state = self.progress
        progress_state.current_step = step
        progress_state.current_message = message
        
        if current_url:
            progress_state.current_url = current_url
        
        # Update processing speed
        if progress_state.start_time:
            elapsed = (datetime.now() - progress_state.start_time).total_seconds()
            if elapsed > 0:
                progress_state.processing_speed = (progress_state.total_processed / elapsed) * 60
        
        st.session_state[self.progress_state_key] = progress_state
    
    def add_progress_log(self, level: str, message: str, url: str = None, details: Dict = None) -> None:
        """Add a log entry to the progress tracking."""
        progress_state = self.progress
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "url": url,
            "details": details
        }
        
        progress_state.log_entries.append(log_entry)
        
        # Update counters
        if level == "ERROR":
            progress_state.error_count += 1
        else:
            progress_state.success_count += 1
        
        progress_state.total_processed = progress_state.success_count + progress_state.error_count
        
        st.session_state[self.progress_state_key] = progress_state
    
    def complete_progress(self, results: Dict[str, Any]) -> None:
        """Complete the progress tracking with final results."""
        progress_state = self.progress
        progress_state.is_active = False
        progress_state.end_time = datetime.now()
        progress_state.final_results = results
        progress_state.completion_status = "completed"
        
        st.session_state[self.progress_state_key] = progress_state
    
    def fail_progress(self, error_message: str) -> None:
        """Mark progress as failed with error message."""
        progress_state = self.progress
        progress_state.is_active = False
        progress_state.end_time = datetime.now()
        progress_state.completion_status = "failed"
        
        # Add error log
        self.add_progress_log("ERROR", error_message)
        
        st.session_state[self.progress_state_key] = progress_state
    
    def reset_progress(self) -> None:
        """Reset the progress state."""
        st.session_state[self.progress_state_key] = ProgressState()
    
    # UI State Management
    @property
    def ui(self) -> UIState:
        """Get the current UI state."""
        return st.session_state[self.ui_state_key]
    
    def set_active_tab(self, tab: str) -> None:
        """Set the active tab."""
        ui_state = self.ui
        ui_state.active_tab = tab
        st.session_state[self.ui_state_key] = ui_state
    
    def toggle_advanced_settings(self) -> None:
        """Toggle the advanced settings visibility."""
        ui_state = self.ui
        ui_state.show_advanced_settings = not ui_state.show_advanced_settings
        st.session_state[self.ui_state_key] = ui_state
    
    def add_chat_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        ui_state = self.ui
        ui_state.chat_history.append({"role": role, "content": content})
        st.session_state[self.ui_state_key] = ui_state
    
    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        ui_state = self.ui
        ui_state.chat_history = []
        st.session_state[self.ui_state_key] = ui_state
    
    def set_selected_collection(self, collection_name: str) -> None:
        """Set the selected collection for chat."""
        ui_state = self.ui
        ui_state.selected_collection = collection_name
        st.session_state[self.ui_state_key] = ui_state
    
    def trigger_success_animation(self) -> None:
        """Trigger the success animation."""
        ui_state = self.ui
        ui_state.show_success_animation = True
        ui_state.success_animation_played = False
        st.session_state[self.ui_state_key] = ui_state
    
    def mark_success_animation_played(self) -> None:
        """Mark the success animation as played."""
        ui_state = self.ui
        ui_state.success_animation_played = True
        st.session_state[self.ui_state_key] = ui_state
    
    def set_error(self, error_message: str) -> None:
        """Set an error message."""
        ui_state = self.ui
        ui_state.last_error = error_message
        ui_state.error_timestamp = datetime.now()
        st.session_state[self.ui_state_key] = ui_state
    
    def clear_error(self) -> None:
        """Clear the current error."""
        ui_state = self.ui
        ui_state.last_error = None
        ui_state.error_timestamp = None
        st.session_state[self.ui_state_key] = ui_state
    
    # Utility Methods
    def cleanup_old_states(self, max_age_hours: int = 24) -> None:
        """Clean up old state data to prevent memory issues."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        # Clean up old log entries
        progress_state = self.progress
        if progress_state.log_entries:
            progress_state.log_entries = [
                entry for entry in progress_state.log_entries
                if datetime.fromisoformat(entry["timestamp"]).timestamp() > cutoff_time
            ]
            st.session_state[self.progress_state_key] = progress_state
        
        # Clean up old chat history if too long
        ui_state = self.ui
        if len(ui_state.chat_history) > 100:
            ui_state.chat_history = ui_state.chat_history[-50:]  # Keep last 50 messages
            st.session_state[self.ui_state_key] = ui_state
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of all current states for debugging."""
        return {
            "form": {
                "url_valid": self.form.url.validation_state == FormValidationState.VALID,
                "name_filled": bool(self.form.name.value),
                "is_submitted": self.form.is_submitted,
                "source_type": self.form.source_type
            },
            "progress": {
                "is_active": self.progress.is_active,
                "current_step": self.progress.current_step,
                "total_steps": self.progress.total_steps,
                "completion_status": self.progress.completion_status,
                "log_entries_count": len(self.progress.log_entries)
            },
            "ui": {
                "active_tab": self.ui.active_tab,
                "selected_collection": self.ui.selected_collection,
                "chat_messages_count": len(self.ui.chat_history),
                "show_success_animation": self.ui.show_success_animation,
                "last_error": self.ui.last_error is not None
            }
        }
    
    def reset_all_states(self) -> None:
        """Reset all application states (use with caution)."""
        st.session_state[self.form_state_key] = FormState()
        st.session_state[self.progress_state_key] = ProgressState()
        st.session_state[self.ui_state_key] = UIState()


# Global state manager instance
def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    if "global_state_manager" not in st.session_state:
        st.session_state["global_state_manager"] = StateManager()
    
    return st.session_state["global_state_manager"]


# Convenience functions for common operations
def get_form_state() -> FormState:
    """Get the current form state."""
    return get_state_manager().form


def get_progress_state() -> ProgressState:
    """Get the current progress state."""
    return get_state_manager().progress


def get_ui_state() -> UIState:
    """Get the current UI state."""
    return get_state_manager().ui


def update_url_validation(url: str, is_valid: bool, error_message: str = None, warning_message: str = None) -> None:
    """Convenience function to update URL validation state."""
    validation_state = FormValidationState.VALID if is_valid else FormValidationState.INVALID
    get_state_manager().update_form_field(
        "url", url, validation_state, error_message, warning_message
    )


def update_name_field(name: str) -> None:
    """Convenience function to update name field."""
    get_state_manager().update_form_field("name", name)


def start_crawling_progress() -> None:
    """Convenience function to start crawling progress."""
    get_state_manager().start_progress(total_steps=5)


def log_crawling_step(level: str, message: str, url: str = None) -> None:
    """Convenience function to log a crawling step."""
    get_state_manager().add_progress_log(level, message, url)