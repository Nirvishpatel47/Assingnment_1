import os
import logging
import re
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet, InvalidToken
import hashlib
from dotenv import load_dotenv
import threading

# --- Encryption/Decryption Functions ---

import logging
from typing import Dict, Any

# Base logger for fallback errors
log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)

if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    log.addHandler(handler)

class logger:
    """Singleton logger with robust error handling for database and security operations."""
    _instance = None
    _lock = threading.Lock()  # ✅ CLASS-LEVEL lock for singleton creation

    def __new__(cls):
        if cls._instance is None:
            # ✅ Use class-level lock for thread-safe singleton creation
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # ✅ Prevent re-initialization
        if self._initialized:
            return
        
        self._initialize()
        self._initialized = True

    def _initialize(self):
        """Initialize the logger with file and console handlers."""
        try:
            self.logger = logging.getLogger('Customer_support')
            self.logger.setLevel(logging.INFO)
            self.logger.handlers.clear()  # Remove old handlers

            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(funcName)-30s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Optional console handler for warnings and above
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        except Exception as e:
            # If logger initialization fails, print instead
            print(f"Logger initialization failed: {e}")

    def info_(self, message: str):
        """Logs INFO messages safely."""
        try:
            self.logger.info(message)
        except Exception as e:
            self._safe_log_error("info", e, {"message": message})

    def log_query(self, operation: str, collection: str, success: bool, duration_ms: float = None):
        """Logs database query operations safely."""
        try:
            status = "SUCCESS" if success else "FAILED"
            duration_str = f" | {duration_ms:.2f}ms" if duration_ms else ""
            self.logger.info(f"QUERY-{operation} | Collection: {collection} | {status}{duration_str}")
        except Exception as e:
            self._safe_log_error("log_query", e, {
                "operation": operation,
                "collection": collection,
                "success": success,
                "duration_ms": duration_ms
            })

    def log_client_operation(self, operation: str, client_id: str, success: bool):
        """Logs client operations safely."""
        try:
            safe_id = client_id[:12] + "..." if len(client_id) > 12 else client_id
            status = "SUCCESS" if success else "FAILED"
            self.logger.info(f"CLIENT-{operation} | Client: {safe_id} | {status}")
        except Exception as e:
            self._safe_log_error("log_client_operation", e, {
                "operation": operation,
                "client_id": client_id,
                "success": success
            })

    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Logs security events safely."""
        try:
            details_str = " | ".join(f"{k}: {v}" for k, v in details.items())
            self.logger.warning(f"SECURITY | {event_type} | {details_str}")
        except Exception as e:
            self._safe_log_error("log_security_event", e, {"event_type": event_type, "details": details})

    def log_error(self, function: str, error: Exception, context: Dict[str, Any] = None):
        """Logs detailed errors safely with filtered context."""
        try:
            context_str = ""
            if context:
                safe_context = {k: v for k, v in context.items() if k not in ['password', 'token', 'api_key', 'secret']}
                context_str = f" | Context: {safe_context}"
            
            error_msg = str(error) if not isinstance(error, str) else error
            self.logger.error(
                f"ERROR in {function} | Type: {type(error).__name__} | "
                f"Message: {error_msg[:200]}{context_str}"
            )
        except Exception as e:
            # If even logging fails, print to console as last resort
            print(f"Critical logging failure in log_error: {e}")

    def _safe_log_error(self, function: str, error: Exception, context: Dict[str, Any] = None):
        """
        Internal fallback: called if any logging function fails.
        Delegates to log_error to ensure the issue is captured.
        """
        try:
            # Avoid recursion if log_error itself fails
            if function != "_safe_log_error":
                self.log_error(function, error, context)
            else:
                print(f"Critical logging failure in _safe_log_error: {error} | Context: {context}")
        except Exception as e:
            print(f"Unhandled logging error: {e} | Original context: {context}")

def get_logger() -> logger:
    """Get a singleton Logger instance."""
    return logger()
# --- Input Sanitization Function ---

import re
import unicodedata
import html
import hashlib

def sanitize_input(text: str) -> str:
    """Simplified sanitizer for RAG input."""
    if not isinstance(text, str): return ""
    # Normalize and remove control characters
    text = unicodedata.normalize("NFKC", text.strip())
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # Basic HTML escaping for safety
    return html.escape(text, quote=True)