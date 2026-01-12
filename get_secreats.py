import os
from dotenv import load_dotenv
from functools import lru_cache

@lru_cache(maxsize=32)
def load_env_from_secret(secrete_name):
    """
    Load environment variable from .env file or system environment.
    """
    try:
        load_dotenv()
        return os.getenv(secrete_name)
    except Exception as e:
        print(f"Error loading environment variable {secrete_name}: {e}")
        return None