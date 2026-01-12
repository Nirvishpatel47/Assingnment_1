"""
Simple String RAG Bot (Retrieval Augmented Generation)

This class provides RAG functionality for a single, large string of text
(e.g., a policy document, a knowledge base dump). It bypasses file loading
and immediately converts the input string into a vector store for querying.

Features retained from UniversalRAGBot:
- LangChain RAG pipeline
- Gemini 2.5 Flash for generation (REST API - NO gRPC)
- FAISS/Embeddings for retrieval
- Prompt injection protection
- Conversation history support
"""

import os
import re
import time
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from encryption_utils import get_logger
from functools import lru_cache
import hashlib
from encryption_utils import sanitize_input
from we_are import We_are

rag_logger = get_logger()
logger = get_logger()

# Third-party imports with error handling
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLLM
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from pydantic import Field
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema
    from get_secreats import load_env_from_secret
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}")


try:
    # Load environment variables
    GEMINI_API_KEY = load_env_from_secret("GEMINI_API_KEY")

    # ✅ CRITICAL FIX: Ensure it's always a plain string
    if hasattr(GEMINI_API_KEY, "get_secret_value"):
        GEMINI_API_KEY = str(GEMINI_API_KEY.get_secret_value())
    else:
        GEMINI_API_KEY = str(GEMINI_API_KEY)
    
    # ✅ Validate it's actually a string now
    if not isinstance(GEMINI_API_KEY, str):
        raise TypeError(f"API key must be string, got {type(GEMINI_API_KEY)}")
    
    rag_logger.logger.info(f"✓ API key loaded (length: {len(GEMINI_API_KEY)})")

except Exception as e:
    rag_logger.log_error("GEMINI_API_KEY -> module. Rag.py", e)
    raise  # ✅ Fail fast if API key is invalid

# Optimized chunk settings for better retrieval
DEFAULT_CHUNK_SIZE = 450
DEFAULT_CHUNK_OVERLAP = 100

DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_MODEL = "gemini-2.5-flash-lite"  # Model name for REST API
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"

# Gemini REST API endpoints
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

schemas = [
    ResponseSchema(name="status", description="true or false"),
    ResponseSchema(name="reason", description="why it’s false"),
    ResponseSchema(name="food_name", description="menu item"),
    ResponseSchema(name="size", description="item size"),
    ResponseSchema(name="price", description="item price"),
    ResponseSchema(name="quantity", description="number of items")
]

# ============================================================================
# CUSTOM GEMINI REST API IMPLEMENTATIONS
# ============================================================================

class GeminiRESTEmbeddings(Embeddings):
    """Custom Gemini Embeddings using REST API instead of gRPC."""
    
    def __init__(self, api_key: str, model: str = DEFAULT_EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"{GEMINI_API_BASE}/models/{model}:embedContent"
        self.batch_endpoint = f"{GEMINI_API_BASE}/models/{model}:batchEmbedContents"
        
    def _make_request(self, texts: List[str], batch: bool = False) -> List[List[float]]:
        """Make REST API request to Gemini."""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            if batch and len(texts) > 1:
                # Batch embedding request
                payload = {
                    "requests": [
                        {
                            "model": f"models/{self.model}",
                            "content": {"parts": [{"text": text}]}
                        }
                        for text in texts
                    ]
                }
                url = f"{self.batch_endpoint}?key={self.api_key}"
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                return [emb["values"] for emb in result.get("embeddings", [])]
            else:
                # Single embedding requests
                embeddings = []
                url = f"{self.endpoint}?key={self.api_key}"
                
                for text in texts:
                    payload = {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": text}]}
                    }
                    
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    response.raise_for_status()
                    
                    result = response.json()
                    embedding = result.get("embedding", {}).get("values", [])
                    embeddings.append(embedding)
                
                return embeddings
                
        except requests.exceptions.RequestException as e:
            rag_logger.log_error("GeminiRESTEmbeddings._make_request", e)
            raise RuntimeError(f"Embedding request failed: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        # Process in batches of 100 (Gemini's limit)
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._make_request(batch, batch=True)
            all_embeddings.extend(embeddings)
            
            # Rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embeddings = self._make_request([text], batch=False)
        return embeddings[0] if embeddings else []


class GeminiRESTChat(BaseLLM):
    """Custom Gemini Chat using REST API instead of gRPC."""
    
    api_key: str = Field(description="Gemini API key")
    model: str = Field(default=DEFAULT_MODEL, description="Model name")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, description="Temperature")
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, description="Max tokens")
    top_p: float = Field(default=0.95, description="Top P")
    top_k: int = Field(default=40, description="Top K")
    endpoint: str = Field(default="", description="API endpoint")
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        top_p: float = 0.95,
        top_k: int = 40,
        **kwargs
    ):
        # Initialize attributes before calling super().__init__
        endpoint = f"{GEMINI_API_BASE}/models/{model}:generateContent"
        
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            endpoint=endpoint,
            **kwargs
        )
    
    @property
    def _llm_type(self) -> str:
        return "gemini-rest"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> Any:
        """Generate responses for prompts."""
        from langchain_core.outputs import LLMResult, Generation
        
        generations = []
        for prompt in prompts:
            try:
                response_text = self._call_api(prompt)
                generations.append([Generation(text=response_text)])
            except Exception as e:
                rag_logger.log_error("GeminiRESTChat._generate", e)
                generations.append([Generation(text="Error generating response.")])
        
        return LLMResult(generations=generations)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> str:
        """Call the Gemini API."""
        return self._call_api(prompt)
    
    def _call_api(self, prompt: str) -> str:
        """Make REST API call to Gemini with proper safety controls."""
        try:
            headers = {"Content-Type": "application/json"}

            payload = {
                "contents": [
                    {"parts": [{"text": prompt}]}
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": 2048,
                    "topP": self.top_p,
                    "topK": self.top_k,
                    "stopSequences": []
                },
                # ✅ FIXED: Use valid threshold values
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                ]
            }

            url = f"{self.endpoint}?key={self.api_key}"

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
            except requests.Timeout:
                rag_logger.log_error("GeminiRESTChat._call_api_timeout", "Request timed out")
                return "⌛ The request took too long — maybe the servers need a quick coffee break. Please try again soon ☕"
            except requests.exceptions.HTTPError as http_err:
                # ✅ NEW: Better error handling for 400 errors
                status_code = http_err.response.status_code
                error_body = http_err.response.text[:500]
                
                rag_logger.log_error(
                    "GeminiRESTChat._call_api_http_status", 
                    f"HTTP {status_code}: {error_body}"
                )
                
                if status_code == 400:
                    return "Hmm… the request didn’t look quite right. Could you please rephrase that?"
                elif status_code == 429:
                    return "😅 Whoa, too many requests at once! Let’s pause for a second and try again."
                elif status_code >= 500:
                    return "The service seems to be having a rough day. Try again after a moment!"
                else:
                    return "😕 I’m having trouble reaching the service right now. Could you try again later?"
                    
            except requests.ConnectionError as e:
                rag_logger.log_error("GeminiRESTChat._call_api_connection", e)
                return "🌐 Looks like there’s a little network hiccup. Please check your connection and try again!"
            except requests.RequestException as e:
                rag_logger.log_error("GeminiRESTChat._call_api_http", e)
                return "😕 I’m having trouble reaching the service right now. Could you try again later?"

            try:
                result = response.json()
            except ValueError as e:
                rag_logger.log_error("GeminiRESTChat._call_api_json_error", f"Invalid JSON: {e}, raw={response.text[:200]}")
                return "🤖 The server sent something strange that I couldn’t read. Let’s give it another try!"

            candidates = result.get("candidates")
            if not candidates or not isinstance(candidates, list):
                rag_logger.log_error("_call_api", f"Malformed response (no candidates): {result}")
                return "🙇 Sorry, I couldn’t process that properly. Mind trying again?"

            candidate = candidates[0] or {}
            finish_reason = (candidate.get("finishReason") or "").upper()

            # Safety or truncation handling
            if finish_reason == "SAFETY":
                rag_logger.log_error("_call_api", "Response blocked by safety filters")
                return "I cannot generate a response because it may violate content policies."
            elif finish_reason == "MAX_TOKENS":
                rag_logger.logger.warning("⚠️ Response truncated due to max tokens")

            content = candidate.get("content") or {}
            parts = content.get("parts")
            if not parts or not isinstance(parts, list):
                rag_logger.log_error("_call_api", f"Malformed candidate content: {candidate}")
                return "I apologize, but I'm having trouble processing your request right now."

            # Join multiple text parts safely
            try:
                text = " ".join(
                    p.get("text", "").strip()
                    for p in parts
                    if isinstance(p, dict) and p.get("text")
                )
            except Exception as e:
                rag_logger.log_error("_call_api_text_extraction", e)
                return "An internal error occurred while parsing the response."

            if not text.strip():
                rag_logger.log_error("_call_api", f"Empty text in response: {candidate}")
                return "I apologize, but I'm having trouble processing your request right now."

            # Warn if incomplete response
            if len(text) < 50 or not text.strip().endswith(('.', '!', '?', '।', '।।')):
                rag_logger.logger.warning(
                    f"⚠️ Potentially incomplete response: len={len(text)}, finish_reason={finish_reason}"
                )

            return text.strip()

        except Exception as e:
            rag_logger.log_error("GeminiRESTChat._call_api_unexpected", e)
            return "I encountered an unexpected error. Please try rephrasing your question."

    
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> Any:
        """Async generate - falls back to sync."""
        return self._generate(prompts, stop, run_manager, **kwargs)

import time
import hashlib
from threading import Lock
from typing import List, Dict, Optional, Any

class RAGCache:
    """Thread-safe in-memory cache for RAG operations with sanitized inputs."""

    def __init__(self, max_history_size: int = 50, max_query_cache: int = 200):
        self._lock = Lock()
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.max_history_size = max_history_size
        self.max_query_cache = max_query_cache
        self.vectorstore = None
        self.bm25_retriever = None
        self.chunks = None
        self.similarity_threshold = 0.3
        
    def add_to_history(self, client_id: str, role: str, content: str):
        """Add conversation turn safely with input validation."""
        try:
            if not isinstance(content, str) or not content.strip():
                return
            with self._lock:
                self.conversation_history.setdefault(client_id, []).append({
                    'role': role,
                    'content': content.strip(),
                    'timestamp': time.time()
                })
                self.conversation_history[client_id] = self.conversation_history[client_id][-self.max_history_size:]
        except Exception as e:
            rag_logger.log_error("add_to_history. Rag.py", e)

    def get_history(self, client_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self.conversation_history.get(client_id, []))

    def clear_history(self, client_id: str):
        with self._lock:
            self.conversation_history.pop(client_id, None)

    def cache_query_result(self, query: str, context: str, relevance_score: float = 1.0):
        """Cache query safely with thread-locking."""
        try:
            if not query or not context:
                return
            query_hash = hashlib.sha256(query.lower().encode()).hexdigest()
            with self._lock:
                if len(self.query_cache) >= self.max_query_cache:
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                self.query_cache[query_hash] = {
                    'context': context,
                    'timestamp': time.time(),
                    'relevance': relevance_score
                }
        except Exception as e:
            rag_logger.log_error("cache_query_result. Rag.py", e)

    def get_cached_query(self, query: str, max_age_seconds: int = 600) -> Optional[Dict]:
        try:
            if not query:
                return None
            query_hash = hashlib.sha256(query.lower().encode()).hexdigest()
            with self._lock:
                data = self.query_cache.get(query_hash)
                if data and (time.time() - data['timestamp'] <= max_age_seconds):
                    return data
                self.query_cache.pop(query_hash, None)
            return None
        except Exception as e:
            rag_logger.log_error("cache_query_result. Rag.py", e)

    def store_vectorstore(self, vectorstore, bm25_retriever, chunks):
        with self._lock:
            self.vectorstore = vectorstore
            self.bm25_retriever = bm25_retriever
            self.chunks = chunks

    def has_vectorstore(self) -> bool:
        with self._lock:
            return self.vectorstore is not None
        
    def get(self, key: str):
        """Retrieve a cached value by key, if available."""
        try:
            with self._lock:
                return self.query_cache.get(key)
        except Exception as e:
            rag_logger.log_error("cache.get", e)
            return None

    def set(self, key: str, value: Any):
        """Store a value in cache with automatic eviction if full."""
        try:
            with self._lock:
                if len(self.query_cache) >= self.max_query_cache:
                    # remove oldest item (FIFO)
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                self.query_cache[key] = {
                    'value': value,
                    'timestamp': time.time()
                }
        except Exception as e:
            rag_logger.log_error("cache.set", e)


class RAGBot:
    """
    Enhanced RAG bot with improved retrieval and answer generation.
    Uses Gemini REST API - NO gRPC.
    """
    
    def __init__(self, client_id: str, document_text: str = We_are(), top_k: int = DEFAULT_TOP_K ):
        """Initialize RAG bot with enhanced configuration."""
        try:
        
            if not document_text or not isinstance(document_text, str):
                raise ValueError("Document text must be a non-empty string.")
            document_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', document_text).strip()
            self.document_text = document_text[:100000]
            self.client_id = client_id
            self.top_k = max(3, min(top_k, 10))

            self.user_name = "User"  # Default value
            self.user_goal = []      # Default value
            self.cache = RAGCache()
            
            self.llm = None
            self.embeddings = None
            self.retriever = None
            self.chain = None
            self.translation_chain = None 
            self.chain_res = None

            # Setup pipeline with REST API
            self._initialize_llm()
            self._initialize_embeddings()
            self._setup_retriever_from_string()
            self._setup_chain()
            
            rag_logger.logger.info(f"Enhanced RAGBot initialized for client: {client_id} (REST API)")
            
        except Exception as e:
            rag_logger.log_error("RAGBot.__init__", e)
            raise RuntimeError(f"Failed to initialize RAG bot: {str(e)}")
    
    def _initialize_llm(self):
        """Initialize Gemini LLM with REST API."""
        try:
            self.llm = GeminiRESTChat(
                api_key=GEMINI_API_KEY,
                model=DEFAULT_MODEL,
                temperature=DEFAULT_TEMPERATURE,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                top_p=0.95,
                top_k=40
            )
            rag_logger.logger.info("✓ LLM initialized (REST API)")
        except Exception as e:
            rag_logger.log_error("_initialize_llm", e)
            raise RuntimeError("Failed to initialize LLM")
    
    def _initialize_embeddings(self):
        """Initialize embeddings with REST API."""
        try:
            api_key = GEMINI_API_KEY
            
            if not isinstance(api_key, str):
                raise TypeError(f"API key must be string, got {type(api_key)}")
            
            if len(api_key) < 20:
                raise ValueError(f"API key too short: {len(api_key)} chars")
            
            rag_logger.logger.info(f"✓ Initializing embeddings with key length: {len(api_key)}")
            
            self.embeddings = GeminiRESTEmbeddings(
                api_key=api_key,
                model=DEFAULT_EMBEDDING_MODEL
            )
            
            rag_logger.logger.info("✓ Embeddings initialized successfully (REST API)")
            
        except Exception as e:
            rag_logger.log_error("_initialize_embeddings", e)
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")
    
    def _setup_retriever_from_string(self):
        """Setup enhanced retriever with better chunking strategy."""
        try:
            if self.cache.has_vectorstore():
                rag_logger.logger.info("✓ Using cached vectorstore")
                vectorstore = self.cache.vectorstore
                bm25_retriever = self.cache.bm25_retriever
                chunks = self.cache.chunks
            else:
                base_doc = Document(
                    page_content=self.document_text,
                    metadata={'source': self.client_id, 'doc_type': 'StringInput'}
                )
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    separators=[
                        "\n\n\n",
                        "\n\n",
                        "\n",
                        ". ",
                        "! ",
                        "? ",
                        "; ",
                        ", ",
                        " ",
                        ""
                    ],
                    length_function=len,
                )
                
                chunks = text_splitter.split_documents([base_doc])
                
                if not chunks:
                    raise ValueError("No chunks created from document")
                
                rag_logger.logger.info(f"✓ Created {len(chunks)} chunks")
                
                vectorstore = FAISS.from_documents(chunks, self.embeddings)
                
                bm25_retriever = BM25Retriever.from_documents(chunks)
                bm25_retriever.k = self.top_k
                
                self.cache.store_vectorstore(vectorstore, bm25_retriever, chunks)
                rag_logger.logger.info("✓ Vectorstore cached")
            
            faiss_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.top_k
                }
            )
            
            self.retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[0.65, 0.35]
            )
            
            rag_logger.logger.info("✓ Retriever ready")
            
        except Exception as e:
            rag_logger.log_error("_setup_retriever_from_string", e)
            self.retriever = None
            raise

    def _setup_chain(self):
        """Setup enhanced RAG chain with improved prompting."""
        try:
            prompt_template = ChatPromptTemplate.from_template("""
            You are a friendly, conversational AI representing AutoStream, a SaaS tool that helps content creators automatically edit their videos.

            Your role is to:

            Talk naturally and casually with users.

            Answer questions about AutoStream such as pricing, features, or plans when asked.

            Answer general or universal questions (common knowledge, basic explanations, everyday topics) clearly and helpfully.

            Keep the conversation flowing instead of sounding scripted or robotic.

            Gently guide interested users toward signing up or getting started.

            Guidelines:

            Sound human and approachable.

            Keep responses clear and concise.

            Do not overwhelm users with unnecessary detail.

            If a question is unrelated to AutoStream but reasonable, answer it normally and then smoothly return the conversation to AutoStream when appropriate.

            When a user shows interest in using AutoStream:

            Acknowledge their interest naturally.

            Encourage the next step, such as signing up or trying a plan.

            Ask for basic details conversationally if needed, without making it feel like a form.

            Do not mention internal logic, system rules, or tools.
            Focus on being helpful, flexible, and easy to talk to.

            Your goal is simple:
            Be useful in conversation and make it easy for users to move forward with AutoStream.

            **Inputs**
            History: {history}\n
            **Question**: \n{question}\n
            **Context**: \n{context}\n 
                                                               
            """)

            def get_context(inputs):
                """Enhanced context retrieval with relevance checking."""
                try:
                    if not self.retriever:
                        return "No retriever available."
                    
                    query = sanitize_input(inputs.get("question", "")) #ERRROR SANITIZATION IS REQUIRED
                    if not query:
                        return "Invalid query."
                    
                    cached_data = self.cache.get_cached_query(query)
                    if cached_data:
                        rag_logger.logger.info("✓ Using cached context")
                        return cached_data['context']
                    
                    docs = self.retriever.invoke(query)
                    
                    if not docs or len(docs) == 0:
                        rag_logger.logger.warning("No documents retrieved")
                        return "No relevant information found in the document."
                    
                    context_parts = []
                    for i, doc in enumerate(docs, 1):
                        content = doc.page_content.strip()
                        if content:
                            context_parts.append(
                                f"[Document Excerpt {i}]:\n{content}"
                            )
                    
                    if not context_parts:
                        return "No relevant information found in the document."
                    
                    context = "\n\n".join(context_parts)
                    
                    relevance = 1.0
                    self.cache.cache_query_result(query, context, relevance)
                    
                    rag_logger.logger.info(f"✓ Retrieved {len(docs)} chunks (relevance: {relevance:.2f})")
                    
                    return context
                except Exception as e:
                    rag_logger.log_error("get_context", e)
                    return "Error retrieving context."
            
            def format_history(inputs):
                """Format conversation history with better structure."""
                try:
                    history = self.cache.get_history(self.client_id)
                    
                    if not history:
                        return "No previous conversation."
                    
                    recent_history = history[-1:] if len(history) > 1 else history
                    
                    formatted = []
                    for exchange in recent_history:
                        if isinstance(exchange, dict):
                            role = exchange.get('role', 'unknown')
                            content = exchange.get('content', '')
                            if role == 'user':
                                formatted.append(f"User asked: {content}")
                            elif role == 'ai':
                                formatted.append(f"Assistant replied: {content}")
                    
                    return "\n".join(formatted) if formatted else "No previous conversation."
                        
                except Exception as e:
                    rag_logger.log_error("format_history", e)
                    return "No previous conversation."
            
            self.chain = (
                RunnableParallel({
                    "context": get_context,
                    "question": lambda x: sanitize_input(x.get("question", "")), #ERROR SENITIZATION IS NEEDED
                    "history": format_history
                })
                | prompt_template
                | self.llm
                | StrOutputParser()
            )

        except Exception as e:
            rag_logger.log_error("_setup_chain", e)
            self.chain = None

    async def invoke(self, query: str, launguage: str) -> str:
        """
        Process query and generate intelligent response.
        """
        try:
            if not query or not isinstance(query, str):
                return "Please provide a question."
            
            if not self.chain:
                return "System not ready. Please try again."
            
            query = str(query)

            self.cache.add_to_history(self.client_id, 'user', query)

            inputs = {
                "question": sanitize_input(query), 
                "launguge": str(launguage)
            }
            
            start_time = time.time()
            response = self.chain.invoke(inputs)
            duration_ms = (time.time() - start_time) * 1000
            
            # ✅ NEW: Validate response completeness
            if not response or len(response.strip()) < 3:
                response = "I apologize, but I couldn't generate a proper response. Could you rephrase your question?"
            
            # ✅ NEW: Check if response seems incomplete
            response = response.strip()
            
            # Check for incomplete sentences (doesn't end with proper punctuation)
            if response and len(response) > 20:
                last_char = response[-1]
                if last_char not in '.!?।।।':
                    rag_logger.logger.warning(f"⚠️ Response may be incomplete (no ending punctuation)")
                    # Add ellipsis to indicate incompleteness
                    response += "..."
            
            self.cache.add_to_history(self.client_id, 'ai', response)
            
            rag_logger.logger.info(f"✅ Response generated in {duration_ms:.0f}ms (length: {len(response)})")
            
            return response
            
        except Exception as e:
            from encryption_utils import hash_for_logging
            rag_logger.log_error("invoke", e, {
                "client_id": hash_for_logging(self.client_id), 
                "query_length": len(query) if query else 0
            })
            return "I encountered an error processing your request. Please try again later."

    def clear_conversation(self):
        """Clear conversation history for this client."""
        self.cache.clear_history(self.client_id)
        rag_logger.logger.info(f"✓ Cleared history for {self.client_id}")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.cache.get_history(self.client_id)

    def clear_query_cache(self):
        """Clear the query result cache."""
        self.cache.query_cache.clear()
        rag_logger.logger.info("✓ Query cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG bot statistics."""
        return {
            "client_id": self.client_id,
            "document_length": len(self.document_text),
            "cached_chunks": len(self.cache.chunks) if self.cache.chunks else 0,
            "conversation_length": len(self.cache.get_history(self.client_id)),
            "query_cache_size": len(self.cache.query_cache),
            "vectorstore_cached": self.cache.has_vectorstore()
        }