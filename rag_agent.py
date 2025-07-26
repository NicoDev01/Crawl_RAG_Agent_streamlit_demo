"""Pydantic AI agent that leverages RAG with a local ChromaDB for crawled website content."""

import os
import sys
import argparse
import hashlib
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator

# ===== STRUCTURED DATA MODELS =====

# Input Data Models
class QueryStrategy(str, Enum):
    """Strategy for query complexity and variation generation."""
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"

class QueryVariations(BaseModel):
    """Model for multi-query generation with validation."""
    original_query: str = Field(..., description="The original user query")
    strategy: QueryStrategy = Field(..., description="Complexity strategy for query generation")
    variations: List[str] = Field(default_factory=list, description="Generated query variations")
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Complexity score (0.0 = simple, 1.0 = complex)")
    
    @validator('variations')
    def validate_variations(cls, v, values):
        """Validate query variations based on strategy."""
        if 'strategy' not in values:
            return v
            
        strategy = values['strategy']
        max_variations = {
            QueryStrategy.SIMPLE: 1,
            QueryStrategy.MODERATE: 2, 
            QueryStrategy.COMPLEX: 3
        }
        
        max_allowed = max_variations.get(strategy, 3)
        if len(v) > max_allowed:
            raise ValueError(f"Too many variations for {strategy} strategy. Max allowed: {max_allowed}")
        
        # Validate that variations are not empty and different from original
        original = values.get('original_query', '').lower().strip()
        for variation in v:
            if not variation.strip():
                raise ValueError("Query variations cannot be empty")
            if variation.lower().strip() == original:
                raise ValueError("Query variations must be different from original query")
        
        return v
    
    @validator('complexity_score')
    def validate_complexity_score(cls, v, values):
        """Ensure complexity score matches strategy."""
        if 'strategy' not in values:
            return v
            
        strategy = values['strategy']
        expected_ranges = {
            QueryStrategy.SIMPLE: (0.0, 0.33),
            QueryStrategy.MODERATE: (0.33, 0.66),
            QueryStrategy.COMPLEX: (0.66, 1.0)
        }
        
        min_score, max_score = expected_ranges[strategy]
        if not (min_score <= v <= max_score):
            raise ValueError(f"Complexity score {v} doesn't match {strategy} strategy range [{min_score}, {max_score}]")
        
        return v

# Processing Data Models
class DocumentMetadata(BaseModel):
    """Metadata for a document chunk."""
    url: str = Field(..., description="Source URL of the document")
    title: Optional[str] = Field(None, description="Document title if available")
    chunk_index: int = Field(..., ge=0, description="Index of this chunk within the document")
    total_chunks: Optional[int] = Field(None, ge=1, description="Total number of chunks for this document")
    timestamp: Optional[str] = Field(None, description="When the document was crawled/processed")
    
    @validator('url')
    def validate_url(cls, v):
        """Ensure URL is not empty and has basic structure."""
        if not v.strip():
            raise ValueError("URL cannot be empty")
        if not (v.startswith('http://') or v.startswith('https://') or v.startswith('file://')):
            raise ValueError("URL must start with http://, https://, or file://")
        return v.strip()

class DocumentChunk(BaseModel):
    """A chunk of document content with metadata."""
    content: str = Field(..., description="The actual text content of the chunk")
    metadata: DocumentMetadata = Field(..., description="Metadata for this chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for this chunk")
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    
    @validator('content')
    def validate_content(cls, v):
        """Ensure content is not empty and has reasonable length."""
        if not v.strip():
            raise ValueError("Document content cannot be empty")
        if len(v.strip()) < 3:
            raise ValueError("Document content too short (minimum 3 characters)")
        return v.strip()
    
    @validator('chunk_id')
    def validate_chunk_id(cls, v):
        """Ensure chunk ID is not empty."""
        if not v.strip():
            raise ValueError("Chunk ID cannot be empty")
        return v.strip()

class RankedDocument(BaseModel):
    """A document with its relevance score."""
    document: DocumentChunk = Field(..., description="The document chunk")
    score: float = Field(..., description="Relevance score for this document")
    rank: int = Field(..., ge=1, description="Rank position (1 = most relevant)")
    
    @validator('score')
    def validate_score(cls, v):
        """Ensure score is a valid number."""
        if not isinstance(v, (int, float)):
            raise ValueError("Score must be a number")
        return float(v)

class RankedDocuments(BaseModel):
    """Collection of ranked documents with metadata."""
    documents: List[RankedDocument] = Field(..., description="List of ranked documents")
    total_candidates: int = Field(..., ge=0, description="Total number of candidate documents before ranking")
    ranking_method: str = Field(..., description="Method used for ranking (e.g., 'vertex_ai', 'score_based')")
    query: str = Field(..., description="Original query used for ranking")
    
    @validator('documents')
    def validate_documents(cls, v):
        """Ensure documents are properly ranked."""
        if not v:
            return v
        
        # Check that ranks are sequential starting from 1
        expected_ranks = list(range(1, len(v) + 1))
        actual_ranks = [doc.rank for doc in v]
        
        if sorted(actual_ranks) != expected_ranks:
            raise ValueError("Document ranks must be sequential starting from 1")
        
        # Check that documents are sorted by rank
        for i in range(len(v) - 1):
            if v[i].rank > v[i + 1].rank:
                raise ValueError("Documents must be sorted by rank (ascending)")
        
        return v

class RetrievalResult(BaseModel):
    """Result of document retrieval with metadata."""
    query_variations: QueryVariations = Field(..., description="Query variations used for retrieval")
    ranked_documents: RankedDocuments = Field(..., description="Retrieved and ranked documents")
    retrieval_time: float = Field(..., ge=0, description="Time taken for retrieval in seconds")
    embedding_cache_hits: int = Field(default=0, ge=0, description="Number of embedding cache hits")
    embedding_cache_misses: int = Field(default=0, ge=0, description="Number of embedding cache misses")
    
    @validator('retrieval_time')
    def validate_retrieval_time(cls, v):
        """Ensure retrieval time is reasonable."""
        if v > 300:  # 5 minutes seems like a reasonable upper bound
            raise ValueError("Retrieval time seems unreasonably long (>300 seconds)")
        return v

# Output Data Models
class SourceReference(BaseModel):
    """Structured source reference with metadata."""
    url: str = Field(..., description="Source URL")
    title: Optional[str] = Field(None, description="Document title if available")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score for this source")
    citation_number: int = Field(..., ge=1, description="Citation number for referencing")
    excerpt: Optional[str] = Field(None, description="Brief excerpt from the source")
    
    @validator('url')
    def validate_url(cls, v):
        """Ensure URL is valid."""
        if not v.strip():
            raise ValueError("Source URL cannot be empty")
        if not (v.startswith('http://') or v.startswith('https://') or v.startswith('file://')):
            raise ValueError("URL must start with http://, https://, or file://")
        return v.strip()
    
    @validator('excerpt')
    def validate_excerpt(cls, v):
        """Ensure excerpt is reasonable length if provided."""
        if v is not None:
            v = v.strip()
            if len(v) > 200:
                raise ValueError("Excerpt should be max 200 characters")
            if len(v) < 10:
                raise ValueError("Excerpt should be at least 10 characters if provided")
        return v

class RetrievalMetadata(BaseModel):
    """Metadata about the retrieval process and quality metrics."""
    total_documents_searched: int = Field(..., ge=0, description="Total documents in the search space")
    documents_retrieved: int = Field(..., ge=0, description="Number of documents retrieved")
    queries_used: int = Field(..., ge=1, description="Number of query variations used")
    retrieval_method: str = Field(..., description="Method used for retrieval (e.g., 'hyde_multi_query')")
    reranking_method: Optional[str] = Field(None, description="Re-ranking method used if any")
    average_relevance_score: float = Field(..., ge=0.0, le=1.0, description="Average relevance score of retrieved documents")
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Cache hit rate for this retrieval")
    processing_time_seconds: float = Field(..., ge=0, description="Total processing time in seconds")
    
    @validator('documents_retrieved')
    def validate_documents_retrieved(cls, v, values):
        """Ensure retrieved documents don't exceed total searched."""
        if 'total_documents_searched' in values and v > values['total_documents_searched']:
            raise ValueError("Retrieved documents cannot exceed total documents searched")
        return v

# Enhanced structured output model for RAG responses
class StructuredRagAnswer(BaseModel):
    """
    Enhanced structured model for RAG system responses with comprehensive metadata.
    """
    summary: str = Field(..., description="Eine prägnante Zusammenfassung der Antwort in 2-3 Sätzen.")
    key_details: List[str] = Field(..., description="Eine Liste der wichtigsten Fakten, Details oder Eigenschaften.")
    contact_info: Optional[str] = Field(None, description="Kontaktinformationen falls verfügbar (E-Mail, Telefon).")
    source_references: List[SourceReference] = Field(..., description="Strukturierte Quellenverweise mit Metadaten.")
    confidence_score: float = Field(..., ge=0, le=1, description="Konfidenzwert (0.0 bis 1.0) basierend auf der Qualität der gefundenen Dokumente.")
    retrieval_metadata: RetrievalMetadata = Field(..., description="Metadaten über den Retrieval-Prozess.")
    answer_type: str = Field(default="informational", description="Type of answer (informational, definition, procedural, etc.)")
    
    # Legacy compatibility - will be deprecated
    sources: List[str] = Field(default_factory=list, description="Legacy: Liste der verwendeten Quellen-URLs (use source_references instead).")
    
    @validator('key_details')
    def validate_key_details(cls, v):
        """Ensure key details are meaningful."""
        if not v:
            raise ValueError("At least one key detail must be provided")
        
        for detail in v:
            if not detail.strip():
                raise ValueError("Key details cannot be empty")
            if len(detail.strip()) < 5:
                raise ValueError("Key details should be at least 5 characters")
        
        return [detail.strip() for detail in v]
    
    @validator('summary')
    def validate_summary(cls, v):
        """Ensure summary is meaningful."""
        if not v.strip():
            raise ValueError("Summary cannot be empty")
        if len(v.strip()) < 20:
            raise ValueError("Summary should be at least 20 characters")
        if len(v.strip()) > 500:
            raise ValueError("Summary should be max 500 characters")
        return v.strip()
    
    @validator('sources', always=True)
    def populate_legacy_sources(cls, v, values):
        """Populate legacy sources field from source_references for backward compatibility."""
        if 'source_references' in values and values['source_references']:
            return [ref.url for ref in values['source_references']]
        return v or []
import asyncio
import chromadb
import logfire
import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
# GoogleModel import with proper provider setup
try:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider
    GOOGLE_MODEL_AVAILABLE = True
    print("✅ GoogleModel and GoogleProvider imported successfully")
except ImportError as e:
    print(f"⚠️ GoogleModel not available: {e}")
    print("💡 To fix this, install: pip install 'pydantic-ai-slim[google]' or pip install pydantic-ai")
    GoogleModel = None
    GoogleProvider = None
    GOOGLE_MODEL_AVAILABLE = False
from openai import AsyncOpenAI

# Import for Vertex AI Reranker
from google.cloud.discoveryengine_v1.services.rank_service import RankServiceAsyncClient
from google.cloud.discoveryengine_v1.types import RankRequest, RankingRecord

from utils import (
    get_chroma_client,
    get_or_create_collection
)
from vertex_ai_utils import get_vertex_text_embedding, init_vertex_ai

# Load environment variables from .env file
dotenv.load_dotenv()

# Check for OpenAI API key (nur für CLI, nicht für Streamlit)
if not os.getenv("OPENAI_API_KEY") and __name__ == "__main__":
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)

# Initialize OpenAI client (fallback)
aclient = AsyncOpenAI()

# Gemini integration now handled natively through Pydantic AI GoogleModel


# ===== OPTIMIZATION: LLM Response Cache =====
class LLMResponseCache:
    """Intelligent cache for LLM responses based on question and context content."""
    
    def __init__(self, ttl_hours: int = 24, max_size: int = 500):
        self.cache = {}  # content_hash -> (response, timestamp, access_count)
        self.ttl = ttl_hours * 3600  # Convert to seconds
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
    
    def _normalize_context(self, context: str) -> str:
        """Normalize context by removing URLs and timestamps, keeping content."""
        import re
        # Remove URLs but keep the content structure
        normalized = re.sub(r'https?://[^\s]+', '', context)
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _get_cache_key(self, question: str, context: str) -> str:
        """Generate cache key from question and normalized context."""
        normalized_context = self._normalize_context(context)
        content = f"{question.lower().strip()}|{normalized_context}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp, _) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _evict_lru(self):
        """Evict least recently used entries if cache is full."""
        if len(self.cache) >= self.max_size:
            # Sort by access count (LRU approximation)
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: (x[1][2], x[1][1])  # Sort by access_count, then timestamp
            )
            # Remove oldest 20% of entries
            remove_count = max(1, len(sorted_items) // 5)
            for key, _ in sorted_items[:remove_count]:
                del self.cache[key]
    
    def get(self, question: str, context: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        self._cleanup_expired()
        
        cache_key = self._get_cache_key(question, context)
        
        if cache_key in self.cache:
            response, timestamp, access_count = self.cache[cache_key]
            # Update access count for LRU
            self.cache[cache_key] = (response, timestamp, access_count + 1)
            self.hit_count += 1
            print(f"🎯 Cache HIT for question: {question[:50]}...")
            return response
        
        self.miss_count += 1
        return None
    
    def store(self, question: str, context: str, response: str):
        """Store response in cache."""
        self._cleanup_expired()
        self._evict_lru()
        
        cache_key = self._get_cache_key(question, context)
        current_time = time.time()
        
        self.cache[cache_key] = (response, current_time, 1)
        print(f"💾 Cached response for: {question[:50]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": round(hit_rate, 1),
            "max_size": self.max_size
        }


# ===== OPTIMIZATION: Query Embedding Cache =====
class QueryEmbeddingCache:
    """Cache for query embeddings to avoid redundant API calls."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}  # query_hash -> (embedding, timestamp, access_count)
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query."""
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode('utf-8')).hexdigest()
    
    def _evict_lru(self):
        """Evict least recently used entries if cache is full."""
        if len(self.cache) >= self.max_size:
            # Sort by access count (LRU approximation)
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: (x[1][2], x[1][1])  # Sort by access_count, then timestamp
            )
            # Remove oldest 20% of entries
            remove_count = max(1, len(sorted_items) // 5)
            for key, _ in sorted_items[:remove_count]:
                del self.cache[key]
    
    def get(self, query: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        query_hash = self._get_query_hash(query)
        
        if query_hash in self.cache:
            embedding, timestamp, access_count = self.cache[query_hash]
            # Update access count for LRU
            self.cache[query_hash] = (embedding, timestamp, access_count + 1)
            self.hit_count += 1
            print(f"🎯 Embedding cache HIT for: {query[:30]}...")
            return embedding
        
        self.miss_count += 1
        return None
    
    def store(self, query: str, embedding: List[float]):
        """Store embedding in cache."""
        self._evict_lru()
        
        query_hash = self._get_query_hash(query)
        current_time = time.time()
        
        self.cache[query_hash] = (embedding, current_time, 1)
        print(f"💾 Cached embedding for: {query[:30]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": round(hit_rate, 1),
            "max_size": self.max_size
        }


# Initialize global cache instances
llm_cache = LLMResponseCache(ttl_hours=1, max_size=500)
embedding_cache = QueryEmbeddingCache(max_size=1000)

# ===== PYDANTIC AI CACHE INTEGRATION =====

class PydanticAICacheConfig(BaseModel):
    """Configuration for Pydantic AI cache integration."""
    enable_llm_cache: bool = Field(default=True, description="Enable LLM response caching")
    enable_embedding_cache: bool = Field(default=True, description="Enable embedding caching")
    llm_cache_ttl_hours: int = Field(default=1, ge=1, le=1, description="LLM cache TTL in hours")
    llm_cache_max_size: int = Field(default=500, ge=10, le=10000, description="Max LLM cache entries")
    embedding_cache_max_size: int = Field(default=1000, ge=10, le=10000, description="Max embedding cache entries")
    cache_hit_logging: bool = Field(default=True, description="Log cache hits/misses")

class CacheMetrics(BaseModel):
    """Cache performance metrics."""
    llm_cache_stats: Dict[str, Any] = Field(default_factory=dict)
    embedding_cache_stats: Dict[str, Any] = Field(default_factory=dict)
    total_cache_hits: int = Field(default=0)
    total_cache_misses: int = Field(default=0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)

class BatchProcessingConfig(BaseModel):
    """Configuration for optimized batch processing."""
    max_batch_size: int = Field(default=10, ge=1, le=50, description="Maximum batch size for parallel processing")
    max_concurrent_batches: int = Field(default=3, ge=1, le=10, description="Maximum concurrent batch operations")
    batch_timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0, description="Timeout for batch operations")
    enable_batch_optimization: bool = Field(default=True, description="Enable batch processing optimizations")
    adaptive_batch_sizing: bool = Field(default=True, description="Automatically adjust batch sizes based on performance")

class BatchProcessingMetrics(BaseModel):
    """Metrics for batch processing performance."""
    total_batches_processed: int = Field(default=0, ge=0)
    average_batch_size: float = Field(default=0.0, ge=0.0)
    average_processing_time: float = Field(default=0.0, ge=0.0)
    successful_batches: int = Field(default=0, ge=0)
    failed_batches: int = Field(default=0, ge=0)
    batch_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)

# ===== STRUCTURED ERROR HANDLING =====

class ErrorType(str, Enum):
    """Types of errors that can occur in the RAG system."""
    MODEL_ERROR = "model_error"
    RETRIEVAL_ERROR = "retrieval_error"
    EMBEDDING_ERROR = "embedding_error"
    RERANKING_ERROR = "reranking_error"
    VALIDATION_ERROR = "validation_error"
    CACHE_ERROR = "cache_error"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(str, Enum):
    """Available recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK_MODEL = "fallback_model"
    FALLBACK_METHOD = "fallback_method"
    SKIP_STEP = "skip_step"
    USE_CACHE = "use_cache"
    REDUCE_COMPLEXITY = "reduce_complexity"
    MANUAL_INTERVENTION = "manual_intervention"
    FAIL_GRACEFULLY = "fail_gracefully"

class RAGError(BaseModel):
    """Structured error model for RAG system failures."""
    error_type: ErrorType = Field(..., description="Type of error that occurred")
    severity: ErrorSeverity = Field(..., description="Severity level of the error")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(None, description="Detailed error information")
    component: str = Field(..., description="Component where error occurred")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the error occurred")
    recovery_strategies: List[RecoveryStrategy] = Field(default_factory=list, description="Suggested recovery strategies")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context about the error")
    recoverable: bool = Field(default=True, description="Whether this error can be recovered from")
    
    @validator('recovery_strategies')
    def validate_recovery_strategies(cls, v, values):
        """Ensure recovery strategies are appropriate for error type."""
        if 'error_type' not in values:
            return v
        
        error_type = values['error_type']
        
        # Define appropriate strategies for each error type
        appropriate_strategies = {
            ErrorType.MODEL_ERROR: [RecoveryStrategy.FALLBACK_MODEL, RecoveryStrategy.RETRY],
            ErrorType.RETRIEVAL_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK_METHOD, RecoveryStrategy.USE_CACHE],
            ErrorType.EMBEDDING_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.USE_CACHE, RecoveryStrategy.FALLBACK_METHOD],
            ErrorType.RERANKING_ERROR: [RecoveryStrategy.FALLBACK_METHOD, RecoveryStrategy.SKIP_STEP],
            ErrorType.VALIDATION_ERROR: [RecoveryStrategy.REDUCE_COMPLEXITY, RecoveryStrategy.RETRY],
            ErrorType.CACHE_ERROR: [RecoveryStrategy.SKIP_STEP, RecoveryStrategy.RETRY],
            ErrorType.TIMEOUT_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.REDUCE_COMPLEXITY],
            ErrorType.NETWORK_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK_MODEL],
            ErrorType.CONFIGURATION_ERROR: [RecoveryStrategy.MANUAL_INTERVENTION, RecoveryStrategy.FAIL_GRACEFULLY],
            ErrorType.UNKNOWN_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.FAIL_GRACEFULLY]
        }
        
        # Filter strategies to only include appropriate ones
        if v:
            appropriate = appropriate_strategies.get(error_type, [])
            filtered_strategies = [s for s in v if s in appropriate]
            if not filtered_strategies and appropriate:
                # If no provided strategies are appropriate, use the first appropriate one
                filtered_strategies = [appropriate[0]]
            return filtered_strategies
        
        return v

class ErrorRecoveryResult(BaseModel):
    """Result of an error recovery attempt."""
    success: bool = Field(..., description="Whether recovery was successful")
    strategy_used: RecoveryStrategy = Field(..., description="Recovery strategy that was used")
    result: Optional[Any] = Field(None, description="Result if recovery was successful")
    new_error: Optional[RAGError] = Field(None, description="New error if recovery failed")
    recovery_time: float = Field(..., ge=0.0, description="Time taken for recovery attempt")
    attempts: int = Field(default=1, ge=1, description="Number of recovery attempts made")

class ErrorContext(BaseModel):
    """Context information for error handling."""
    operation: str = Field(..., description="Operation being performed when error occurred")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data that caused the error")
    system_state: Dict[str, Any] = Field(default_factory=dict, description="System state at time of error")
    user_context: Optional[str] = Field(None, description="User context or query that led to error")
    
def create_rag_error(
    error_type: ErrorType,
    message: str,
    component: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    details: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    recoverable: bool = True
) -> RAGError:
    """Factory function to create structured RAG errors with appropriate recovery strategies."""
    
    # Define default recovery strategies for each error type
    default_strategies = {
        ErrorType.MODEL_ERROR: [RecoveryStrategy.FALLBACK_MODEL, RecoveryStrategy.RETRY],
        ErrorType.RETRIEVAL_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK_METHOD],
        ErrorType.EMBEDDING_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.USE_CACHE],
        ErrorType.RERANKING_ERROR: [RecoveryStrategy.FALLBACK_METHOD, RecoveryStrategy.SKIP_STEP],
        ErrorType.VALIDATION_ERROR: [RecoveryStrategy.REDUCE_COMPLEXITY, RecoveryStrategy.RETRY],
        ErrorType.CACHE_ERROR: [RecoveryStrategy.SKIP_STEP],
        ErrorType.TIMEOUT_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.REDUCE_COMPLEXITY],
        ErrorType.NETWORK_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK_MODEL],
        ErrorType.CONFIGURATION_ERROR: [RecoveryStrategy.MANUAL_INTERVENTION],
        ErrorType.UNKNOWN_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.FAIL_GRACEFULLY]
    }
    
    recovery_strategies = default_strategies.get(error_type, [RecoveryStrategy.FAIL_GRACEFULLY])
    
    return RAGError(
        error_type=error_type,
        severity=severity,
        message=message,
        details=details,
        component=component,
        recovery_strategies=recovery_strategies,
        context=context or {},
        recoverable=recoverable
    )

async def handle_rag_error(
    error: Exception,
    context: ErrorContext,
    max_retries: int = 3
) -> ErrorRecoveryResult:
    """Handle RAG errors with structured recovery strategies."""
    import time
    start_time = time.time()
    
    # Convert exception to structured RAG error
    if isinstance(error, RetrievalError):
        rag_error = create_rag_error(
            ErrorType.RETRIEVAL_ERROR,
            str(error),
            context.operation,
            details=str(error.__cause__) if error.__cause__ else None,
            context={"operation": context.operation, "input": context.input_data}
        )
    elif isinstance(error, ReRankingError):
        rag_error = create_rag_error(
            ErrorType.RERANKING_ERROR,
            str(error),
            context.operation,
            details=str(error.__cause__) if error.__cause__ else None,
            context={"operation": context.operation}
        )
    elif "timeout" in str(error).lower():
        rag_error = create_rag_error(
            ErrorType.TIMEOUT_ERROR,
            str(error),
            context.operation,
            severity=ErrorSeverity.HIGH,
            context={"operation": context.operation}
        )
    elif "network" in str(error).lower() or "connection" in str(error).lower():
        rag_error = create_rag_error(
            ErrorType.NETWORK_ERROR,
            str(error),
            context.operation,
            severity=ErrorSeverity.HIGH,
            context={"operation": context.operation}
        )
    else:
        rag_error = create_rag_error(
            ErrorType.UNKNOWN_ERROR,
            str(error),
            context.operation,
            details=str(error.__cause__) if error.__cause__ else None,
            context={"operation": context.operation, "error_type": type(error).__name__}
        )
    
    print(f"🚨 RAG Error: {rag_error.error_type.value} in {rag_error.component}")
    print(f"   Message: {rag_error.message}")
    print(f"   Severity: {rag_error.severity.value}")
    print(f"   Recovery strategies: {[s.value for s in rag_error.recovery_strategies]}")
    
    # Attempt recovery based on available strategies
    for attempt in range(1, max_retries + 1):
        if not rag_error.recovery_strategies:
            break
        
        strategy = rag_error.recovery_strategies[0]  # Try first strategy
        print(f"🔄 Recovery attempt {attempt}/{max_retries} using strategy: {strategy.value}")
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                # Simple retry - re-raise the original error to be caught by caller
                recovery_time = time.time() - start_time
                return ErrorRecoveryResult(
                    success=False,
                    strategy_used=strategy,
                    new_error=rag_error,
                    recovery_time=recovery_time,
                    attempts=attempt
                )
            
            elif strategy == RecoveryStrategy.FALLBACK_METHOD:
                # This would be handled by the calling function
                recovery_time = time.time() - start_time
                return ErrorRecoveryResult(
                    success=False,
                    strategy_used=strategy,
                    new_error=rag_error,
                    recovery_time=recovery_time,
                    attempts=attempt
                )
            
            elif strategy == RecoveryStrategy.SKIP_STEP:
                # Return a default/empty result
                recovery_time = time.time() - start_time
                return ErrorRecoveryResult(
                    success=True,
                    strategy_used=strategy,
                    result=None,  # Indicates step was skipped
                    recovery_time=recovery_time,
                    attempts=attempt
                )
            
            elif strategy == RecoveryStrategy.FAIL_GRACEFULLY:
                # Log error and return graceful failure
                logfire.error("RAG operation failed gracefully", error=rag_error.dict())
                recovery_time = time.time() - start_time
                return ErrorRecoveryResult(
                    success=False,
                    strategy_used=strategy,
                    new_error=rag_error,
                    recovery_time=recovery_time,
                    attempts=attempt
                )
        
        except Exception as recovery_error:
            print(f"❌ Recovery strategy {strategy.value} failed: {recovery_error}")
            continue
    
    # All recovery attempts failed
    recovery_time = time.time() - start_time
    return ErrorRecoveryResult(
        success=False,
        strategy_used=rag_error.recovery_strategies[0] if rag_error.recovery_strategies else RecoveryStrategy.FAIL_GRACEFULLY,
        new_error=rag_error,
        recovery_time=recovery_time,
        attempts=max_retries
    )

# ===== MODEL FALLBACK LOGIC =====

class ModelProvider(str, Enum):
    """Available model providers."""
    GOOGLE = "google"
    OPENAI = "openai"

class ModelFallbackConfig(BaseModel):
    """Configuration for model fallback behavior."""
    enable_fallback: bool = Field(default=True, description="Enable automatic model fallback")
    primary_provider: ModelProvider = Field(default=ModelProvider.GOOGLE, description="Primary model provider")
    fallback_provider: ModelProvider = Field(default=ModelProvider.OPENAI, description="Fallback model provider")
    max_fallback_attempts: int = Field(default=2, ge=1, le=5, description="Maximum fallback attempts")
    fallback_timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0, description="Timeout before fallback")
    circuit_breaker_threshold: int = Field(default=3, ge=1, le=10, description="Failures before circuit breaker opens")
    circuit_breaker_reset_time: int = Field(default=300, ge=60, le=3600, description="Time before circuit breaker resets (seconds)")

class ModelFallbackState(BaseModel):
    """Current state of model fallback system."""
    current_provider: ModelProvider = Field(..., description="Currently active provider")
    primary_failures: int = Field(default=0, ge=0, description="Number of primary provider failures")
    fallback_failures: int = Field(default=0, ge=0, description="Number of fallback provider failures")
    circuit_breaker_open: bool = Field(default=False, description="Whether circuit breaker is open")
    last_failure_time: Optional[str] = Field(None, description="Timestamp of last failure")
    total_requests: int = Field(default=0, ge=0, description="Total requests processed")
    successful_requests: int = Field(default=0, ge=0, description="Successful requests")

class ModelFallbackResult(BaseModel):
    """Result of model execution with fallback information."""
    success: bool = Field(..., description="Whether the operation was successful")
    provider_used: ModelProvider = Field(..., description="Provider that was used")
    result: Optional[Any] = Field(None, description="Result if successful")
    error: Optional[RAGError] = Field(None, description="Error if failed")
    fallback_occurred: bool = Field(default=False, description="Whether fallback was used")
    attempts_made: int = Field(default=1, ge=1, description="Number of attempts made")
    total_time: float = Field(..., ge=0.0, description="Total execution time")

# Global fallback state
_model_fallback_state = ModelFallbackState(current_provider=ModelProvider.GOOGLE)

def get_model_fallback_state() -> ModelFallbackState:
    """Get current model fallback state."""
    return _model_fallback_state

def reset_model_fallback_state() -> None:
    """Reset model fallback state to initial values."""
    global _model_fallback_state
    _model_fallback_state = ModelFallbackState(current_provider=ModelProvider.GOOGLE)

async def execute_with_model_fallback(
    operation_func: callable,
    config: Optional[ModelFallbackConfig] = None,
    context: Optional[ErrorContext] = None
) -> ModelFallbackResult:
    """Execute an operation with automatic model fallback."""
    import time
    import asyncio
    
    start_time = time.time()
    global _model_fallback_state
    
    # Use default config if none provided
    if config is None:
        config = ModelFallbackConfig()
    
    if context is None:
        context = ErrorContext(operation="model_execution")
    
    _model_fallback_state.total_requests += 1
    
    # Check circuit breaker
    if _model_fallback_state.circuit_breaker_open and config.enable_fallback:
        if _model_fallback_state.last_failure_time:
            last_failure = datetime.fromisoformat(_model_fallback_state.last_failure_time)
            time_since_failure = (datetime.now() - last_failure).total_seconds()
            
            if time_since_failure > config.circuit_breaker_reset_time:
                print("🔄 Circuit breaker reset - attempting primary provider")
                _model_fallback_state.circuit_breaker_open = False
                _model_fallback_state.primary_failures = 0
                _model_fallback_state.current_provider = config.primary_provider
    
    providers_to_try = []
    
    if not _model_fallback_state.circuit_breaker_open:
        providers_to_try.append(config.primary_provider)
    
    if config.enable_fallback and config.fallback_provider != config.primary_provider:
        providers_to_try.append(config.fallback_provider)
    
    if not providers_to_try:
        # No providers available
        error = create_rag_error(
            ErrorType.CONFIGURATION_ERROR,
            "No model providers available",
            "model_fallback",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False
        )
        return ModelFallbackResult(
            success=False,
            provider_used=_model_fallback_state.current_provider,
            error=error,
            attempts_made=0,
            total_time=time.time() - start_time
        )
    
    last_error = None
    
    for attempt, provider in enumerate(providers_to_try, 1):
        try:
            print(f"🤖 Attempting {provider.value} model (attempt {attempt}/{len(providers_to_try)})")
            
            # Set current provider
            _model_fallback_state.current_provider = provider
            
            # Execute operation with timeout
            result = await asyncio.wait_for(
                operation_func(provider),
                timeout=config.fallback_timeout_seconds
            )
            
            # Success!
            _model_fallback_state.successful_requests += 1
            total_time = time.time() - start_time
            
            print(f"✅ {provider.value} model succeeded in {total_time:.2f}s")
            
            return ModelFallbackResult(
                success=True,
                provider_used=provider,
                result=result,
                fallback_occurred=(attempt > 1),
                attempts_made=attempt,
                total_time=total_time
            )
            
        except asyncio.TimeoutError:
            error_msg = f"{provider.value} model timeout after {config.fallback_timeout_seconds}s"
            last_error = create_rag_error(
                ErrorType.TIMEOUT_ERROR,
                error_msg,
                "model_execution",
                severity=ErrorSeverity.HIGH,
                context={"provider": provider.value, "timeout": config.fallback_timeout_seconds}
            )
            print(f"⏰ {error_msg}")
            
        except Exception as e:
            error_msg = f"{provider.value} model failed: {str(e)}"
            last_error = create_rag_error(
                ErrorType.MODEL_ERROR,
                error_msg,
                "model_execution",
                details=str(e),
                context={"provider": provider.value, "error_type": type(e).__name__}
            )
            print(f"❌ {error_msg}")
        
        # Update failure counts
        if provider == config.primary_provider:
            _model_fallback_state.primary_failures += 1
            
            # Check if circuit breaker should open
            if _model_fallback_state.primary_failures >= config.circuit_breaker_threshold:
                _model_fallback_state.circuit_breaker_open = True
                _model_fallback_state.last_failure_time = datetime.now().isoformat()
                print(f"🚨 Circuit breaker opened after {_model_fallback_state.primary_failures} failures")
        else:
            _model_fallback_state.fallback_failures += 1
    
    # All providers failed
    total_time = time.time() - start_time
    print(f"💥 All model providers failed after {len(providers_to_try)} attempts")
    
    return ModelFallbackResult(
        success=False,
        provider_used=_model_fallback_state.current_provider,
        error=last_error,
        fallback_occurred=(len(providers_to_try) > 1),
        attempts_made=len(providers_to_try),
        total_time=total_time
    )

def configure_pydantic_ai_cache(config: PydanticAICacheConfig) -> None:
    """Configure Pydantic AI cache integration with existing cache systems."""
    global llm_cache, embedding_cache
    
    if config.enable_llm_cache:
        llm_cache = LLMResponseCache(
            ttl_hours=config.llm_cache_ttl_hours,
            max_size=config.llm_cache_max_size
        )
        print(f"✅ LLM Cache configured: TTL={config.llm_cache_ttl_hours}h, Max={config.llm_cache_max_size}")
    
    if config.enable_embedding_cache:
        embedding_cache = QueryEmbeddingCache(max_size=config.embedding_cache_max_size)
        print(f"✅ Embedding Cache configured: Max={config.embedding_cache_max_size}")
    
    print(f"🎯 Cache hit logging: {'enabled' if config.cache_hit_logging else 'disabled'}")

def get_cache_metrics() -> CacheMetrics:
    """Get comprehensive cache performance metrics."""
    llm_stats = llm_cache.get_stats()
    embedding_stats = embedding_cache.get_stats()
    
    total_hits = llm_stats.get("hit_count", 0) + embedding_stats.get("hit_count", 0)
    total_misses = llm_stats.get("miss_count", 0) + embedding_stats.get("miss_count", 0)
    total_requests = total_hits + total_misses
    
    hit_rate = (total_hits / total_requests) if total_requests > 0 else 0.0
    
    return CacheMetrics(
        llm_cache_stats=llm_stats,
        embedding_cache_stats=embedding_stats,
        total_cache_hits=total_hits,
        total_cache_misses=total_misses,
        cache_hit_rate=hit_rate
    )


# Removed generate_with_gemini function - now using native Pydantic AI GoogleModel integration

# Custom Exceptions
class RetrievalError(Exception):
    """Custom exception for errors during document retrieval."""
    pass

class ReRankingError(Exception):
    """Custom exception for errors during re-ranking."""
    pass

# Define Dependencies with Cache Integration
@dataclass
class RAGDeps:
    """Enhanced dependencies for the RAG agent with cache integration."""
    chroma_client: chromadb.Client
    collection_name: str
    embedding_model_name: str
    embedding_provider: str
    vertex_project_id: Optional[str] = None
    vertex_location: Optional[str] = None
    # Vertex AI Reranker fields
    use_vertex_reranker: bool = False
    vertex_reranker_model: Optional[str] = None
    # Cache configuration
    cache_config: Optional[PydanticAICacheConfig] = None
    # Batch processing configuration
    batch_config: Optional[BatchProcessingConfig] = None
    
    def __post_init__(self):
        """Initialize cache configuration if provided."""
        if self.cache_config:
            configure_pydantic_ai_cache(self.cache_config)

# Define the system prompt as a separate variable
SYSTEM_PROMPT_TEMPLATE = (
    "Du bist ein **präziser Faktenanalyst**. Deine Kernkompetenz ist es, aus einem gegebenen Kontext die maximale Menge an **spezifischen, überprüfbaren Details** zu extrahieren und diese in einer klaren, gut strukturierten Form zu präsentieren. Deine Antwort muss **ausschließlich auf dem bereitgestellten Kontext basieren**.\n\n"

    "**Aufgabe:** Beantworte die folgende Anfrage als detaillierte und faktenbasierte Analyse.\n\n"
    
    "**Anfrage:** {question}\n\n"
    
    "**Anforderungen an deine Antwort (Qualitätsmerkmale):**\n"
    "**1. Priorität: Maximale Faktendichte und Granularität:**\n"
    "    - **Dein primäres Ziel ist es, allgemeine Aussagen zu vermeiden.** Ersetze jede allgemeine Formulierung durch die spezifischsten Informationen, die der Kontext bietet.\n"
    "    - **Integriere proaktiv und mit hoher Dichte die folgenden Detailtypen:**\n"
    "        - **Quantitative Daten:** Konkrete Zahlen, Statistiken, Geldbeträge, Höhenangaben, Prozentwerte.\n"
    "        - **Namen und Eigennamen:** Spezifische Personen, Organisationen, Gesetze, Programme, Technologien.\n"
    "        - **Chronologische Marker:** Genaue Jahreszahlen oder Daten für Ereignisse und Meilensteine.\n"
    "        - **Technische/Kausale Details:** Kurze Erklärungen für das 'Warum', 'Was', 'Weshalb', 'Wieso' oder 'Wie', falls im Kontext vorhanden.\n"
    "    - **Leitprinzip:** Lieber eine spezifische, aber enger gefasste Antwort, die vor Details strotzt, als eine breite, aber oberflächliche Antwort.\n"

    "**2. Umfassende thematische Struktur (als Rahmen für die Fakten):**\n"
    "    - Ordne die extrahierten Fakten in eine logische und umfassende Struktur ein. Beleuchte dabei, falls im Kontext enthalten, Aspekte wie historische Entwicklung, aktuelle Situation, zukünftige Konzepte und Auswirkungen (positiv/negativ).\n"
    "    - Diese Struktur dient als Gerüst, das mit den unter Punkt 1 geforderten, dichten Fakten gefüllt wird.\n"
    
    "**3. Exzellente Lesbarkeit:**\n"
    "    - Gliedere deine Antwort mit klaren, aussagekräftigen Überschriften (z.B. `## Hauptthema`, `### Unterpunkt`).\n"
    "    - Beginne mit einer kurzen Einleitung, die den Rahmen setzt, und schließe mit einem prägnanten Fazit, das die wichtigsten Fakten zusammenfasst.\n"
    "    - Nutze Aufzählungen (`• Liste Item`) oder nummerierte Listen (`1. Liste Item`) zur übersichtlichen Darstellung von konkreten Fakten.\n"
    
    "**4. Strikte Sachlichkeit und Quellenintegration:**\n"
    "    - Deine Antwort MUSS zu 100% auf den Informationen im `--- KONTEXT ---` basieren. Erfinde oder schlussfolgere keine Informationen, die nicht explizit genannt werden.\n"
    "    - Belege jede Information direkt im Text mit klickbaren Quellenverweisen, auch von sprungmarken innerhalb einer Website im Format `[Nummer](URL)`. Die Referenznummern (z.B. `[Quelle 1]`) und die zugehörigen URLs findest du im Kontext. Wandle 'Quelle X' in die reine Zahl 'X' um.\n"
    "    - Führe am Ende unter der Überschrift `## Quellen` nur die tatsächlich zitierten Quellen und ihre URLs auf.\n"
    "    - Wenn der Kontext keine ausreichenden Informationen zur Beantwortung der Frage zulässt, antworte klar: 'Basierend auf den verfügbaren Informationen kann ich diese Frage nicht vollständig beantworten.' oder 'Der bereitgestellte Kontext enthält keine Informationen zu dieser Frage.'\n\n"

    "--- KONTEXT ---\n"
    "{context}\n"
    "--- END KONTEXT ---\n"
)

# Configure primary and fallback models
def get_primary_model():
    """Get the primary model - GoogleModel with proper provider setup if available, otherwise OpenAI."""
    try:
        # Check if GoogleModel is available
        if not GOOGLE_MODEL_AVAILABLE:
            print("GoogleModel not available - using OpenAI as primary")
            return "gpt-4.1-mini"
        
        # Check for the correct API key (GOOGLE_API_KEY, not GEMINI_API_KEY)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")  # Legacy support
        
        api_key = google_api_key or gemini_api_key
        
        if not api_key:
            print("Warning: Neither GOOGLE_API_KEY nor GEMINI_API_KEY found. Using OpenAI as primary.")
            print("💡 For GoogleModel, set: export GOOGLE_API_KEY='your-api-key'")
            return "gpt-4.1-mini"
        
        # Create GoogleProvider with the API key
        if google_api_key:
            print("✅ Using GOOGLE_API_KEY for GoogleModel")
            provider = GoogleProvider(api_key=google_api_key)
        else:
            print("⚠️ Using legacy GEMINI_API_KEY - consider switching to GOOGLE_API_KEY")
            provider = GoogleProvider(api_key=gemini_api_key)
        
        # Return GoogleModel with proper provider
        model = GoogleModel("gemini-1.5-flash", provider=provider)
        print("✅ GoogleModel initialized successfully with provider")
        return model
        
    except Exception as e:
        print(f"Error initializing GoogleModel: {e}. Using OpenAI as primary.")
        print("💡 Make sure you have installed: pip install 'pydantic-ai-slim[google]'")
        return "gpt-4.1-mini"

def get_fallback_model():
    """Get the fallback OpenAI model."""
    return "gpt-4.1-mini"

def check_google_model_setup():
    """Check and provide guidance for GoogleModel setup."""
    print("\n🔍 GoogleModel Setup Check:")
    
    # Check if GoogleModel is available
    if not GOOGLE_MODEL_AVAILABLE:
        print("❌ GoogleModel not available")
        print("💡 To install: pip install 'pydantic-ai-slim[google]' or pip install pydantic-ai")
        return False
    
    # Check API keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if google_api_key:
        print("✅ GOOGLE_API_KEY found")
        return True
    elif gemini_api_key:
        print("⚠️ GEMINI_API_KEY found (legacy) - consider switching to GOOGLE_API_KEY")
        return True
    else:
        print("❌ No Google API key found")
        print("💡 Set your API key: export GOOGLE_API_KEY='your-api-key'")
        print("💡 Get API key from: https://aistudio.google.com")
        return False

async def check_openai_api_setup():
    """Check OpenAI API key and connection."""
    print("\n🔍 OpenAI API Setup Check:")
    
    # Check if API key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        print("💡 Set your API key: export OPENAI_API_KEY='your-api-key'")
        print("💡 Get API key from: https://platform.openai.com/api-keys")
        return False
    
    print(f"✅ OPENAI_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API connection
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        
        print("🔄 Testing API connection...")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello! This is a test."}],
            max_tokens=20
        )
        
        print("✅ OpenAI API working!")
        print(f"✅ Test response: {response.choices[0].message.content}")
        print(f"✅ Model used: {response.model}")
        print(f"✅ Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        
        # Specific error handling
        if "401" in str(e):
            print("💡 Invalid API key - check your OPENAI_API_KEY")
        elif "429" in str(e):
            print("💡 Rate limit exceeded - wait a moment and try again")
        elif "insufficient_quota" in str(e):
            print("💡 No credits left - add credits to your OpenAI account")
        else:
            print("💡 Check your internet connection and API key")
        
        return False

def check_all_api_keys():
    """Check all API keys and services."""
    print("🔍 Complete API Setup Check:")
    print("=" * 50)
    
    # Check OpenAI
    import asyncio
    openai_ok = asyncio.run(check_openai_api_setup())
    
    # Check Google (if available)
    google_ok = check_google_model_setup()
    
    # Summary
    print("\n📊 Summary:")
    print(f"OpenAI API: {'✅ Working' if openai_ok else '❌ Failed'}")
    print(f"Google API: {'✅ Working' if google_ok else '❌ Not available'}")
    
    if openai_ok:
        print("\n🎉 Your system is ready to use!")
    else:
        print("\n⚠️ Fix the API issues above before using the system")
    
    return openai_ok, google_ok

# Define the Pydantic AI Agent with GoogleModel integration
agent = Agent(
    name="StructuredRAGAgent",
    description="Advanced RAG system with structured outputs and multi-query retrieval using Pydantic AI.",
    system_prompt=SYSTEM_PROMPT_TEMPLATE,
    dependencies=RAGDeps,
    model=get_primary_model()
)

# Create fallback agent for when primary model fails
fallback_agent = Agent(
    name="FallbackRAGAgent", 
    description="Fallback RAG agent using OpenAI when GoogleModel fails.",
    system_prompt=SYSTEM_PROMPT_TEMPLATE,
    dependencies=RAGDeps,
    model=get_fallback_model()
)

# ===== PYDANTIC AI TOOLS =====

def analyze_question_complexity_structured(question: str) -> Tuple[QueryStrategy, float]:
    """Analyze question complexity and return structured result."""
    word_count = len(question.split())
    question_lower = question.lower()
    
    # Simple question indicators
    simple_patterns = ['was ist', 'was bedeutet', 'wer ist', 'wo ist', 'wann ist']
    definition_words = ['definition', 'bedeutung', 'erklärung']
    
    # Complex question indicators
    complex_words = ['unterschied', 'vergleich', 'vs', 'versus', 'warum', 'wie funktioniert', 'welche arten']
    has_multiple_concepts = len([w for w in question.split() if len(w) > 6]) > 2
    has_comparison = any(word in question_lower for word in ['vs', 'versus', 'unterschied', 'vergleich'])
    has_complex_words = any(word in question_lower for word in complex_words)
    
    # Classification logic with scoring
    if word_count <= 4 and any(pattern in question_lower for pattern in simple_patterns):
        return QueryStrategy.SIMPLE, 0.2
    elif any(word in question_lower for word in definition_words) and word_count <= 6:
        return QueryStrategy.SIMPLE, 0.25
    elif has_comparison or has_multiple_concepts or has_complex_words or word_count > 10:
        complexity_score = min(0.8 + (word_count - 10) * 0.02, 1.0)
        return QueryStrategy.COMPLEX, complexity_score
    else:
        complexity_score = 0.4 + (word_count - 5) * 0.05
        return QueryStrategy.MODERATE, min(complexity_score, 0.65)

@agent.tool
async def generate_query_variations_tool(
    ctx: RunContext[RAGDeps], 
    original_query: str
) -> QueryVariations:
    """Generate adaptive query variations based on question complexity with structured output."""
    print(f"🧠 Generating query variations for: '{original_query}'")
    
    # Analyze question complexity
    strategy, complexity_score = analyze_question_complexity_structured(original_query)
    print(f"---> Question complexity: {strategy.value} (score: {complexity_score:.2f})")
    
    # Create base QueryVariations object
    query_variations = QueryVariations(
        original_query=original_query,
        strategy=strategy,
        complexity_score=complexity_score,
        variations=[]
    )
    
    # Adaptive variation generation
    if strategy == QueryStrategy.SIMPLE:
        variation_prompt = f"""Generate 1 alternative way to ask this question using different words and synonyms for better search coverage.
        
        Original question: {original_query}
        
        Focus on using alternative terminology while keeping the same meaning.
        Return only the 1 variation without numbering or explanation."""
        max_variations = 1
    elif strategy == QueryStrategy.MODERATE:
        variation_prompt = f"""Generate 2 different ways to ask the same question using varied vocabulary for comprehensive search:
        1. One using synonyms and alternative terms
        2. One using more specific or technical language
        
        Original question: {original_query}
        
        Return only the 2 variations, one per line, without numbering or explanation."""
        max_variations = 2
    else:  # COMPLEX
        variation_prompt = f"""Generate 3 comprehensive variations of this question for maximum search coverage:
        1. One using synonyms and alternative terminology
        2. One using more specific/technical terms  
        3. One that rephrases the core concepts differently
        
        Original question: {original_query}
        
        Return only the 3 variations, one per line, without numbering or explanation."""
        max_variations = 3
    
    try:
        # Use parallel processing for better performance
        variations_text = await run_agent_with_fallback(
            prompt=variation_prompt,
            deps=ctx.deps,
            system_prompt_override="You generate query variations for better search coverage. Create semantically different but equivalent questions. Keep variations focused and avoid expanding the scope."
        )
        
        # Parse variations
        new_variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
        query_variations.variations = new_variations[:max_variations]
        
        print(f"---> Generated {len(query_variations.variations)} variations: {query_variations.variations}")
        
    except Exception as e:
        print(f"Error generating query variations: {e}. Using original query only.")
        # Keep empty variations list - validation will handle this gracefully
    
    return query_variations

@agent.tool
async def retrieve_documents_structured(
    ctx: RunContext[RAGDeps], 
    query_variations: QueryVariations,
    n_results: int = 25
) -> RetrievalResult:
    """Structured retrieval tool with multi-query support and comprehensive metadata."""
    import time
    start_time = time.time()
    
    print(f"--- Structured Retrieval Tool Called ---")
    print(f"Original Query: '{query_variations.original_query}'")
    print(f"Strategy: {query_variations.strategy.value}")
    print(f"Variations: {len(query_variations.variations)} additional queries")
    
    # Prepare all queries (original + variations)
    all_queries = [query_variations.original_query] + query_variations.variations
    print(f"---> Processing {len(all_queries)} total queries")
    
    # --- Parallel HyDE Generation for all queries ---
    async def generate_hyde_for_query(query: str) -> str:
        """Generate hypothetical answer for a single query."""
        hyde_prompt = f"Generate a detailed, plausible paragraph that directly answers the following question as if it were extracted from a relevant document or webpage. Use varied terminology and synonyms that might appear in different sources. Include both formal and informal ways of expressing the same concepts. Question: {query}"
        
        try:
            hypothetical_answer = await run_agent_with_fallback(
                prompt=hyde_prompt,
                deps=ctx.deps,
                system_prompt_override="You generate hypothetical answers for RAG retrieval. Create realistic content that could be found in documentation, articles, or informational websites. Be factual and relevant to the specific question asked."
            )
            return hypothetical_answer
        except Exception as e:
            print(f"Error generating HyDE for '{query}': {e}")
            return query  # Fallback to original query
    
    # Generate HyDE answers in parallel
    print("🧠 Generating hypothetical answers for all queries...")
    hyde_answers = await asyncio.gather(*[generate_hyde_for_query(q) for q in all_queries])
    
    for i, (query, hyde) in enumerate(zip(all_queries, hyde_answers)):
        print(f"---> HyDE {i+1}: '{query[:30]}...' -> '{hyde[:50]}...'")
    
    # --- Parallel Retrieval for all HyDE answers ---
    initial_n_results = max(50, n_results * 3)
    print(f"🔍 Retrieving {initial_n_results} candidates per query...")
    
    try:
        collection = get_or_create_collection(
            client=ctx.deps.chroma_client,
            collection_name=ctx.deps.collection_name
        )
        
        # Get total document count
        total_documents = collection.count()
        
        # Detect embedding method - mit besserer Fehlerbehandlung
        collection_embedding_dim = None
        try:
            sample = collection.get(limit=1, include=["embeddings"])
            if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
                collection_embedding_dim = len(sample["embeddings"][0])
                print(f"---> Detected embedding dimension: {collection_embedding_dim}")
            else:
                print("---> No embeddings found in collection, using text-based search")
        except Exception as e:
            print(f"---> Embedding detection failed: {e}, falling back to text search")
        
        # Parallel retrieval for all HyDE answers
        async def retrieve_for_hyde(hyde_answer: str) -> Tuple[List[str], List[Dict]]:
            """Retrieve documents for a single HyDE answer."""
            if collection_embedding_dim == 384:
                # ChromaDB default embeddings
                results = collection.query(
                    query_texts=[hyde_answer],
                    n_results=initial_n_results,
                    include=['metadatas', 'documents']
                )
            else:
                # Vertex AI embeddings with cache-aware processing
                query_embedding = None
                cache_enabled = (
                    ctx.deps.cache_config is None or 
                    ctx.deps.cache_config.enable_embedding_cache
                )
                
                if cache_enabled:
                    query_embedding = embedding_cache.get(hyde_answer)
                    if query_embedding and ctx.deps.cache_config and ctx.deps.cache_config.cache_hit_logging:
                        print(f"🎯 Embedding cache HIT for: {hyde_answer[:30]}...")
                
                if query_embedding is None:
                    # Generate consistent embedding with proper task_type
                    query_embedding = get_vertex_text_embedding(
                        text=hyde_answer,
                        model_name=ctx.deps.embedding_model_name,
                        task_type="RETRIEVAL_QUERY",  # Consistent task type for queries
                        project_id=ctx.deps.vertex_project_id,
                        location=ctx.deps.vertex_location
                    )
                    if query_embedding and cache_enabled:
                        # Normalize embedding before caching (L2 normalization)
                        import numpy as np
                        query_vec = np.array(query_embedding)
                        query_embedding = (query_vec / np.linalg.norm(query_vec)).tolist()
                        embedding_cache.store(hyde_answer, query_embedding)
                        if ctx.deps.cache_config and ctx.deps.cache_config.cache_hit_logging:
                            print(f"💾 Normalized embedding cached for: {hyde_answer[:30]}...")
                
                if query_embedding is None:
                    # Fallback auf Text-basierte Suche wenn Embedding fehlschlägt
                    print(f"⚠️ Embedding failed, using text-based fallback for: {hyde_answer[:30]}...")
                    results = collection.query(
                        query_texts=[hyde_answer],
                        n_results=initial_n_results,
                        include=['metadatas', 'documents']
                    )
                else:
                    # HYBRID RETRIEVAL: Combine semantic + text search for better recall
                    print(f"🔍 Using hybrid retrieval (semantic + text) for: {hyde_answer[:30]}...")
                    
                    # Semantic search with normalized embedding
                    semantic_results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=initial_n_results // 2,  # Half from semantic
                        include=['metadatas', 'documents', 'distances']
                    )
                    
                    # Text-based search for keyword matches (BM25-like)
                    text_results = collection.query(
                        query_texts=[hyde_answer],
                        n_results=initial_n_results // 2,  # Half from text
                        include=['metadatas', 'documents']
                    )
                    
                    # Combine results (simple merge - could be improved with score fusion)
                    if semantic_results['documents'][0] and text_results['documents'][0]:
                        combined_docs = semantic_results['documents'][0] + text_results['documents'][0]
                        combined_metas = semantic_results['metadatas'][0] + text_results['metadatas'][0]
                        
                        # Simple deduplication
                        seen_docs = set()
                        final_docs = []
                        final_metas = []
                        
                        for doc, meta in zip(combined_docs, combined_metas):
                            # Improved deduplication: full content hash + length
                            import hashlib
                            content_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()
                            content_length = len(doc)
                            dedup_key = f"{content_hash}_{content_length}"
                            
                            if dedup_key not in seen_docs:
                                seen_docs.add(dedup_key)
                                final_docs.append(doc)
                                final_metas.append(meta)
                        
                        results = {
                            'documents': [final_docs[:initial_n_results]],
                            'metadatas': [final_metas[:initial_n_results]]
                        }
                        print(f"---> Hybrid retrieval: {len(final_docs)} unique results")
                    else:
                        # Fallback to whichever worked
                        results = semantic_results if semantic_results['documents'][0] else text_results
            
            if results and results.get('documents') and results['documents'][0]:
                return results['documents'][0], results['metadatas'][0]
            return [], []
        
        # Retrieve for all HyDE answers in parallel
        retrieval_results = await asyncio.gather(*[retrieve_for_hyde(hyde) for hyde in hyde_answers])
        
        # Zusätzlicher Text-Fallback wenn keine Ergebnisse
        total_results = sum(len(docs) for docs, _ in retrieval_results)
        if total_results == 0:
            print("🚨 Keine Embedding-Ergebnisse - versuche direkten Text-Fallback...")
            try:
                # Direkte Text-Suche mit Original-Query
                text_results = collection.query(
                    query_texts=[query_variations.original_query],
                    n_results=initial_n_results,
                    include=['metadatas', 'documents']
                )
                if text_results and text_results.get('documents') and text_results['documents'][0]:
                    retrieval_results.append((text_results['documents'][0], text_results['metadatas'][0]))
                    print(f"---> Text-Fallback fand {len(text_results['documents'][0])} zusätzliche Dokumente")
            except Exception as e:
                print(f"---> Text-Fallback fehlgeschlagen: {e}")
        
        # Combine and deduplicate results
        all_docs = []
        all_metadatas = []
        seen_docs = set()
        
        for docs, metadatas in retrieval_results:
            for doc, metadata in zip(docs, metadatas):
                # Improved deduplication: full content hash + length check
                import hashlib
                content_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()
                content_length = len(doc)
                dedup_key = f"{content_hash}_{content_length}"
                
                if dedup_key not in seen_docs:
                    seen_docs.add(dedup_key)
                    all_docs.append(doc)
                    all_metadatas.append(metadata)
        
        print(f"---> Combined results: {len(all_docs)} unique documents")
        
    except Exception as e:
        raise RetrievalError(f"Failed to retrieve documents: {e}") from e
    
    if not all_docs:
        print("🚨 WARNING: No documents retrieved from ChromaDB!")
        print(f"---> Collection has {total_documents} total documents")
        print(f"---> Queries used: {[q[:50] + '...' for q in all_queries]}")
        print(f"---> HyDE answers: {[h[:50] + '...' for h in hyde_answers]}")
        
        # Return empty result
        empty_ranked_docs = RankedDocuments(
            documents=[],
            total_candidates=0,
            ranking_method="none",
            query=query_variations.original_query
        )
        
        retrieval_time = time.time() - start_time
        return RetrievalResult(
            query_variations=query_variations,
            ranked_documents=empty_ranked_docs,
            retrieval_time=retrieval_time,
            embedding_cache_hits=0,
            embedding_cache_misses=len(hyde_answers)
        )
    
    # --- Create structured document chunks ---
    document_chunks = []
    for i, (doc_text, metadata) in enumerate(zip(all_docs, all_metadatas)):
        try:
            doc_metadata = DocumentMetadata(
                url=metadata.get('url', 'unknown'),
                title=metadata.get('title'),
                chunk_index=metadata.get('chunk_index', i),
                total_chunks=metadata.get('total_chunks'),
                timestamp=metadata.get('timestamp')
            )
            
            chunk = DocumentChunk(
                content=doc_text,
                metadata=doc_metadata,
                chunk_id=f"chunk_{i}_{hash(doc_text[:50])}"
            )
            document_chunks.append(chunk)
            
        except Exception as e:
            print(f"Error creating document chunk {i}: {e}")
            continue
    
    # --- HIGH RECALL: Minimal filtering with entropy check ---
    print("🎯 Applying high-recall minimal filtering with entropy check...")
    
    def calculate_content_entropy(text: str) -> float:
        """Calculate content entropy to filter out boilerplate."""
        import re
        from collections import Counter
        
        # Normalize text: remove special chars, lowercase
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        words = normalized.split()
        
        if len(words) < 3:
            return 0.0
        
        # Calculate word frequency entropy
        word_counts = Counter(words)
        total_words = len(words)
        entropy = 0.0
        
        for count in word_counts.values():
            prob = count / total_words
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def is_boilerplate_content(text: str) -> bool:
        """Detect boilerplate content (cookie banners, footers, etc.)."""
        text_lower = text.lower().strip()
        
        # Common boilerplate patterns
        boilerplate_patterns = [
            r'©\s*\d{4}',  # Copyright notices
            r'cookie.*accept',  # Cookie banners
            r'privacy.*policy',  # Privacy links
            r'terms.*service',  # Terms links
            r'all rights reserved',  # Copyright text
            r'powered by',  # Powered by notices
            r'follow us on',  # Social media
        ]
        
        boilerplate_count = sum(1 for pattern in boilerplate_patterns 
                               if re.search(pattern, text_lower))
        
        # High stop-word ratio indicates boilerplate
        stop_words = {'der', 'die', 'das', 'und', 'oder', 'aber', 'ist', 'sind', 
                     'was', 'wie', 'wo', 'wann', 'warum', 'mit', 'von', 'zu', 'für'}
        words = text_lower.split()
        if len(words) > 0:
            stop_word_ratio = len([w for w in words if w in stop_words]) / len(words)
        else:
            stop_word_ratio = 0
        
        return boilerplate_count >= 2 or stop_word_ratio > 0.7
    
    # Enhanced filtering with entropy and boilerplate detection
    filtered_chunks = []
    seen_content_hashes = set()
    
    for chunk in document_chunks:
        content = chunk.content.strip()
        
        # Improved deduplication: full content hash + length check
        import hashlib
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        content_length = len(content)
        dedup_key = f"{content_hash}_{content_length}"
        
        if dedup_key in seen_content_hashes:
            continue
        seen_content_hashes.add(dedup_key)
        
        # Enhanced micro-chunk filtering
        if len(content) < 30:  # Increased from 20 to 30 chars
            continue
        
        # Entropy-based quality filter
        entropy = calculate_content_entropy(content)
        if entropy < 1.0:  # Very low entropy = repetitive content
            continue
        
        # Boilerplate detection
        if is_boilerplate_content(content):
            continue
            
        # Assign minimal score for sorting - actual ranking happens in re-ranking step
        basic_score = 0.5  # Neutral score, re-ranker will determine actual relevance
        filtered_chunks.append((chunk, basic_score))
    
    print(f"---> High-recall filtering: {len(filtered_chunks)} chunks (removed duplicates, micro-chunks, boilerplate)")
    
    # Progressive K-Value Strategy: Start conservative, expand if needed
    def calculate_progressive_k(n_results: int, available_chunks: int, query_complexity: str) -> int:
        """Calculate optimal candidate pool size based on context."""
        base_multiplier = 3  # Start conservative
        
        # Adjust based on query complexity
        complexity_multipliers = {
            "simple": 3,
            "moderate": 5, 
            "complex": 8
        }
        
        multiplier = complexity_multipliers.get(query_complexity, 5)
        initial_k = max(50, n_results * multiplier)
        
        # Cap at available chunks and reasonable maximum
        max_reasonable_k = min(200, available_chunks)
        return min(initial_k, max_reasonable_k)
    
    # Determine query complexity from variations
    query_complexity = query_variations.strategy.value if hasattr(query_variations, 'strategy') else "moderate"
    candidate_count = calculate_progressive_k(n_results, len(filtered_chunks), query_complexity)
    
    candidate_chunks = filtered_chunks[:candidate_count]
    
    print(f"---> Progressive K-strategy: {len(candidate_chunks)} candidates (complexity: {query_complexity}, target: {candidate_count})")
    
    # --- Create ranked documents from candidate pool ---
    ranked_documents = []
    for rank, (chunk, score) in enumerate(candidate_chunks, 1):
        ranked_doc = RankedDocument(
            document=chunk,
            score=score,
            rank=rank
        )
        ranked_documents.append(ranked_doc)
    
    # Calculate confidence metrics for hallucination prevention
    avg_relevance = sum(d.score for d in ranked_documents) / len(ranked_documents) if ranked_documents else 0.0
    confidence_threshold = 0.25
    
    ranking_method = "high_recall_candidate_generation"
    if avg_relevance < confidence_threshold:
        ranking_method += "_low_confidence"
        print(f"⚠️ Low relevance pool detected (avg: {avg_relevance:.3f} < {confidence_threshold})")
        print("   → Consider refining search terms or expanding document collection")
    
    ranked_docs_collection = RankedDocuments(
        documents=ranked_documents,
        total_candidates=len(all_docs),
        ranking_method=ranking_method,
        query=query_variations.original_query
    )
    
    # --- Calculate cache statistics ---
    cache_stats = embedding_cache.get_stats()
    cache_hits = cache_stats.get("hit_count", 0)
    cache_misses = cache_stats.get("miss_count", 0)
    
    # --- Create final result ---
    retrieval_time = time.time() - start_time
    
    result = RetrievalResult(
        query_variations=query_variations,
        ranked_documents=ranked_docs_collection,
        retrieval_time=retrieval_time,
        embedding_cache_hits=cache_hits,
        embedding_cache_misses=cache_misses
    )
    
    print(f"✅ Structured retrieval completed in {retrieval_time:.2f}s")
    print(f"---> Retrieved {len(ranked_documents)} documents with avg score {sum(d.score for d in ranked_documents)/len(ranked_documents):.3f}")
    
    return result

@agent.tool
async def rerank_documents_tool(
    ctx: RunContext[RAGDeps],
    retrieval_result: RetrievalResult,
    top_n: int = 10
) -> RankedDocuments:
    """Re-rank documents using Vertex AI or fallback to score-based ranking."""
    print(f"--- Re-Ranking Tool Called ---")
    print(f"Input documents: {len(retrieval_result.ranked_documents.documents)}")
    print(f"Target top_n: {top_n}")
    
    if not retrieval_result.ranked_documents.documents:
        print("---> No documents to re-rank")
        return retrieval_result.ranked_documents
    
    # Check if Vertex AI re-ranking is enabled and available
    use_vertex_reranker = (
        ctx.deps.use_vertex_reranker and 
        ctx.deps.vertex_reranker_model and
        ctx.deps.vertex_project_id
    )
    
    if use_vertex_reranker:
        try:
            print(f"🔄 Using Vertex AI re-ranking with model: {ctx.deps.vertex_reranker_model}")
            
            # Prepare documents for Vertex AI API
            documents_for_api = []
            for ranked_doc in retrieval_result.ranked_documents.documents:
                documents_for_api.append({
                    'document': ranked_doc.document.content,
                    'metadata': {
                        'url': ranked_doc.document.metadata.url,
                        'title': ranked_doc.document.metadata.title,
                        'chunk_index': ranked_doc.document.metadata.chunk_index,
                        'original_score': ranked_doc.score
                    }
                })
            
            # Create records for the API call
            req_documents = [
                RankingRecord(id=str(i), content=doc['document']) 
                for i, doc in enumerate(documents_for_api)
            ]

            request = RankRequest(
                ranking_config=ctx.deps.vertex_reranker_model, 
                query=retrieval_result.query_variations.original_query,
                records=req_documents,
                top_n=min(top_n, len(documents_for_api)),
                ignore_record_details_in_response=True,
            )
            
            # Execute Vertex AI re-ranking
            async with RankServiceAsyncClient() as client:
                response = await client.rank(request=request)
            
            # Create new ranked documents with Vertex AI scores
            reranked_documents = []
            for rank, record in enumerate(response.records, 1):
                original_index = int(record.id)
                original_ranked_doc = retrieval_result.ranked_documents.documents[original_index]
                
                # Create new RankedDocument with updated score and rank
                reranked_doc = RankedDocument(
                    document=original_ranked_doc.document,
                    score=float(record.score),
                    rank=rank
                )
                reranked_documents.append(reranked_doc)
            
            # Create new RankedDocuments collection
            reranked_collection = RankedDocuments(
                documents=reranked_documents,
                total_candidates=retrieval_result.ranked_documents.total_candidates,
                ranking_method="vertex_ai_reranker",
                query=retrieval_result.query_variations.original_query
            )
            
            print(f"✅ Vertex AI re-ranking completed: {len(reranked_documents)} documents")
            for i, doc in enumerate(reranked_documents[:3]):
                print(f"  Rank {doc.rank}: Score {doc.score:.4f}, URL: {doc.document.metadata.url}")
            
            return reranked_collection
            
        except Exception as e:
            print(f"❌ Vertex AI re-ranking failed: {e}")
            print("---> Falling back to score-based ranking")
            logfire.exception("Vertex AI re-ranking failed", exception=e)
            # Fall through to score-based ranking
    
    # Fallback: Domain-agnostic semantic similarity ranking (no token-overlap heuristics)
    print("🎯 Using domain-agnostic semantic similarity ranking")
    
    try:
        # Use embedding-based similarity instead of token overlap heuristics
        query_embedding = None
        cache_enabled = (
            ctx.deps.cache_config is None or 
            ctx.deps.cache_config.enable_embedding_cache
        )
        
        if cache_enabled:
            query_embedding = embedding_cache.get(retrieval_result.query_variations.original_query)
        
        if query_embedding is None:
            query_embedding = get_vertex_text_embedding(
                text=retrieval_result.query_variations.original_query,
                model_name=ctx.deps.embedding_model_name,
                task_type="RETRIEVAL_QUERY",
                project_id=ctx.deps.vertex_project_id,
                location=ctx.deps.vertex_location
            )
            if query_embedding and cache_enabled:
                embedding_cache.store(retrieval_result.query_variations.original_query, query_embedding)
        
        if query_embedding is None:
            print("⚠️ Could not generate query embedding, using basic ranking")
            # Simple fallback without token overlap heuristics
            enhanced_documents = retrieval_result.ranked_documents.documents[:top_n]
        else:
            # Calculate semantic similarity scores
            enhanced_documents = []
            
            for ranked_doc in retrieval_result.ranked_documents.documents:
                # Get or generate document embedding
                doc_embedding = ranked_doc.document.embedding
                
                if doc_embedding is None:
                    # Generate consistent embedding for document with proper task_type
                    doc_embedding = get_vertex_text_embedding(
                        text=ranked_doc.document.content,
                        model_name=ctx.deps.embedding_model_name,
                        task_type="RETRIEVAL_DOCUMENT",  # Consistent task type for documents
                        project_id=ctx.deps.vertex_project_id,
                        location=ctx.deps.vertex_location
                    )
                    
                    # Normalize document embedding (L2 normalization)
                    if doc_embedding is not None:
                        import numpy as np
                        doc_vec = np.array(doc_embedding)
                        doc_embedding = (doc_vec / np.linalg.norm(doc_vec)).tolist()
                
                if doc_embedding is not None:
                    # Calculate cosine similarity (domain-agnostic)
                    import numpy as np
                    query_vec = np.array(query_embedding)
                    doc_vec = np.array(doc_embedding)
                    
                    # Normalize vectors
                    query_vec = query_vec / np.linalg.norm(query_vec)
                    doc_vec = doc_vec / np.linalg.norm(doc_vec)
                    
                    # Cosine similarity
                    similarity_score = float(np.dot(query_vec, doc_vec))
                else:
                    # Fallback to original score if embedding fails
                    similarity_score = ranked_doc.score
                
                enhanced_doc = RankedDocument(
                    document=ranked_doc.document,
                    score=similarity_score,
                    rank=ranked_doc.rank  # Will be updated after sorting
                )
                enhanced_documents.append(enhanced_doc)
            
            # Sort by semantic similarity score
            enhanced_documents.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks and take top_n
        for rank, doc in enumerate(enhanced_documents[:top_n], 1):
            doc.rank = rank
        
        final_documents = enhanced_documents[:top_n]
        
    except Exception as e:
        print(f"❌ Semantic similarity ranking failed: {e}")
        # Ultimate fallback: use original ranking
        final_documents = retrieval_result.ranked_documents.documents[:top_n]
        for rank, doc in enumerate(final_documents, 1):
            doc.rank = rank
    
    # Create final RankedDocuments collection
    final_collection = RankedDocuments(
        documents=final_documents,
        total_candidates=retrieval_result.ranked_documents.total_candidates,
        ranking_method="domain_agnostic_semantic_similarity",
        query=retrieval_result.query_variations.original_query
    )
    
    print(f"✅ Domain-agnostic semantic ranking completed: {len(final_documents)} documents")
    avg_score = sum(d.score for d in final_documents) / len(final_documents) if final_documents else 0
    print(f"---> Average semantic similarity score: {avg_score:.3f}")
    
    return final_collection

@agent.tool
async def format_context_tool(
    ctx: RunContext[RAGDeps],
    ranked_documents: RankedDocuments,
    include_metadata: bool = True
) -> str:
    """Format ranked documents into structured context with validation and quality scoring."""
    print(f"--- Context Formatter Tool Called ---")
    print(f"Input documents: {len(ranked_documents.documents)}")
    print(f"Ranking method: {ranked_documents.ranking_method}")
    
    if not ranked_documents.documents:
        return "No relevant context found."
    
    # Superscript mapping for citations
    superscript_map = {
        1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹', 10: '¹⁰',
        11: '¹¹', 12: '¹²', 13: '¹³', 14: '¹⁴', 15: '¹⁵', 16: '¹⁶', 17: '¹⁷', 18: '¹⁸', 19: '¹⁹', 20: '²⁰'
    }
    
    # Validate document chunks and metadata consistency
    validated_documents = []
    for doc in ranked_documents.documents:
        try:
            # Validate that document content is meaningful
            if len(doc.document.content.strip()) < 10:
                print(f"⚠️ Skipping document with insufficient content (rank {doc.rank})")
                continue
            
            # Validate metadata
            if not doc.document.metadata.url or not doc.document.metadata.url.strip():
                print(f"⚠️ Document at rank {doc.rank} has invalid URL metadata")
                continue
            
            validated_documents.append(doc)
            
        except Exception as e:
            print(f"❌ Error validating document at rank {doc.rank}: {e}")
            continue
    
    if not validated_documents:
        return "No valid context documents found after validation."
    
    print(f"✅ Validated {len(validated_documents)} documents")
    
    # Create URL to reference number mapping (deduplicate URLs)
    url_to_ref_num = {}
    unique_sources = []
    context_parts = []
    
    # Calculate context quality score
    total_content_length = 0
    total_relevance_score = 0.0
    
    for ranked_doc in validated_documents:
        source_url = ranked_doc.document.metadata.url
        
        # Assign reference number (deduplicate URLs)
        if source_url not in url_to_ref_num:
            ref_num = len(url_to_ref_num) + 1
            url_to_ref_num[source_url] = ref_num
            superscript = superscript_map.get(ref_num, f"^{ref_num}")
            
            # Create source reference with metadata
            source_info = f"{superscript} {source_url}"
            if ranked_doc.document.metadata.title:
                source_info += f" ({ranked_doc.document.metadata.title})"
            unique_sources.append(source_info)
        else:
            ref_num = url_to_ref_num[source_url]
        
        # Format document content with citation
        superscript = superscript_map.get(ref_num, f"^{ref_num}")
        
        # Add quality indicators if requested
        quality_info = ""
        if include_metadata:
            quality_info = f" [Relevanz: {ranked_doc.score:.3f}, Rang: {ranked_doc.rank}]"
        
        context_part = f"[Quelle {superscript}] {ranked_doc.document.content.strip()}{quality_info}"
        context_parts.append(context_part)
        
        # Update quality metrics
        total_content_length += len(ranked_doc.document.content)
        total_relevance_score += ranked_doc.score
    
    # Calculate context quality metrics
    avg_relevance = total_relevance_score / len(validated_documents)
    avg_content_length = total_content_length / len(validated_documents)
    
    print(f"📊 Context Quality Metrics:")
    print(f"  - Average relevance score: {avg_relevance:.3f}")
    print(f"  - Average content length: {avg_content_length:.0f} chars")
    print(f"  - Unique sources: {len(unique_sources)}")
    
    # Combine context parts
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Create clickable references for the sources section
    clickable_references = []
    for url, ref_num in url_to_ref_num.items():
        superscript = superscript_map.get(ref_num, f"^{ref_num}")
        clickable_references.append(f"[{superscript}]({url}) {url}")
    
    references_text = "\n".join(clickable_references)
    
    # Create citation mapping instructions for the LLM
    url_mapping_instructions = "ANWEISUNG FÜR ZITATIONEN: Verwende diese hochgestellten Zahlen als KLICKBARE LINKS im Text:\n"
    for url, ref_num in url_to_ref_num.items():
        superscript = superscript_map.get(ref_num, f"^{ref_num}")
        url_mapping_instructions += f"[{superscript}]({url}) für {url}\n"
    
    url_mapping_instructions += "\nWICHTIG: Verwende im Fließtext IMMER das Format [{superscript}](URL) für klickbare Quellenverweise! NIEMALS nur hochgestellte Zahlen ohne Links verwenden.\n"
    url_mapping_instructions += "KONSISTENZ: Halte die gleiche Detailtiefe und Formatierung wie in vorherigen Antworten bei. Jede Antwort soll vollständig und umfassend sein."
    
    # Add context metadata if requested
    metadata_section = ""
    if include_metadata:
        metadata_section = f"\n\n--- KONTEXT-METADATEN ---\n"
        metadata_section += f"Ranking-Methode: {ranked_documents.ranking_method}\n"
        metadata_section += f"Durchschnittliche Relevanz: {avg_relevance:.3f}\n"
        metadata_section += f"Gesamte Kandidaten: {ranked_documents.total_candidates}\n"
        metadata_section += f"Verwendete Dokumente: {len(validated_documents)}\n"
    
    # Construct final formatted context
    formatted_context = f"{context_text}\n\n--- QUELLENVERZEICHNIS ---\n{references_text}\n\n--- ZITATIONS-MAPPING ---\n{url_mapping_instructions}{metadata_section}"
    
    print(f"✅ Context formatting completed")
    print(f"  - Total context length: {len(formatted_context)} characters")
    print(f"  - Context parts: {len(context_parts)}")
    print(f"  - Unique sources: {len(unique_sources)}")
    
    return formatted_context

@agent.tool
async def get_cache_metrics_tool(ctx: RunContext[RAGDeps]) -> CacheMetrics:
    """Get comprehensive cache performance metrics as a tool."""
    print("--- Cache Metrics Tool Called ---")
    metrics = get_cache_metrics()
    
    print(f"📊 Cache Performance:")
    print(f"  - Total cache hit rate: {metrics.cache_hit_rate:.1%}")
    print(f"  - LLM cache: {metrics.llm_cache_stats.get('hit_count', 0)} hits, {metrics.llm_cache_stats.get('miss_count', 0)} misses")
    print(f"  - Embedding cache: {metrics.embedding_cache_stats.get('hit_count', 0)} hits, {metrics.embedding_cache_stats.get('miss_count', 0)} misses")
    
    return metrics

# ===== OPTIMIZED BATCH PROCESSING =====

class BatchItem(BaseModel):
    """Individual item in a batch processing operation."""
    id: str = Field(..., description="Unique identifier for this batch item")
    data: Any = Field(..., description="Data to be processed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class BatchResult(BaseModel):
    """Result of a batch processing operation."""
    id: str = Field(..., description="Unique identifier matching the input item")
    success: bool = Field(..., description="Whether processing was successful")
    result: Optional[Any] = Field(None, description="Processing result if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")

class BatchProcessingResult(BaseModel):
    """Complete result of batch processing with metrics."""
    results: List[BatchResult] = Field(..., description="Individual batch results")
    metrics: BatchProcessingMetrics = Field(..., description="Batch processing metrics")
    total_processing_time: float = Field(..., ge=0.0, description="Total time for entire batch")

async def process_batch_with_structured_models(
    items: List[BatchItem],
    processor_func: callable,
    config: Optional[BatchProcessingConfig] = None,
    semaphore_limit: Optional[int] = None
) -> BatchProcessingResult:
    """Optimized batch processing with structured models and error handling."""
    import time
    import asyncio
    from typing import Callable
    
    start_time = time.time()
    
    # Use default config if none provided
    if config is None:
        config = BatchProcessingConfig()
    
    if not config.enable_batch_optimization:
        # Process items sequentially if batch optimization is disabled
        results = []
        for item in items:
            item_start = time.time()
            try:
                result = await processor_func(item.data)
                results.append(BatchResult(
                    id=item.id,
                    success=True,
                    result=result,
                    processing_time=time.time() - item_start
                ))
            except Exception as e:
                results.append(BatchResult(
                    id=item.id,
                    success=False,
                    error=str(e),
                    processing_time=time.time() - item_start
                ))
    else:
        # Optimized parallel batch processing
        semaphore = asyncio.Semaphore(semaphore_limit or config.max_concurrent_batches)
        
        async def process_single_item(item: BatchItem) -> BatchResult:
            """Process a single item with error handling and timing."""
            async with semaphore:
                item_start = time.time()
                try:
                    # Add timeout to prevent hanging
                    result = await asyncio.wait_for(
                        processor_func(item.data),
                        timeout=config.batch_timeout_seconds
                    )
                    return BatchResult(
                        id=item.id,
                        success=True,
                        result=result,
                        processing_time=time.time() - item_start
                    )
                except asyncio.TimeoutError:
                    return BatchResult(
                        id=item.id,
                        success=False,
                        error=f"Processing timeout after {config.batch_timeout_seconds}s",
                        processing_time=time.time() - item_start
                    )
                except Exception as e:
                    return BatchResult(
                        id=item.id,
                        success=False,
                        error=str(e),
                        processing_time=time.time() - item_start
                    )
        
        # Process all items in parallel with controlled concurrency
        print(f"🚀 Processing {len(items)} items with max {config.max_concurrent_batches} concurrent operations")
        results = await asyncio.gather(*[process_single_item(item) for item in items])
    
    # Calculate metrics
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    total_processing_time = time.time() - start_time
    avg_processing_time = sum(r.processing_time for r in results) / len(results) if results else 0.0
    
    metrics = BatchProcessingMetrics(
        total_batches_processed=1,
        average_batch_size=float(len(items)),
        average_processing_time=avg_processing_time,
        successful_batches=1 if successful_results else 0,
        failed_batches=1 if failed_results else 0,
        batch_success_rate=len(successful_results) / len(results) if results else 0.0
    )
    
    print(f"✅ Batch processing completed: {len(successful_results)}/{len(results)} successful")
    print(f"   Total time: {total_processing_time:.2f}s, Avg per item: {avg_processing_time:.2f}s")
    
    return BatchProcessingResult(
        results=results,
        metrics=metrics,
        total_processing_time=total_processing_time
    )

@agent.tool
async def process_batch_embeddings_tool(
    ctx: RunContext[RAGDeps],
    texts: List[str],
    task_type: str = "RETRIEVAL_QUERY"
) -> List[Optional[List[float]]]:
    """Optimized batch embedding generation with structured models."""
    print(f"--- Batch Embeddings Tool Called ---")
    print(f"Processing {len(texts)} texts for embeddings")
    
    # Create batch items
    batch_items = [
        BatchItem(id=f"text_{i}", data=text, metadata={"task_type": task_type})
        for i, text in enumerate(texts)
    ]
    
    # Define the embedding processor function
    async def embedding_processor(text: str) -> Optional[List[float]]:
        """Process a single text to generate embedding."""
        # Check cache first if enabled
        cache_enabled = (
            ctx.deps.cache_config is None or 
            ctx.deps.cache_config.enable_embedding_cache
        )
        
        if cache_enabled:
            cached_embedding = embedding_cache.get(text)
            if cached_embedding:
                return cached_embedding
        
        # Generate new embedding
        embedding = get_vertex_text_embedding(
            text=text,
            model_name=ctx.deps.embedding_model_name,
            task_type=task_type,
            project_id=ctx.deps.vertex_project_id,
            location=ctx.deps.vertex_location
        )
        
        # Cache the result if enabled
        if embedding and cache_enabled:
            embedding_cache.store(text, embedding)
        
        return embedding
    
    # Process batch with optimization
    batch_result = await process_batch_with_structured_models(
        items=batch_items,
        processor_func=embedding_processor,
        config=ctx.deps.batch_config,
        semaphore_limit=10  # Limit concurrent embedding requests
    )
    
    # Extract embeddings in original order
    embeddings = [None] * len(texts)
    for result in batch_result.results:
        index = int(result.id.split('_')[1])
        if result.success:
            embeddings[index] = result.result
        else:
            print(f"❌ Embedding failed for text {index}: {result.error}")
    
    print(f"📊 Batch Embedding Metrics:")
    print(f"  - Success rate: {batch_result.metrics.batch_success_rate:.1%}")
    print(f"  - Average processing time: {batch_result.metrics.average_processing_time:.2f}s")
    
    return embeddings

async def run_agent_with_fallback(prompt: str, deps: RAGDeps, system_prompt_override: str = None) -> str:
    """Enhanced agent execution with structured model fallback logic."""
    
    # Get fallback configuration from deps or use default
    fallback_config = ModelFallbackConfig()
    if hasattr(deps, 'fallback_config') and deps.fallback_config:
        fallback_config = deps.fallback_config
    
    # Define the agent execution function
    async def execute_agent(provider: ModelProvider) -> str:
        """Execute agent with specified provider."""
        
        # Select appropriate model based on provider
        if provider == ModelProvider.GOOGLE and GOOGLE_MODEL_AVAILABLE:
            model = get_primary_model()
            agent_name = "GoogleRAGAgent"
        else:  # OpenAI (fallback or primary if Google not available)
            model = get_fallback_model()
            agent_name = "OpenAIRAGAgent"
        
        # Create agent with appropriate model
        if system_prompt_override:
            temp_agent = Agent(
                name=agent_name,
                description=f"RAG agent using {provider.value} model",
                system_prompt=system_prompt_override,
                dependencies=RAGDeps,
                model=model
            )
            result = await temp_agent.run(prompt, deps=deps)
        else:
            # Use pre-configured agents
            if provider == ModelProvider.GOOGLE:
                result = await agent.run(prompt, deps=deps)
            else:
                result = await fallback_agent.run(prompt, deps=deps)
        
        return result.data if hasattr(result, 'data') else str(result)
    
    # Execute with fallback logic
    context = ErrorContext(
        operation="agent_execution",
        input_data={"prompt": prompt[:100], "system_prompt_override": bool(system_prompt_override)},
        user_context=prompt
    )
    
    fallback_result = await execute_with_model_fallback(
        operation_func=execute_agent,
        config=fallback_config,
        context=context
    )
    
    if fallback_result.success:
        if fallback_result.fallback_occurred:
            print(f"✅ Agent execution successful with {fallback_result.provider_used.value} (fallback used)")
        return fallback_result.result
    else:
        # Handle structured error
        error_context = ErrorContext(
            operation="agent_execution_final_failure",
            input_data={"prompt": prompt[:100]},
            user_context=prompt
        )
        
        if fallback_result.error:
            recovery_result = await handle_rag_error(
                Exception(fallback_result.error.message),
                error_context
            )
            
            if recovery_result.success and recovery_result.result is not None:
                return recovery_result.result
        
        # Final failure
        error_msg = f"All model providers failed after {fallback_result.attempts_made} attempts"
        if fallback_result.error:
            error_msg += f": {fallback_result.error.message}"
        
        raise RuntimeError(error_msg)

# ===== SERVICE FALLBACK MECHANISMS =====

class ServiceType(str, Enum):
    """Types of services that can have fallback mechanisms."""
    RERANKING = "reranking"
    EMBEDDING = "embedding"
    MULTI_QUERY = "multi_query"
    RETRIEVAL = "retrieval"

class ServiceFallbackConfig(BaseModel):
    """Configuration for service-level fallback mechanisms."""
    enable_reranking_fallback: bool = Field(default=True, description="Enable fallback from Vertex AI to score-based ranking")
    enable_embedding_fallback: bool = Field(default=True, description="Enable fallback for embedding generation")
    enable_multi_query_fallback: bool = Field(default=True, description="Enable fallback from multi-query to single-query")
    max_service_retries: int = Field(default=2, ge=1, le=5, description="Maximum retries for service operations")
    service_timeout_seconds: float = Field(default=15.0, ge=5.0, le=60.0, description="Timeout for service operations")
    graceful_degradation: bool = Field(default=True, description="Enable graceful degradation on service failures")

class ServiceFallbackResult(BaseModel):
    """Result of service execution with fallback information."""
    success: bool = Field(..., description="Whether the service operation was successful")
    service_type: ServiceType = Field(..., description="Type of service that was executed")
    primary_method_used: bool = Field(..., description="Whether primary method was used")
    fallback_method: Optional[str] = Field(None, description="Fallback method used if primary failed")
    result: Optional[Any] = Field(None, description="Service result if successful")
    error: Optional[RAGError] = Field(None, description="Error if failed")
    degraded_quality: bool = Field(default=False, description="Whether result quality is degraded due to fallback")
    execution_time: float = Field(..., ge=0.0, description="Total execution time")

async def execute_with_service_fallback(
    service_type: ServiceType,
    primary_func: callable,
    fallback_func: callable,
    config: Optional[ServiceFallbackConfig] = None,
    context: Optional[ErrorContext] = None
) -> ServiceFallbackResult:
    """Execute a service operation with fallback mechanism."""
    import time
    start_time = time.time()
    
    if config is None:
        config = ServiceFallbackConfig()
    
    if context is None:
        context = ErrorContext(operation=f"{service_type.value}_execution")
    
    print(f"🔧 Executing {service_type.value} service...")
    
    # Try primary method first
    try:
        result = await asyncio.wait_for(
            primary_func(),
            timeout=config.service_timeout_seconds
        )
        
        execution_time = time.time() - start_time
        print(f"✅ {service_type.value} primary method succeeded in {execution_time:.2f}s")
        
        return ServiceFallbackResult(
            success=True,
            service_type=service_type,
            primary_method_used=True,
            result=result,
            execution_time=execution_time
        )
        
    except Exception as primary_error:
        print(f"❌ {service_type.value} primary method failed: {primary_error}")
        
        # Determine if fallback should be attempted
        should_fallback = True
        
        if service_type == ServiceType.RERANKING and not config.enable_reranking_fallback:
            should_fallback = False
        elif service_type == ServiceType.EMBEDDING and not config.enable_embedding_fallback:
            should_fallback = False
        elif service_type == ServiceType.MULTI_QUERY and not config.enable_multi_query_fallback:
            should_fallback = False
        
        if not should_fallback or not fallback_func:
            # No fallback available or disabled
            execution_time = time.time() - start_time
            error = create_rag_error(
                ErrorType.UNKNOWN_ERROR,
                f"{service_type.value} service failed and no fallback available",
                f"{service_type.value}_service",
                details=str(primary_error),
                context={"service": service_type.value, "fallback_disabled": not should_fallback}
            )
            
            return ServiceFallbackResult(
                success=False,
                service_type=service_type,
                primary_method_used=False,
                error=error,
                execution_time=execution_time
            )
        
        # Try fallback method
        print(f"🔄 Attempting {service_type.value} fallback method...")
        
        try:
            fallback_result = await asyncio.wait_for(
                fallback_func(),
                timeout=config.service_timeout_seconds
            )
            
            execution_time = time.time() - start_time
            print(f"✅ {service_type.value} fallback method succeeded in {execution_time:.2f}s")
            
            return ServiceFallbackResult(
                success=True,
                service_type=service_type,
                primary_method_used=False,
                fallback_method="fallback_implementation",
                result=fallback_result,
                degraded_quality=True,  # Fallback may have lower quality
                execution_time=execution_time
            )
            
        except Exception as fallback_error:
            print(f"💥 {service_type.value} fallback method also failed: {fallback_error}")
            
            execution_time = time.time() - start_time
            error = create_rag_error(
                ErrorType.UNKNOWN_ERROR,
                f"{service_type.value} service failed completely",
                f"{service_type.value}_service",
                details=f"Primary: {primary_error}, Fallback: {fallback_error}",
                context={"service": service_type.value, "both_methods_failed": True}
            )
            
            return ServiceFallbackResult(
                success=False,
                service_type=service_type,
                primary_method_used=False,
                fallback_method="fallback_implementation",
                error=error,
                execution_time=execution_time
            )

@agent.tool
async def rerank_with_service_fallback(
    ctx: RunContext[RAGDeps],
    retrieval_result: RetrievalResult,
    top_n: int = 10
) -> RankedDocuments:
    """Re-ranking with automatic fallback from Vertex AI to score-based ranking."""
    print(f"--- Re-Ranking with Service Fallback ---")
    
    # Get service fallback config
    service_config = ServiceFallbackConfig()
    if hasattr(ctx.deps, 'service_fallback_config'):
        service_config = ctx.deps.service_fallback_config
    
    # Define primary method (Vertex AI re-ranking)
    async def vertex_ai_reranking():
        if not (ctx.deps.use_vertex_reranker and ctx.deps.vertex_reranker_model):
            raise Exception("Vertex AI re-ranking not configured")
        
        # Use the existing rerank_documents_tool logic for Vertex AI
        return await rerank_documents_tool(ctx, retrieval_result, top_n)
    
    # Define fallback method (enhanced score-based ranking)
    async def score_based_reranking():
        print("🎯 Using enhanced score-based ranking as fallback")
        
        enhanced_documents = []
        query_words = set(retrieval_result.query_variations.original_query.lower().split())
        
        # Remove stop words
        stop_words = {'der', 'die', 'das', 'und', 'oder', 'aber', 'ist', 'sind', 'was', 'wie', 'wo', 'wann', 'warum'}
        query_words = query_words - stop_words
        
        for ranked_doc in retrieval_result.ranked_documents.documents:
            content = ranked_doc.document.content.lower()
            content_words = set(content.split()) - stop_words
            
            # Enhanced scoring factors
            base_score = ranked_doc.score
            
            # 1. Word overlap bonus
            if query_words:
                word_overlap = len(query_words.intersection(content_words))
                overlap_bonus = (word_overlap / len(query_words)) * 0.3
            else:
                overlap_bonus = 0.0
            
            # 2. Content quality bonus
            content_length = len(ranked_doc.document.content)
            if 200 <= content_length <= 2000:
                quality_bonus = 0.1
            elif content_length < 50:
                quality_bonus = -0.2
            else:
                quality_bonus = min(content_length / 2000, 0.1)
            
            # 3. URL quality bonus
            url_bonus = 0.0
            url = ranked_doc.document.metadata.url.lower()
            if any(domain in url for domain in ['docs.', 'documentation', 'help.', 'support.']):
                url_bonus = 0.05
            
            # Calculate enhanced score
            enhanced_score = min(base_score + overlap_bonus + quality_bonus + url_bonus, 1.0)
            
            enhanced_doc = RankedDocument(
                document=ranked_doc.document,
                score=enhanced_score,
                rank=ranked_doc.rank  # Will be updated after sorting
            )
            enhanced_documents.append(enhanced_doc)
        
        # Sort by enhanced score and update ranks
        enhanced_documents.sort(key=lambda x: x.score, reverse=True)
        for rank, doc in enumerate(enhanced_documents[:top_n], 1):
            doc.rank = rank
        
        # Create final RankedDocuments collection
        return RankedDocuments(
            documents=enhanced_documents[:top_n],
            total_candidates=retrieval_result.ranked_documents.total_candidates,
            ranking_method="enhanced_score_based_fallback",
            query=retrieval_result.query_variations.original_query
        )
    
    # Execute with service fallback
    context = ErrorContext(
        operation="reranking_service",
        input_data={"documents_count": len(retrieval_result.ranked_documents.documents), "top_n": top_n}
    )
    
    fallback_result = await execute_with_service_fallback(
        service_type=ServiceType.RERANKING,
        primary_func=vertex_ai_reranking,
        fallback_func=score_based_reranking,
        config=service_config,
        context=context
    )
    
    if fallback_result.success:
        if fallback_result.degraded_quality:
            print("⚠️ Using fallback re-ranking method - quality may be reduced")
        return fallback_result.result
    else:
        # Return original ranking if both methods fail
        print("💥 All re-ranking methods failed, returning original ranking")
        return retrieval_result.ranked_documents

# ===== STRUCTURED OUTPUT FORMATS =====

@agent.tool
async def generate_structured_response(
    ctx: RunContext[RAGDeps],
    query: str,
    retrieval_result: RetrievalResult,
    ranked_documents: RankedDocuments,
    formatted_context: str
) -> StructuredRagAnswer:
    """Generate structured RAG response with comprehensive metadata and confidence scoring."""
    print(f"--- Structured Response Generation ---")
    print(f"Query: {query}")
    print(f"Documents: {len(ranked_documents.documents)}")
    
    # Calculate confidence score based on retrieval quality metrics
    confidence_factors = []
    
    # 1. Average relevance score of retrieved documents
    if ranked_documents.documents:
        avg_relevance = sum(doc.score for doc in ranked_documents.documents) / len(ranked_documents.documents)
        confidence_factors.append(("relevance", avg_relevance, 0.4))  # 40% weight
    else:
        confidence_factors.append(("relevance", 0.0, 0.4))
    
    # 2. Number of high-quality documents (score > 0.7)
    high_quality_docs = len([doc for doc in ranked_documents.documents if doc.score > 0.7])
    quality_ratio = high_quality_docs / len(ranked_documents.documents) if ranked_documents.documents else 0
    confidence_factors.append(("quality_ratio", quality_ratio, 0.2))  # 20% weight
    
    # 3. Diversity of sources (unique URLs)
    unique_urls = set(doc.document.metadata.url for doc in ranked_documents.documents)
    source_diversity = min(len(unique_urls) / max(len(ranked_documents.documents), 1), 1.0)
    confidence_factors.append(("source_diversity", source_diversity, 0.15))  # 15% weight
    
    # 4. Retrieval method quality (Vertex AI > score-based)
    method_quality = 1.0 if "vertex_ai" in ranked_documents.ranking_method else 0.7
    confidence_factors.append(("method_quality", method_quality, 0.15))  # 15% weight
    
    # 5. Cache hit rate (higher = more reliable)
    cache_metrics = get_cache_metrics()
    cache_quality = cache_metrics.cache_hit_rate
    confidence_factors.append(("cache_quality", cache_quality, 0.1))  # 10% weight
    
    # Calculate weighted confidence score
    confidence_score = sum(score * weight for _, score, weight in confidence_factors)
    confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]
    
    print(f"📊 Confidence Factors:")
    for name, score, weight in confidence_factors:
        print(f"  - {name}: {score:.3f} (weight: {weight:.1%})")
    print(f"  - Final confidence: {confidence_score:.3f}")
    
    # Generate the actual response using the agent
    response_prompt = f"""Based on the provided context, answer the following question comprehensively:

Question: {query}

Context: {formatted_context}

Please provide:
1. A concise summary (2-3 sentences)
2. Key details and facts
3. Contact information if available
4. Maintain the same detailed formatting as previous responses
"""
    
    try:
        # Generate response with fallback
        response_text = await run_agent_with_fallback(
            prompt=response_prompt,
            deps=ctx.deps,
            system_prompt_override="You are a helpful assistant that provides comprehensive, well-structured answers based on the given context. Always cite sources using the provided citation format."
        )
        
        # Parse the response to extract components
        # This is a simplified parsing - in production, you might want more sophisticated NLP
        lines = response_text.split('\n')
        summary_lines = []
        key_details = []
        contact_info = None
        
        current_section = "summary"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section changes
            if any(keyword in line.lower() for keyword in ['details:', 'wichtige', 'key']):
                current_section = "details"
                continue
            elif any(keyword in line.lower() for keyword in ['kontakt', 'contact']):
                current_section = "contact"
                continue
            
            # Add content to appropriate section
            if current_section == "summary" and len(summary_lines) < 3:
                summary_lines.append(line)
            elif current_section == "details" and line.startswith(('-', '•', '*')):
                key_details.append(line.lstrip('-•* '))
            elif current_section == "contact":
                contact_info = line
        
        # Fallback parsing if structured parsing fails
        if not summary_lines:
            # Use first few sentences as summary
            sentences = response_text.split('.')
            summary_lines = [s.strip() + '.' for s in sentences[:2] if s.strip()]
        
        if not key_details:
            # Extract bullet points or numbered items
            import re
            bullet_pattern = r'^[-•*]\s*(.+)$'
            for line in lines:
                match = re.match(bullet_pattern, line.strip())
                if match:
                    key_details.append(match.group(1))
        
        # Ensure we have at least some content
        summary = ' '.join(summary_lines) if summary_lines else response_text[:200] + "..."
        if not key_details:
            key_details = ["Detailed information available in the provided sources."]
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        # Fallback response
        summary = f"Based on the available information, here is what I found regarding: {query}"
        key_details = ["Information retrieved from multiple sources", "Please refer to the source documents for complete details"]
        contact_info = None
        confidence_score *= 0.5  # Reduce confidence due to generation error
    
    # Create source references with metadata
    source_references = []
    for i, doc in enumerate(ranked_documents.documents[:10], 1):  # Limit to top 10 sources
        # Extract excerpt from document
        excerpt = doc.document.content[:150] + "..." if len(doc.document.content) > 150 else doc.document.content
        
        source_ref = SourceReference(
            url=doc.document.metadata.url,
            title=doc.document.metadata.title,
            relevance_score=doc.score,
            citation_number=i,
            excerpt=excerpt
        )
        source_references.append(source_ref)
    
    # Create retrieval metadata
    retrieval_metadata = RetrievalMetadata(
        total_documents_searched=retrieval_result.ranked_documents.total_candidates,
        documents_retrieved=len(ranked_documents.documents),
        queries_used=len(retrieval_result.query_variations.variations) + 1,  # +1 for original
        retrieval_method=retrieval_result.ranked_documents.ranking_method,
        reranking_method=ranked_documents.ranking_method,
        average_relevance_score=sum(doc.score for doc in ranked_documents.documents) / len(ranked_documents.documents) if ranked_documents.documents else 0.0,
        cache_hit_rate=cache_metrics.cache_hit_rate,
        processing_time_seconds=retrieval_result.retrieval_time
    )
    
    # Determine answer type based on query
    answer_type = "informational"
    query_lower = query.lower()
    if any(word in query_lower for word in ['was ist', 'definition', 'bedeutung']):
        answer_type = "definition"
    elif any(word in query_lower for word in ['wie', 'anleitung', 'schritte']):
        answer_type = "procedural"
    elif any(word in query_lower for word in ['warum', 'grund', 'ursache']):
        answer_type = "explanatory"
    
    # Create final structured answer
    structured_answer = StructuredRagAnswer(
        summary=summary,
        key_details=key_details,
        contact_info=contact_info,
        source_references=source_references,
        confidence_score=confidence_score,
        retrieval_metadata=retrieval_metadata,
        answer_type=answer_type
    )
    
    print(f"✅ Structured response generated:")
    print(f"  - Summary length: {len(summary)} chars")
    print(f"  - Key details: {len(key_details)} items")
    print(f"  - Source references: {len(source_references)} sources")
    print(f"  - Confidence score: {confidence_score:.3f}")
    print(f"  - Answer type: {answer_type}")
    
    return structured_answer

class OutputFormat(str, Enum):
    """Available output formats for RAG responses."""
    STRUCTURED = "structured"
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"

class OutputFormatConfig(BaseModel):
    """Configuration for output format switching."""
    default_format: OutputFormat = Field(default=OutputFormat.TEXT, description="Default output format")
    allow_format_switching: bool = Field(default=True, description="Allow users to switch output formats")
    include_metadata_in_text: bool = Field(default=False, description="Include metadata in text format")
    include_confidence_score: bool = Field(default=True, description="Include confidence score in output")
    max_sources_in_text: int = Field(default=5, ge=1, le=20, description="Maximum sources to show in text format")

@agent.tool
async def format_response_output(
    ctx: RunContext[RAGDeps],
    structured_answer: StructuredRagAnswer,
    output_format: OutputFormat = OutputFormat.TEXT,
    config: Optional[OutputFormatConfig] = None
) -> str:
    """Format structured response according to specified output format."""
    print(f"--- Output Format Switching ---")
    print(f"Format: {output_format.value}")
    
    if config is None:
        config = OutputFormatConfig()
    
    if output_format == OutputFormat.STRUCTURED:
        # Return the structured answer as a formatted string representation
        return format_structured_answer(structured_answer)
    
    elif output_format == OutputFormat.TEXT:
        # Format as readable text (existing format)
        formatted_response = structured_answer.summary
        
        if structured_answer.key_details:
            formatted_response += "\n\n**Wichtige Details:**\n"
            for detail in structured_answer.key_details:
                formatted_response += f"- {detail}\n"
        
        if structured_answer.contact_info:
            formatted_response += f"\n**Kontakt:**\n{structured_answer.contact_info}\n"
        
        if config.include_confidence_score:
            formatted_response += f"\n**Konfidenz:** {structured_answer.confidence_score:.1%}\n"
        
        # Add sources with clickable links
        if structured_answer.source_references:
            formatted_response += "\n**Quellen:**\n"
            sources_to_show = structured_answer.source_references[:config.max_sources_in_text]
            
            for ref in sources_to_show:
                if ref.title:
                    formatted_response += f"[{ref.citation_number}]({ref.url}) {ref.title}\n"
                else:
                    formatted_response += f"[{ref.citation_number}]({ref.url})\n"
        
        # Add metadata if requested
        if config.include_metadata_in_text:
            metadata = structured_answer.retrieval_metadata
            formatted_response += f"\n**Retrieval-Informationen:**\n"
            formatted_response += f"- Durchsuchte Dokumente: {metadata.total_documents_searched}\n"
            formatted_response += f"- Verwendete Queries: {metadata.queries_used}\n"
            formatted_response += f"- Durchschnittliche Relevanz: {metadata.average_relevance_score:.3f}\n"
            formatted_response += f"- Verarbeitungszeit: {metadata.processing_time_seconds:.2f}s\n"
        
        return formatted_response
    
    elif output_format == OutputFormat.MARKDOWN:
        # Format as Markdown
        md_response = f"# Antwort\n\n{structured_answer.summary}\n\n"
        
        if structured_answer.key_details:
            md_response += "## Wichtige Details\n\n"
            for detail in structured_answer.key_details:
                md_response += f"- {detail}\n"
            md_response += "\n"
        
        if structured_answer.contact_info:
            md_response += f"## Kontakt\n\n{structured_answer.contact_info}\n\n"
        
        if config.include_confidence_score:
            md_response += f"## Qualitätsbewertung\n\n**Konfidenz:** {structured_answer.confidence_score:.1%}\n\n"
        
        # Add sources as markdown links
        if structured_answer.source_references:
            md_response += "## Quellen\n\n"
            sources_to_show = structured_answer.source_references[:config.max_sources_in_text]
            
            for ref in sources_to_show:
                title = ref.title or "Quelle"
                md_response += f"{ref.citation_number}. [{title}]({ref.url})"
                if ref.excerpt:
                    md_response += f"\n   > {ref.excerpt}"
                md_response += "\n\n"
        
        # Add metadata section
        if config.include_metadata_in_text:
            metadata = structured_answer.retrieval_metadata
            md_response += "## Retrieval-Informationen\n\n"
            md_response += f"- **Durchsuchte Dokumente:** {metadata.total_documents_searched}\n"
            md_response += f"- **Verwendete Queries:** {metadata.queries_used}\n"
            md_response += f"- **Retrieval-Methode:** {metadata.retrieval_method}\n"
            md_response += f"- **Re-Ranking-Methode:** {metadata.reranking_method or 'Keine'}\n"
            md_response += f"- **Durchschnittliche Relevanz:** {metadata.average_relevance_score:.3f}\n"
            md_response += f"- **Cache-Hit-Rate:** {metadata.cache_hit_rate:.1%}\n"
            md_response += f"- **Verarbeitungszeit:** {metadata.processing_time_seconds:.2f}s\n"
        
        return md_response
    
    elif output_format == OutputFormat.JSON:
        # Return as JSON string
        import json
        
        # Convert to dictionary for JSON serialization
        response_dict = {
            "summary": structured_answer.summary,
            "key_details": structured_answer.key_details,
            "contact_info": structured_answer.contact_info,
            "confidence_score": structured_answer.confidence_score,
            "answer_type": structured_answer.answer_type,
            "sources": [
                {
                    "citation_number": ref.citation_number,
                    "url": ref.url,
                    "title": ref.title,
                    "relevance_score": ref.relevance_score,
                    "excerpt": ref.excerpt
                }
                for ref in structured_answer.source_references[:config.max_sources_in_text]
            ]
        }
        
        if config.include_metadata_in_text:
            response_dict["retrieval_metadata"] = {
                "total_documents_searched": structured_answer.retrieval_metadata.total_documents_searched,
                "documents_retrieved": structured_answer.retrieval_metadata.documents_retrieved,
                "queries_used": structured_answer.retrieval_metadata.queries_used,
                "retrieval_method": structured_answer.retrieval_metadata.retrieval_method,
                "reranking_method": structured_answer.retrieval_metadata.reranking_method,
                "average_relevance_score": structured_answer.retrieval_metadata.average_relevance_score,
                "cache_hit_rate": structured_answer.retrieval_metadata.cache_hit_rate,
                "processing_time_seconds": structured_answer.retrieval_metadata.processing_time_seconds
            }
        
        return json.dumps(response_dict, indent=2, ensure_ascii=False)
    
    else:
        # Fallback to text format
        print(f"⚠️ Unknown output format {output_format}, falling back to text")
        return await format_response_output(ctx, structured_answer, OutputFormat.TEXT, config)

@agent.tool
async def complete_rag_pipeline(
    ctx: RunContext[RAGDeps],
    query: str,
    output_format: OutputFormat = OutputFormat.TEXT,
    n_results: int = 15
) -> str:
    """Complete RAG pipeline with structured processing and configurable output format."""
    print(f"=== Complete RAG Pipeline Started ===")
    print(f"Query: {query}")
    print(f"Output format: {output_format.value}")
    print(f"Results requested: {n_results}")
    
    try:
        # Step 1: Generate query variations
        print("\n🧠 Step 1: Generating query variations...")
        query_variations = await generate_query_variations_tool(ctx, query)
        
        # Step 2: Retrieve documents
        print("\n🔍 Step 2: Retrieving documents...")
        retrieval_result = await retrieve_documents_structured(ctx, query_variations, n_results * 2)
        
        # Step 3: Re-rank documents
        print("\n📊 Step 3: Re-ranking documents...")
        ranked_documents = await rerank_with_service_fallback(ctx, retrieval_result, n_results)
        
        # Step 4: Format context
        print("\n📝 Step 4: Formatting context...")
        formatted_context = await format_context_tool(ctx, ranked_documents, include_metadata=True)
        
        # Step 5: Generate structured response
        print("\n🎯 Step 5: Generating structured response...")
        structured_answer = await generate_structured_response(
            ctx, query, retrieval_result, ranked_documents, formatted_context
        )
        
        # Step 6: Format output according to requested format
        print("\n🎨 Step 6: Formatting output...")
        final_response = await format_response_output(ctx, structured_answer, output_format)
        
        print(f"\n✅ RAG Pipeline completed successfully!")
        print(f"   - Query variations: {len(query_variations.variations) + 1}")
        print(f"   - Documents retrieved: {len(retrieval_result.ranked_documents.documents)}")
        print(f"   - Final documents: {len(ranked_documents.documents)}")
        print(f"   - Confidence score: {structured_answer.confidence_score:.3f}")
        print(f"   - Output format: {output_format.value}")
        
        return final_response
        
    except Exception as e:
        print(f"❌ RAG Pipeline failed: {e}")
        
        # Create error context and attempt recovery
        error_context = ErrorContext(
            operation="complete_rag_pipeline",
            input_data={"query": query, "output_format": output_format.value},
            user_context=query
        )
        
        recovery_result = await handle_rag_error(e, error_context)
        
        if recovery_result.success and recovery_result.result:
            return recovery_result.result
        else:
            # Return error message in requested format
            error_response = f"Es tut mir leid, aber ich konnte Ihre Anfrage nicht verarbeiten: {str(e)}"
            
            if output_format == OutputFormat.JSON:
                import json
                return json.dumps({"error": error_response, "success": False}, ensure_ascii=False)
            elif output_format == OutputFormat.MARKDOWN:
                return f"# Fehler\n\n{error_response}"
            else:
                return error_response

# ===== BACKWARD COMPATIBILITY AND INTEGRATION =====

class LegacyConfigAdapter:
    """Adapter to map legacy CLI parameters and environment variables to new Pydantic models."""
    
    @staticmethod
    def create_rag_deps_from_legacy_args(args) -> RAGDeps:
        """Create RAGDeps from legacy command line arguments."""
        print("🔄 Converting legacy configuration to structured models...")
        
        # Extract basic parameters
        chroma_client = get_chroma_client(args.db_dir) if hasattr(args, 'db_dir') else None
        collection_name = getattr(args, 'collection_name', 'default_collection')
        embedding_model_name = getattr(args, 'embedding_model', 'text-embedding-004')
        embedding_provider = getattr(args, 'embedding_provider', 'vertex_ai')
        
        # Vertex AI parameters
        vertex_project_id = getattr(args, 'vertex_project_id', None) or os.getenv('GOOGLE_CLOUD_PROJECT')
        vertex_location = getattr(args, 'vertex_location', None) or os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        # Vertex AI Reranker parameters
        use_vertex_reranker = getattr(args, 'use_vertex_reranker', False)
        vertex_reranker_model = getattr(args, 'vertex_reranker_model', None)
        
        # Create cache configuration from legacy settings
        cache_config = PydanticAICacheConfig(
            enable_llm_cache=getattr(args, 'enable_cache', True),
            enable_embedding_cache=getattr(args, 'enable_embedding_cache', True),
            llm_cache_ttl_hours=getattr(args, 'cache_ttl_hours', 1),
            llm_cache_max_size=getattr(args, 'cache_max_size', 500),
            embedding_cache_max_size=getattr(args, 'embedding_cache_max_size', 1000),
            cache_hit_logging=getattr(args, 'verbose', False)
        )
        
        # Create batch processing configuration
        batch_config = BatchProcessingConfig(
            max_batch_size=getattr(args, 'batch_size', 10),
            max_concurrent_batches=getattr(args, 'max_concurrent', 3),
            batch_timeout_seconds=getattr(args, 'timeout', 30.0),
            enable_batch_optimization=getattr(args, 'enable_batch_optimization', True),
            adaptive_batch_sizing=getattr(args, 'adaptive_batching', True)
        )
        
        # Create RAGDeps with all configurations
        deps = RAGDeps(
            chroma_client=chroma_client,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider,
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            use_vertex_reranker=use_vertex_reranker,
            vertex_reranker_model=vertex_reranker_model,
            cache_config=cache_config,
            batch_config=batch_config
        )
        
        print(f"✅ Legacy configuration converted:")
        print(f"  - Collection: {collection_name}")
        print(f"  - Embedding model: {embedding_model_name}")
        print(f"  - Vertex AI project: {vertex_project_id}")
        print(f"  - Reranker enabled: {use_vertex_reranker}")
        print(f"  - Cache enabled: {cache_config.enable_llm_cache}")
        
        return deps
    
    @staticmethod
    def create_rag_deps_from_env() -> RAGDeps:
        """Create RAGDeps from environment variables."""
        print("🔄 Creating configuration from environment variables...")
        
        # Get ChromaDB client (in-memory for compatibility)
        try:
            import chromadb
            chroma_client = chromadb.Client()
        except Exception as e:
            print(f"Warning: Could not create ChromaDB client: {e}")
            chroma_client = None
        
        # Extract from environment variables
        collection_name = os.getenv('COLLECTION_NAME', 'default_collection')
        embedding_model_name = os.getenv('EMBEDDING_MODEL', 'text-embedding-004')
        embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'vertex_ai')
        
        # Vertex AI settings
        vertex_project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        vertex_location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        # Reranker settings
        use_vertex_reranker = os.getenv('USE_VERTEX_RERANKER', 'false').lower() == 'true'
        vertex_reranker_model = os.getenv('VERTEX_RERANKER_MODEL')
        
        # Cache configuration from environment
        cache_config = PydanticAICacheConfig(
            enable_llm_cache=os.getenv('ENABLE_LLM_CACHE', 'true').lower() == 'true',
            enable_embedding_cache=os.getenv('ENABLE_EMBEDDING_CACHE', 'true').lower() == 'true',
            llm_cache_ttl_hours=int(os.getenv('CACHE_TTL_HOURS', '1')),
            llm_cache_max_size=int(os.getenv('CACHE_MAX_SIZE', '500')),
            embedding_cache_max_size=int(os.getenv('EMBEDDING_CACHE_MAX_SIZE', '1000')),
            cache_hit_logging=os.getenv('VERBOSE', 'false').lower() == 'true'
        )
        
        # Batch processing configuration from environment
        batch_config = BatchProcessingConfig(
            max_batch_size=int(os.getenv('BATCH_SIZE', '10')),
            max_concurrent_batches=int(os.getenv('MAX_CONCURRENT', '3')),
            batch_timeout_seconds=float(os.getenv('BATCH_TIMEOUT', '30.0')),
            enable_batch_optimization=os.getenv('ENABLE_BATCH_OPTIMIZATION', 'true').lower() == 'true',
            adaptive_batch_sizing=os.getenv('ADAPTIVE_BATCHING', 'true').lower() == 'true'
        )
        
        deps = RAGDeps(
            chroma_client=chroma_client,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider,
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            use_vertex_reranker=use_vertex_reranker,
            vertex_reranker_model=vertex_reranker_model,
            cache_config=cache_config,
            batch_config=batch_config
        )
        
        print(f"✅ Environment configuration loaded:")
        print(f"  - Collection: {collection_name}")
        print(f"  - Vertex AI project: {vertex_project_id}")
        print(f"  - Reranker: {use_vertex_reranker}")
        
        return deps
    
    @staticmethod
    def validate_legacy_configuration(deps: RAGDeps) -> List[str]:
        """Validate legacy configuration and return list of warnings/errors."""
        warnings = []
        
        # Check required components
        if not deps.chroma_client:
            warnings.append("ChromaDB client not available - some features may not work")
        
        if not deps.vertex_project_id:
            warnings.append("Vertex AI project ID not set - embedding generation may fail")
        
        if deps.use_vertex_reranker and not deps.vertex_reranker_model:
            warnings.append("Vertex AI reranker enabled but model not specified")
        
        # Check API keys
        if not os.getenv('GEMINI_API_KEY'):
            warnings.append("GEMINI_API_KEY not set - will fallback to OpenAI")
        
        if not os.getenv('OPENAI_API_KEY'):
            warnings.append("OPENAI_API_KEY not set - fallback may not work")
        
        # Check cache configuration
        if deps.cache_config and deps.cache_config.llm_cache_max_size > 10000:
            warnings.append("LLM cache size very large - may consume significant memory")
        
        return warnings

def create_legacy_compatible_entrypoint(
    question: str,
    collection_name: str = "default_collection",
    db_dir: str = "./chroma_db",
    embedding_model: str = "text-embedding-004",
    vertex_project_id: str = None,
    vertex_location: str = "us-central1",
    use_vertex_reranker: bool = False,
    vertex_reranker_model: str = None,
    llm_model: str = "gemini-2.5-flash",
    output_format: str = "text",
    n_results: int = 10,
    enable_cache: bool = True,
    verbose: bool = False
) -> str:
    """Legacy-compatible entrypoint that maintains the same interface as the original system."""
    print("🔄 Legacy-compatible RAG execution started...")
    
    # Create a mock args object for compatibility
    class MockArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = MockArgs(
        db_dir=db_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_provider="vertex_ai",
        vertex_project_id=vertex_project_id,
        vertex_location=vertex_location,
        use_vertex_reranker=use_vertex_reranker,
        vertex_reranker_model=vertex_reranker_model,
        enable_cache=enable_cache,
        verbose=verbose,
        batch_size=10,
        max_concurrent=3,
        timeout=30.0
    )
    
    # Convert to structured configuration
    deps = LegacyConfigAdapter.create_rag_deps_from_legacy_args(args)
    
    # Validate configuration
    warnings = LegacyConfigAdapter.validate_legacy_configuration(deps)
    for warning in warnings:
        print(f"⚠️ Configuration warning: {warning}")
    
    # Map output format
    format_mapping = {
        "text": OutputFormat.TEXT,
        "markdown": OutputFormat.MARKDOWN,
        "json": OutputFormat.JSON,
        "structured": OutputFormat.STRUCTURED
    }
    
    output_fmt = format_mapping.get(output_format.lower(), OutputFormat.TEXT)
    
    # Execute the new pipeline
    import asyncio
    
    async def run_pipeline():
        # Create a mock context for the agent tools
        class MockContext:
            def __init__(self, deps):
                self.deps = deps
        
        ctx = MockContext(deps)
        
        # Use the complete RAG pipeline
        return await complete_rag_pipeline(ctx, question, output_fmt, n_results)
    
    try:
        # Run the async pipeline
        if hasattr(asyncio, 'run'):
            result = asyncio.run(run_pipeline())
        else:
            # Fallback for older Python versions
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(run_pipeline())
        
        print("✅ Legacy-compatible execution completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Legacy-compatible execution failed: {e}")
        
        # Fallback to simple error message
        error_msg = f"Es tut mir leid, aber ich konnte Ihre Anfrage nicht verarbeiten: {str(e)}"
        
        if output_format.lower() == "json":
            import json
            return json.dumps({"error": error_msg, "success": False}, ensure_ascii=False)
        else:
            return error_msg

# ===== STREAMLIT INTEGRATION ADAPTER =====

class StreamlitIntegrationAdapter:
    """Adapter for Streamlit integration with the new Pydantic AI system."""
    
    @staticmethod
    def create_streamlit_compatible_function():
        """Create a function that's compatible with existing Streamlit integration."""
        
        async def streamlit_rag_agent_entrypoint(
            question: str,
            deps: RAGDeps,
            llm_model: str = "gemini-2.5-flash"
        ) -> str:
            """Streamlit-compatible entrypoint that uses the new structured system."""
            print(f"🎨 Streamlit RAG execution: {question[:50]}...")
            
            try:
                # Create a mock context for the agent tools
                class MockContext:
                    def __init__(self, deps):
                        self.deps = deps
                
                ctx = MockContext(deps)
                
                # Use the complete RAG pipeline with text output (Streamlit default)
                result = await complete_rag_pipeline(
                    ctx=ctx,
                    query=question,
                    output_format=OutputFormat.TEXT,
                    n_results=15
                )
                
                return result
                
            except Exception as e:
                print(f"❌ Streamlit RAG execution failed: {e}")
                
                # Create error context and attempt recovery
                error_context = ErrorContext(
                    operation="streamlit_rag_execution",
                    input_data={"question": question[:100], "model": llm_model},
                    user_context=question
                )
                
                recovery_result = await handle_rag_error(e, error_context)
                
                if recovery_result.success and recovery_result.result:
                    return recovery_result.result
                else:
                    return f"Es tut mir leid, aber ich konnte Ihre Anfrage nicht verarbeiten: {str(e)}"
        
        return streamlit_rag_agent_entrypoint
    
    @staticmethod
    def create_streamlit_deps_from_session_state(st_session_state) -> RAGDeps:
        """Create RAGDeps from Streamlit session state."""
        print("🎨 Creating Streamlit-compatible configuration...")
        
        # Extract configuration from Streamlit session state or use defaults
        collection_name = getattr(st_session_state, 'collection_name', 'default_collection')
        
        # Get ChromaDB client (typically in-memory for Streamlit)
        try:
            import chromadb
            chroma_client = chromadb.Client()
        except Exception as e:
            print(f"Warning: Could not create ChromaDB client: {e}")
            chroma_client = None
        
        # Vertex AI configuration
        vertex_project_id = getattr(st_session_state, 'vertex_project_id', None) or os.getenv('GOOGLE_CLOUD_PROJECT')
        vertex_location = getattr(st_session_state, 'vertex_location', 'us-central1')
        
        # Reranker configuration
        use_vertex_reranker = getattr(st_session_state, 'use_vertex_reranker', False)
        vertex_reranker_model = getattr(st_session_state, 'vertex_reranker_model', None)
        
        # Cache configuration optimized for Streamlit
        cache_config = PydanticAICacheConfig(
            enable_llm_cache=True,  # Always enable for Streamlit performance
            enable_embedding_cache=True,
            llm_cache_ttl_hours=2,  # Longer TTL for Streamlit sessions
            llm_cache_max_size=200,  # Smaller cache for memory efficiency
            embedding_cache_max_size=500,
            cache_hit_logging=False  # Disable verbose logging in Streamlit
        )
        
        # Batch configuration optimized for Streamlit
        batch_config = BatchProcessingConfig(
            max_batch_size=5,  # Smaller batches for responsiveness
            max_concurrent_batches=2,  # Lower concurrency for stability
            batch_timeout_seconds=20.0,  # Shorter timeout for UI responsiveness
            enable_batch_optimization=True,
            adaptive_batch_sizing=False  # Disable for predictable performance
        )
        
        deps = RAGDeps(
            chroma_client=chroma_client,
            collection_name=collection_name,
            embedding_model_name="text-embedding-004",
            embedding_provider="vertex_ai",
            vertex_project_id=vertex_project_id,
            vertex_location=vertex_location,
            use_vertex_reranker=use_vertex_reranker,
            vertex_reranker_model=vertex_reranker_model,
            cache_config=cache_config,
            batch_config=batch_config
        )
        
        print(f"✅ Streamlit configuration created:")
        print(f"  - Collection: {collection_name}")
        print(f"  - Vertex AI project: {vertex_project_id}")
        print(f"  - Cache optimized for Streamlit")
        
        return deps
    
    @staticmethod
    def get_streamlit_performance_metrics() -> dict:
        """Get performance metrics formatted for Streamlit display."""
        cache_metrics = get_cache_metrics()
        fallback_state = get_model_fallback_state()
        
        return {
            "cache_performance": {
                "total_hit_rate": f"{cache_metrics.cache_hit_rate:.1%}",
                "llm_cache_hits": cache_metrics.llm_cache_stats.get("hit_count", 0),
                "embedding_cache_hits": cache_metrics.embedding_cache_stats.get("hit_count", 0),
                "total_requests": cache_metrics.total_cache_hits + cache_metrics.total_cache_misses
            },
            "model_performance": {
                "current_provider": fallback_state.current_provider.value,
                "total_requests": fallback_state.total_requests,
                "successful_requests": fallback_state.successful_requests,
                "success_rate": f"{(fallback_state.successful_requests / max(fallback_state.total_requests, 1)):.1%}",
                "circuit_breaker_open": fallback_state.circuit_breaker_open
            }
        }

# Create the Streamlit-compatible function
streamlit_rag_agent_entrypoint = StreamlitIntegrationAdapter.create_streamlit_compatible_function()

# Backward compatibility alias for existing Streamlit code
async def run_rag_agent_entrypoint(question: str, deps: RAGDeps, llm_model: str = "gemini-2.5-flash") -> str:
    """Backward compatibility function for existing Streamlit integration."""
    return await streamlit_rag_agent_entrypoint(question, deps, llm_model)

# ===== CLI INTEGRATION =====

def update_main_cli_function():
    """Update the main CLI function to use the new structured system."""
    
    # Find and update the existing main function
    import sys
    
    # Check if this is being run as main
    if __name__ == "__main__":
        print("🖥️ CLI execution with new Pydantic AI system...")
        
        # Parse command line arguments (existing argparse logic)
        parser = argparse.ArgumentParser(description="RAG Agent with Pydantic AI")
        parser.add_argument("--question", required=True, help="Question to ask")
        parser.add_argument("--collection-name", default="default_collection", help="ChromaDB collection name")
        parser.add_argument("--db-dir", default="./chroma_db", help="ChromaDB directory")
        parser.add_argument("--embedding-model", default="text-embedding-004", help="Embedding model name")
        parser.add_argument("--vertex-project-id", help="Vertex AI project ID")
        parser.add_argument("--vertex-location", default="us-central1", help="Vertex AI location")
        parser.add_argument("--use-vertex-reranker", action="store_true", help="Use Vertex AI reranker")
        parser.add_argument("--vertex-reranker-model", help="Vertex AI reranker model")
        parser.add_argument("--llm-model", default="gemini-2.5-flash", help="LLM model to use")
        parser.add_argument("--output-format", default="text", choices=["text", "markdown", "json", "structured"], help="Output format")
        parser.add_argument("--n-results", type=int, default=10, help="Number of results to return")
        parser.add_argument("--enable-cache", action="store_true", default=True, help="Enable caching")
        parser.add_argument("--verbose", action="store_true", help="Verbose output")
        
        args = parser.parse_args()
        
        try:
            # Use the legacy-compatible entrypoint
            result = create_legacy_compatible_entrypoint(
                question=args.question,
                collection_name=args.collection_name,
                db_dir=args.db_dir,
                embedding_model=args.embedding_model,
                vertex_project_id=args.vertex_project_id,
                vertex_location=args.vertex_location,
                use_vertex_reranker=args.use_vertex_reranker,
                vertex_reranker_model=args.vertex_reranker_model,
                llm_model=args.llm_model,
                output_format=args.output_format,
                n_results=args.n_results,
                enable_cache=args.enable_cache,
                verbose=args.verbose
            )
            
            print("\n" + "="*80)
            print("RESULT:")
            print("="*80)
            print(result)
            
        except Exception as e:
            print(f"❌ CLI execution failed: {e}")
            sys.exit(1)

# Execute CLI if this is the main module
update_main_cli_function()

# ===== DOCUMENTATION AND EXAMPLES =====

"""
# Pydantic AI RAG Migration - Complete Documentation

## Overview
This module implements a complete migration from manual API calls to native Pydantic AI integration
for a Retrieval-Augmented Generation (RAG) system. The migration provides structured data models,
comprehensive error handling, and advanced caching mechanisms.

## Installation Requirements

### For GoogleModel Support (Recommended)
```bash
# Install Pydantic AI with Google support
pip install "pydantic-ai-slim[google]"

# Or install the full version
pip install pydantic-ai

# Set your Google API key
export GOOGLE_API_KEY="your-api-key"  # Get from https://aistudio.google.com
```

### For OpenAI Fallback
```bash
# OpenAI is included in the base installation
pip install pydantic-ai

# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-key"
```

### Check Installation
```python
# Run this to check your setup
check_google_model_setup()
```

## Key Features

### 1. Native Pydantic AI Integration
- GoogleModel integration with automatic OpenAI fallback
- Structured @agent.tool functions for all RAG operations
- Type-safe dependency injection with RAGDeps

### 2. Structured Data Models
- QueryVariations: Multi-query generation with adaptive strategies
- DocumentChunk/DocumentMetadata: Structured document representation
- RankedDocuments: Re-ranking results with scores and metadata
- StructuredRagAnswer: Comprehensive response format with confidence scoring

### 3. Advanced Error Handling
- RAGError: Structured error models with recovery strategies
- Circuit Breaker Pattern: Automatic model fallback with failure tracking
- Service Fallback: Graceful degradation for all critical services

### 4. Performance Optimization
- Intelligent Caching: LLM response and embedding caching with metrics
- Batch Processing: Optimized parallel processing with structured models
- Cache Integration: Native Pydantic AI cache configuration

### 5. Backward Compatibility
- LegacyConfigAdapter: Seamless migration from existing configurations
- StreamlitIntegrationAdapter: Optimized for Streamlit applications
- CLI Compatibility: Full support for existing command-line interfaces

## Usage Examples

### Basic Usage (Legacy Compatible)
```python
# Simple question answering
result = create_legacy_compatible_entrypoint(
    question="What is machine learning?",
    collection_name="ml_docs",
    output_format="text"
)
print(result)
```

### Advanced Usage (New Structured System)
```python
import asyncio

async def advanced_rag_example():
    # Create structured configuration
    cache_config = PydanticAICacheConfig(
        enable_llm_cache=True,
        llm_cache_ttl_hours=2,
        cache_hit_logging=True
    )
    
    batch_config = BatchProcessingConfig(
        max_batch_size=10,
        enable_batch_optimization=True
    )
    
    deps = RAGDeps(
        chroma_client=chromadb.Client(),
        collection_name="my_collection",
        embedding_model_name="text-embedding-004",
        embedding_provider="vertex_ai",
        vertex_project_id="my-project",
        cache_config=cache_config,
        batch_config=batch_config
    )
    
    # Create mock context
    class MockContext:
        def __init__(self, deps):
            self.deps = deps
    
    ctx = MockContext(deps)
    
    # Use the complete RAG pipeline
    result = await complete_rag_pipeline(
        ctx=ctx,
        query="Explain quantum computing",
        output_format=OutputFormat.MARKDOWN,
        n_results=15
    )
    
    return result

# Run the example
result = asyncio.run(advanced_rag_example())
```

### Streamlit Integration
```python
import streamlit as st

# Create Streamlit-optimized configuration
deps = StreamlitIntegrationAdapter.create_streamlit_deps_from_session_state(st.session_state)

# Use the Streamlit-compatible function
async def streamlit_example():
    result = await streamlit_rag_agent_entrypoint(
        question=st.text_input("Ask a question:"),
        deps=deps
    )
    return result

# Display result
if st.button("Ask"):
    result = asyncio.run(streamlit_example())
    st.write(result)
    
    # Show performance metrics
    metrics = StreamlitIntegrationAdapter.get_streamlit_performance_metrics()
    st.json(metrics)
```

### Individual Tool Usage
```python
async def tool_examples():
    # 1. Query variation generation
    query_variations = await generate_query_variations_tool(ctx, "What is AI?")
    print(f"Generated {len(query_variations.variations)} variations")
    
    # 2. Document retrieval
    retrieval_result = await retrieve_documents_structured(ctx, query_variations, 10)
    print(f"Retrieved {len(retrieval_result.ranked_documents.documents)} documents")
    
    # 3. Re-ranking with fallback
    ranked_docs = await rerank_with_service_fallback(ctx, retrieval_result, 5)
    print(f"Re-ranked to {len(ranked_docs.documents)} documents")
    
    # 4. Context formatting
    formatted_context = await format_context_tool(ctx, ranked_docs)
    print(f"Context length: {len(formatted_context)} characters")
    
    # 5. Structured response generation
    structured_answer = await generate_structured_response(
        ctx, "What is AI?", retrieval_result, ranked_docs, formatted_context
    )
    print(f"Confidence score: {structured_answer.confidence_score:.3f}")
    
    # 6. Output format switching
    text_output = await format_response_output(ctx, structured_answer, OutputFormat.TEXT)
    json_output = await format_response_output(ctx, structured_answer, OutputFormat.JSON)
    
    return text_output, json_output
```

## Configuration Options

### Cache Configuration
```python
cache_config = PydanticAICacheConfig(
    enable_llm_cache=True,              # Enable LLM response caching
    enable_embedding_cache=True,        # Enable embedding caching
    llm_cache_ttl_hours=1,             # Cache TTL in hours
    llm_cache_max_size=500,            # Max cache entries
    embedding_cache_max_size=1000,     # Max embedding cache entries
    cache_hit_logging=True             # Log cache hits/misses
)
```

### Batch Processing Configuration
```python
batch_config = BatchProcessingConfig(
    max_batch_size=10,                 # Maximum items per batch
    max_concurrent_batches=3,          # Max concurrent operations
    batch_timeout_seconds=30.0,        # Timeout per batch
    enable_batch_optimization=True,    # Enable optimizations
    adaptive_batch_sizing=True         # Auto-adjust batch sizes
)
```

### Model Fallback Configuration
```python
fallback_config = ModelFallbackConfig(
    enable_fallback=True,              # Enable automatic fallback
    primary_provider=ModelProvider.GOOGLE,     # Primary model provider
    fallback_provider=ModelProvider.OPENAI,    # Fallback provider
    max_fallback_attempts=2,           # Max fallback attempts
    circuit_breaker_threshold=3,       # Failures before circuit breaker
    circuit_breaker_reset_time=300     # Reset time in seconds
)
```

### Service Fallback Configuration
```python
service_config = ServiceFallbackConfig(
    enable_reranking_fallback=True,    # Vertex AI → score-based ranking
    enable_embedding_fallback=True,    # Embedding generation fallback
    enable_multi_query_fallback=True,  # Multi-query → single-query
    max_service_retries=2,             # Max service retries
    service_timeout_seconds=15.0,      # Service timeout
    graceful_degradation=True          # Enable graceful degradation
)
```

## Error Handling

### Structured Error Types
- `MODEL_ERROR`: LLM model failures
- `RETRIEVAL_ERROR`: Document retrieval failures
- `EMBEDDING_ERROR`: Embedding generation failures
- `RERANKING_ERROR`: Re-ranking service failures
- `VALIDATION_ERROR`: Data validation failures
- `CACHE_ERROR`: Cache operation failures
- `TIMEOUT_ERROR`: Operation timeouts
- `NETWORK_ERROR`: Network connectivity issues
- `CONFIGURATION_ERROR`: Configuration problems

### Recovery Strategies
- `RETRY`: Simple retry with backoff
- `FALLBACK_MODEL`: Switch to fallback model
- `FALLBACK_METHOD`: Use alternative method
- `SKIP_STEP`: Skip optional step
- `USE_CACHE`: Use cached result if available
- `REDUCE_COMPLEXITY`: Simplify operation
- `FAIL_GRACEFULLY`: Return partial result

## Performance Monitoring

### Cache Metrics
```python
metrics = get_cache_metrics()
print(f"Cache hit rate: {metrics.cache_hit_rate:.1%}")
print(f"LLM cache hits: {metrics.llm_cache_stats['hit_count']}")
print(f"Embedding cache hits: {metrics.embedding_cache_stats['hit_count']}")
```

### Model Fallback State
```python
state = get_model_fallback_state()
print(f"Current provider: {state.current_provider}")
print(f"Success rate: {state.successful_requests / state.total_requests:.1%}")
print(f"Circuit breaker open: {state.circuit_breaker_open}")
```

## Migration Guide

### From Legacy System
1. **Replace direct API calls** with `complete_rag_pipeline`
2. **Update configuration** using `LegacyConfigAdapter`
3. **Add error handling** with structured RAGError
4. **Enable caching** with PydanticAICacheConfig
5. **Test fallback scenarios** with different providers

### Environment Variables
```bash
# Required for GoogleModel (new format)
export GOOGLE_API_KEY="your-google-api-key"  # Get from https://aistudio.google.com
export OPENAI_API_KEY="your-openai-key"      # Fallback model
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Legacy support (will be deprecated)
export GEMINI_API_KEY="your-gemini-key"      # Use GOOGLE_API_KEY instead

# Optional
export GOOGLE_CLOUD_LOCATION="us-central1"
export COLLECTION_NAME="my_collection"
export EMBEDDING_MODEL="text-embedding-004"
export USE_VERTEX_RERANKER="true"
export VERTEX_RERANKER_MODEL="your-reranker-model"

# Cache settings
export ENABLE_LLM_CACHE="true"
export CACHE_TTL_HOURS="2"
export CACHE_MAX_SIZE="1000"

# Batch processing
export BATCH_SIZE="10"
export MAX_CONCURRENT="3"
export BATCH_TIMEOUT="30.0"
```

## Troubleshooting

### Common Issues
1. **"GoogleModel failed"**: Check GEMINI_API_KEY and fallback to OpenAI
2. **"Embedding generation failed"**: Verify GOOGLE_CLOUD_PROJECT and credentials
3. **"Circuit breaker open"**: Wait for reset or check model availability
4. **"Cache errors"**: Disable caching temporarily or check memory usage
5. **"Timeout errors"**: Increase timeout values or reduce batch sizes

### Debug Mode
```python
# Enable verbose logging
deps.cache_config.cache_hit_logging = True

# Check system state
print("Cache metrics:", get_cache_metrics())
print("Fallback state:", get_model_fallback_state())

# Validate configuration
warnings = LegacyConfigAdapter.validate_legacy_configuration(deps)
for warning in warnings:
    print(f"Warning: {warning}")
```

## API Reference

### Main Functions
- `complete_rag_pipeline()`: End-to-end RAG processing
- `create_legacy_compatible_entrypoint()`: Legacy compatibility
- `streamlit_rag_agent_entrypoint()`: Streamlit integration

### Agent Tools
- `generate_query_variations_tool()`: Multi-query generation
- `retrieve_documents_structured()`: Document retrieval
- `rerank_with_service_fallback()`: Document re-ranking
- `format_context_tool()`: Context formatting
- `generate_structured_response()`: Response generation
- `format_response_output()`: Output formatting

### Configuration Classes
- `RAGDeps`: Main dependency injection
- `PydanticAICacheConfig`: Cache configuration
- `BatchProcessingConfig`: Batch processing settings
- `ModelFallbackConfig`: Model fallback settings
- `ServiceFallbackConfig`: Service fallback settings

### Data Models
- `QueryVariations`: Query generation results
- `RetrievalResult`: Document retrieval results
- `RankedDocuments`: Re-ranking results
- `StructuredRagAnswer`: Final response format
- `RAGError`: Structured error information

This completes the comprehensive documentation for the Pydantic AI RAG Migration.
"""


def _unified_relevance_filter(docs_with_meta: list[dict], question: str, max_results: int = 15) -> tuple[list[str], list[dict]]:
    """Unified relevance filtering for both retrieval paths."""
    query_words = set(question.lower().split())
    
    # Entferne Stoppwörter für bessere Relevanz
    stop_words = {'der', 'die', 'das', 'und', 'oder', 'aber', 'ist', 'sind', 'was', 'wie', 'wo', 'wann', 'warum'}
    query_words = query_words - stop_words
    
    filtered_docs = []
    
    for doc_meta in docs_with_meta:
        doc_text = doc_meta['document'].lower()
        doc_words = set(doc_text.split()) - stop_words
        
        # 1. Word overlap ratio (most important)
        word_overlap = len(query_words.intersection(doc_words))
        word_overlap_ratio = word_overlap / max(len(query_words), 1)
        
        # 2. Exact phrase matching (weighted by word length)
        phrase_bonus = 0
        for word in query_words:
            if word in doc_text:
                word_weight = max(1, len(word) / 3)  # Longer words get higher bonus
                phrase_bonus += word_weight
        
        # 3. Question type bonus (generic)
        keyword_bonus = 0
        question_indicators = {'definition', 'was ist', 'bedeutung', 'erklärung', 'beschreibung'}
        if any(indicator in question.lower() for indicator in question_indicators):
            definition_words = {'definition', 'bedeutet', 'bezeichnet', 'versteht man', 'ist ein', 'sind'}
            for word in definition_words:
                if word in doc_text:
                    keyword_bonus += 2
        
        # 4. Document quality score
        doc_length = len(doc_text)
        if 200 <= doc_length <= 2000:  # Sweet spot
            length_bonus = 1.5
        elif doc_length < 50:  # Too short
            length_bonus = -2
        else:
            length_bonus = min(doc_length / 1000, 1.5)  # Scale with length
        
        # 5. Calculate unified score
        total_score = (
            word_overlap_ratio * 20 +  # Primary weight
            phrase_bonus * 5 +         # High bonus for exact matches
            keyword_bonus * 3 +        # Moderate bonus for question type
            length_bonus * 2           # Quality bonus
        )
        
        # Only include docs with meaningful relevance
        if total_score > 2.0:  # Higher threshold for better quality
            filtered_docs.append((doc_meta, total_score))
    
    # Sort by score and take top results
    filtered_docs.sort(key=lambda x: x[1], reverse=True)
    top_filtered = filtered_docs[:max_results]
    
    final_docs = [item[0]['document'] for item in top_filtered]
    final_metadatas = [item[0]['metadata'] for item in top_filtered]
    
    return final_docs, final_metadatas


def _format_context_parts(docs_texts: list[str], metadatas: list[dict]) -> str:
    """Formats the retrieved documents into a single context string with deduplicated numbered references using superscript numbers."""
    if len(docs_texts) != len(metadatas):
        raise ValueError("docs_texts and metadatas must have the same length")
    
    # Mapping für hochgestellte Zahlen
    superscript_map = {
        1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹', 10: '¹⁰',
        11: '¹¹', 12: '¹²', 13: '¹³', 14: '¹⁴', 15: '¹⁵', 16: '¹⁶', 17: '¹⁷', 18: '¹⁸', 19: '¹⁹', 20: '²⁰'
    }
    
    # Create a mapping of unique URLs to reference numbers
    url_to_ref_num = {}
    unique_sources = []
    context_parts = []
    
    for doc_text, metadata in zip(docs_texts, metadatas):
        source_url = metadata.get('url', 'Unknown Source')
        
        # Assign reference number (deduplicate URLs)
        if source_url not in url_to_ref_num:
            ref_num = len(url_to_ref_num) + 1
            url_to_ref_num[source_url] = ref_num
            superscript = superscript_map.get(ref_num, f"^{ref_num}")
            unique_sources.append(f"{superscript} {source_url}")
        else:
            ref_num = url_to_ref_num[source_url]
        
        # Add numbered reference to the document text with superscript
        superscript = superscript_map.get(ref_num, f"^{ref_num}")
        context_parts.append(f"[Quelle {superscript}] {doc_text}")
    
    if not context_parts:
        return "No relevant context found."
    
    # Combine context with deduplicated source references at the end
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Create clickable references for the sources section
    clickable_references = []
    for url, ref_num in url_to_ref_num.items():
        superscript = superscript_map.get(ref_num, f"^{ref_num}")
        clickable_references.append(f"[{superscript}]({url}) {url}")
    
    references_text = "\n".join(clickable_references)
    
    # Provide URL mapping for the LLM to use in responses with clickable links
    url_mapping_instructions = "ANWEISUNG FÜR ZITATIONEN: Verwende diese hochgestellten Zahlen als KLICKBARE LINKS im Text:\n"
    for url, ref_num in url_to_ref_num.items():
        superscript = superscript_map.get(ref_num, f"^{ref_num}")
        url_mapping_instructions += f"[{superscript}]({url}) für {url}\n"
    
    url_mapping_instructions += "\nWICHTIG: Verwende im Fließtext IMMER das Format [{superscript}](URL) für klickbare Quellenverweise! NIEMALS nur hochgestellte Zahlen ohne Links verwenden.\n"
    url_mapping_instructions += "KONSISTENZ: Halte die gleiche Detailtiefe und Formatierung wie in vorherigen Antworten bei. Jede Antwort soll vollständig und umfassend sein."
    
    return f"{context_text}\n\n--- QUELLENVERZEICHNIS ---\n{references_text}\n\n--- ZITATIONS-MAPPING ---\n{url_mapping_instructions}"

def format_structured_answer(structured_answer: StructuredRagAnswer) -> str:
    """Format a structured answer into a readable text format."""
    formatted_response = f"""**Zusammenfassung:**
{structured_answer.summary}

**Wichtige Details:**
"""
    
    for detail in structured_answer.key_details:
        formatted_response += f"- {detail}\n"
    
    if structured_answer.contact_info:
        formatted_response += f"\n**Kontakt:**\n{structured_answer.contact_info}\n"
    
    formatted_response += f"\n**Konfidenz:** {structured_answer.confidence_score:.1%}\n"
    
    formatted_response += "\n**Quellen:**\n"
    for i, source in enumerate(structured_answer.sources, 1):
        formatted_response += f"({i}): {source}\n"
    
    return formatted_response

async def rerank_with_vertex_ai(
    query: str,
    documents: List[dict], # Expects list of dicts with 'document' and 'metadata'
    model_name: str,
    top_n: int
) -> List[dict]:
    """Reranks documents using the Vertex AI Ranking API."""
    print(f"---> Re-ranking {len(documents)} initial candidates with Vertex AI Ranker...")
    
    if not documents:
        return []

    # Create records for the API call, keeping track of original index
    req_documents = [
        RankingRecord(id=str(i), content=doc['document']) 
        for i, doc in enumerate(documents)
    ]

    request = RankRequest(
        ranking_config=model_name, 
        query=query,
        records=req_documents,
        top_n=top_n,
        ignore_record_details_in_response=True,
    )
    
    try:
        # Instantiate the client inside the async function to attach to the correct event loop
        async with RankServiceAsyncClient() as client:
            response = await client.rank(request=request)
        
        # Map reranked results back to original documents and metadata
        reranked_docs_with_meta = []
        for record in response.records:
            original_index = int(record.id)
            original_doc = documents[original_index]
            reranked_docs_with_meta.append({
                "document": original_doc['document'],
                "metadata": original_doc['metadata'],
                "score": record.score
            })
        
        print(f"---> Top {len(reranked_docs_with_meta)} results after Vertex AI re-ranking:")
        for res_item in reranked_docs_with_meta:
            print(f"  Score: {res_item['score']:.4f}, Source: {res_item['metadata'].get('url', 'N/A')}")
            
        return reranked_docs_with_meta

    except Exception as e:
        logfire.exception("Error during Vertex AI re-ranking.", exception=e)
        raise ReRankingError(f"Failed to re-rank documents with Vertex AI: {e}") from e

@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 10) -> str:
    """Retrieve relevant documents from ChromaDB using HyDE, then re-rank them.
    
    Args:
        context: The run context containing dependencies.
        search_query: The original search query from the user.
        n_results: Final number of results to return after re-ranking (default: 10).
        
    Returns:
        Formatted context information from the re-ranked retrieved documents.
    """
    print("--- Retrieve Tool Called (HyDE + Vertex AI Re-ranking) ---")
    print(f"Original Query: '{search_query}'")
    
    # --- HyDE Step --- 
    hyde_prompt = f"Generate a detailed, plausible paragraph that directly answers the following question as if it were extracted from a relevant document or webpage. Focus on providing factual information that would typically be found in documentation, articles, or informational content. Question: {search_query}"
    hypothetical_answer = search_query
    try:
        # Use Pydantic AI Agent with GoogleModel for HyDE generation
        hypothetical_answer = await run_agent_with_fallback(
            prompt=hyde_prompt,
            deps=context.deps,
            system_prompt_override="You generate hypothetical answers for RAG retrieval. Create realistic content that could be found in documentation, articles, or informational websites. Be factual and relevant to the specific question asked."
        )
        print(f"---> Generated Hypothetical Answer: '{hypothetical_answer}'")
    except Exception as e:
        print(f"Error generating hypothetical answer: {e}. Falling back to original query.")

    # --- Initial Retrieval Step --- 
    initial_n_results = max(50, n_results * 3)  # Mehr Kandidaten für besseres Retrieval
    print(f"---> Querying ChromaDB for {initial_n_results} initial candidates...")

    # Intelligente Embedding-Auswahl basierend auf Collection-Typ
    try:
        collection = get_or_create_collection(
            client=context.deps.chroma_client,
            collection_name=context.deps.collection_name
        )
        
        # Prüfe die Collection-Dimensionen durch Abrufen eines Beispiel-Dokuments
        collection_embedding_dim = None
        try:
            sample = collection.get(limit=1, include=["embeddings"])
            if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
                collection_embedding_dim = len(sample["embeddings"][0])
                print(f"---> Detected collection embedding dimension: {collection_embedding_dim}")
        except Exception as e:
            print(f"Could not detect collection embedding dimension: {e}")
        
        # Wähle die passende Embedding-Methode
        query_embedding_for_chroma = None
        
        if collection_embedding_dim == 384:
            # Collection verwendet ChromaDB Default Embeddings
            print("---> Using ChromaDB default embeddings for query (384D)")
            # Verwende ChromaDB's eingebaute Embedding-Funktion durch query_texts
            results = collection.query(
                query_texts=[hypothetical_answer],
                n_results=initial_n_results,
                include=['metadatas', 'documents']
            )
            print("---> Query executed with ChromaDB default embeddings")
        else:
            # Collection verwendet wahrscheinlich Vertex AI Embeddings
            print("---> Using Vertex AI embeddings for query")
            
            # Check embedding cache first
            query_embedding_for_chroma = embedding_cache.get(hypothetical_answer)
            
            if query_embedding_for_chroma is None:
                # Generate consistent, normalized embedding
                query_embedding_for_chroma = get_vertex_text_embedding(
                    text=hypothetical_answer,
                    model_name=context.deps.embedding_model_name,
                    task_type="RETRIEVAL_QUERY",  # Consistent task type
                    project_id=context.deps.vertex_project_id,
                    location=context.deps.vertex_location
                )
                if query_embedding_for_chroma is None:
                    return "Error generating query embedding with Vertex AI."
                
                # Normalize embedding before caching (L2 normalization for consistency)
                import numpy as np
                query_vec = np.array(query_embedding_for_chroma)
                query_embedding_for_chroma = (query_vec / np.linalg.norm(query_vec)).tolist()
                
                # Cache the normalized embedding
                embedding_cache.store(hypothetical_answer, query_embedding_for_chroma)
                print(f"💾 Normalized embedding cached for HyDE answer")
            
            results = collection.query(
                query_embeddings=[query_embedding_for_chroma],
                n_results=initial_n_results,
                include=['metadatas', 'documents']
            )
            print("---> Query executed with Vertex AI embeddings")
            
        if not results or not results.get('ids') or not results['ids'][0]:
             return "No relevant context found."
    except Exception as e:
        raise RetrievalError(f"Failed to retrieve documents from ChromaDB: {e}") from e

    # --- Re-ranking Step ---
    initial_docs_texts = results['documents'][0]
    initial_metadatas = results['metadatas'][0]
    
    # Combine documents and metadata for easier handling
    initial_docs_with_meta = [
        {"document": doc, "metadata": meta} 
        for doc, meta in zip(initial_docs_texts, initial_metadatas)
    ]

    if context.deps.use_vertex_reranker and context.deps.vertex_reranker_model:
        reranked_results = await rerank_with_vertex_ai(
            query=search_query,
            documents=initial_docs_with_meta,
            model_name=context.deps.vertex_reranker_model,
            top_n=n_results
        )
        final_docs = [item['document'] for item in reranked_results]
        final_metadatas = [item['metadata'] for item in reranked_results]
    else:
        print("INFO: Verwende domänen-agnostische semantische Ähnlichkeit (ohne Token-Overlap-Heuristiken)")
        
        try:
            # Generate query embedding for semantic similarity
            query_embedding = embedding_cache.get(search_query)
            
            if query_embedding is None:
                query_embedding = get_vertex_text_embedding(
                    text=search_query,
                    model_name=context.deps.embedding_model_name,
                    task_type="RETRIEVAL_QUERY",
                    project_id=context.deps.vertex_project_id,
                    location=context.deps.vertex_location
                )
                if query_embedding:
                    # Normalize query embedding
                    import numpy as np
                    query_vec = np.array(query_embedding)
                    query_embedding = (query_vec / np.linalg.norm(query_vec)).tolist()
                    embedding_cache.store(search_query, query_embedding)
            
            if query_embedding is None:
                print("⚠️ Could not generate query embedding, using high-recall fallback")
                # High-recall fallback: take more documents, let re-ranking handle quality
                candidate_count = max(50, n_results * 5)
                final_docs = [item['document'] for item in initial_docs_with_meta[:candidate_count]]
                final_metadatas = [item['metadata'] for item in initial_docs_with_meta[:candidate_count]]
            else:
                # Calculate semantic similarity for all candidates
                scored_docs = []
                
                for doc_meta in initial_docs_with_meta:
                    doc_text = doc_meta['document']
                    
                    # Generate document embedding if needed
                    doc_embedding = get_vertex_text_embedding(
                        text=doc_text,
                        model_name=context.deps.embedding_model_name,
                        task_type="RETRIEVAL_DOCUMENT",
                        project_id=context.deps.vertex_project_id,
                        location=context.deps.vertex_location
                    )
                    
                    if doc_embedding is not None:
                        # Normalize document embedding
                        doc_vec = np.array(doc_embedding)
                        doc_embedding = (doc_vec / np.linalg.norm(doc_vec)).tolist()
                        
                        # Calculate cosine similarity (domain-agnostic)
                        query_vec = np.array(query_embedding)
                        doc_vec = np.array(doc_embedding)
                        similarity_score = float(np.dot(query_vec, doc_vec))
                    else:
                        # Fallback score if embedding fails
                        similarity_score = 0.1
                    
                    scored_docs.append((doc_meta, similarity_score))
                
                # Sort by semantic similarity (no arbitrary thresholds)
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                # Take top candidates for re-ranking (high recall approach)
                candidate_count = max(50, n_results * 5)
                top_candidates = scored_docs[:candidate_count]
                
                final_docs = [item[0]['document'] for item in top_candidates]
                final_metadatas = [item[0]['metadata'] for item in top_candidates]
                
                print(f"---> Semantic similarity ranking: {len(final_docs)} candidates")
                if top_candidates:
                    avg_score = sum(item[1] for item in top_candidates) / len(top_candidates)
                    print(f"---> Average similarity score: {avg_score:.3f}")
        
        except Exception as e:
            print(f"❌ Semantic similarity ranking failed: {e}")
            # Ultimate high-recall fallback
            candidate_count = max(50, n_results * 5)
            final_docs = [item['document'] for item in initial_docs_with_meta[:candidate_count]]
            final_metadatas = [item['metadata'] for item in initial_docs_with_meta[:candidate_count]]

    print("--- Context Provided to LLM ---")
    return _format_context_parts(final_docs, final_metadatas)

async def run_rag_agent_entrypoint(
    question: str,
    deps: RAGDeps,
    llm_model: str = None,
) -> str:
    """Main entry point for running the RAG agent with native Pydantic AI integration."""
    try:
        print("--- Using Pydantic AI Agent with GoogleModel/OpenAI fallback ---")
        
        # Use the native Pydantic AI agent with automatic fallback
        # The agent will automatically use the retrieve tool when needed
        result = await agent.run(question, deps=deps)
        
        # Extract the response data
        response = result.data if hasattr(result, 'data') else str(result)
        
        return response
        
    except Exception as e:
        logfire.exception("Agent execution failed", question=question, model=llm_model)
        
        # Try fallback agent if primary fails
        try:
            print("Primary agent failed, trying fallback agent...")
            result = await fallback_agent.run(question, deps=deps)
            response = result.data if hasattr(result, 'data') else str(result)
            return response
        except Exception as fallback_error:
            logfire.exception("Fallback agent also failed", question=question)
            raise RuntimeError(f"Both primary and fallback agents failed: {e}, {fallback_error}") from e

def analyze_question_complexity(question: str) -> str:
    """Analyze question complexity for adaptive query expansion."""
    word_count = len(question.split())
    question_lower = question.lower()
    
    # Simple question indicators
    simple_patterns = ['was ist', 'was bedeutet', 'wer ist', 'wo ist', 'wann ist']
    definition_words = ['definition', 'bedeutung', 'erklärung']
    
    # Complex question indicators
    complex_words = ['unterschied', 'vergleich', 'vs', 'versus', 'warum', 'wie funktioniert', 'welche arten']
    has_multiple_concepts = len([w for w in question.split() if len(w) > 6]) > 2
    has_comparison = any(word in question_lower for word in ['vs', 'versus', 'unterschied', 'vergleich'])
    has_complex_words = any(word in question_lower for word in complex_words)
    
    # Classification logic
    if word_count <= 4 and any(pattern in question_lower for pattern in simple_patterns):
        return "simple"
    elif any(word in question_lower for word in definition_words) and word_count <= 6:
        return "simple"
    elif has_comparison or has_multiple_concepts or has_complex_words or word_count > 10:
        return "complex"
    else:
        return "moderate"


async def generate_query_variations(question: str, deps: RAGDeps) -> List[str]:
    """Generate adaptive query variations based on question complexity."""
    variations = [question]  # Original query
    
    # Analyze question complexity
    complexity = analyze_question_complexity(question)
    print(f"🧠 Question complexity: {complexity}")
    
    # Adaptive variation generation
    if complexity == "simple":
        # Simple questions: Only 1 variation to avoid semantic drift
        variation_prompt = f"""Generate 1 alternative way to ask this simple question. Keep it focused and direct, don't expand the scope.
        
        Original question: {question}
        
        Return only the 1 variation without numbering or explanation."""
        max_variations = 1
    elif complexity == "moderate":
        # Moderate questions: 2 variations
        variation_prompt = f"""Generate 2 different ways to ask the same question for better search results. 
        Make them semantically different but asking for the same core information.
        
        Original question: {question}
        
        Return only the 2 variations, one per line, without numbering or explanation."""
        max_variations = 2
    else:  # complex
        # Complex questions: 3 variations for better coverage
        variation_prompt = f"""Generate 3 different ways to ask this complex question for comprehensive search results. 
        Make them semantically different but asking for the same detailed information.
        
        Original question: {question}
        
        Return only the 3 variations, one per line, without numbering or explanation."""
        max_variations = 3
    
    try:
        variations_text = await run_agent_with_fallback(
            prompt=variation_prompt,
            deps=deps,
            system_prompt_override="You generate query variations for better search coverage. Create semantically different but equivalent questions. Keep variations focused and avoid expanding the scope."
        )
        
        # Parse variations
        new_variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
        variations.extend(new_variations[:max_variations])
        
    except Exception as e:
        print(f"Error generating query variations: {e}. Using original query only.")
    
    return variations

async def retrieve_context_for_gemini(question: str, deps: RAGDeps) -> str:
    """Helper function to retrieve context for Gemini with Multi-Query Retrieval."""
    print("--- Retrieve Context for Gemini (Multi-Query + HyDE) ---")
    print(f"Original Query: '{question}'")
    
    # --- Multi-Query Generation ---
    print("🔄 Generating query variations...")
    query_variations = await generate_query_variations(question, deps)
    print(f"---> Generated {len(query_variations)} query variations")
    
    # --- Parallel HyDE + Multi-Query Retrieval ---
    print("🧠 Generating hypothetical answers for all queries...")
    
    async def process_single_query(query: str) -> Tuple[str, str]:
        """Process a single query with HyDE."""
        # Für sehr einfache Fragen, direktere HyDE-Prompts verwenden
        if len(query.split()) <= 3 and any(word in query.lower() for word in ['was ist', 'was bedeutet', 'definition']):
            hyde_prompt = f"""Generate a concise, direct definition or explanation that answers this simple question. 
            Focus on the core meaning and avoid expanding into unrelated topics.
            
            Question: {query}
            
            Provide a focused, dictionary-style answer that directly addresses what is being asked."""
        else:
            hyde_prompt = f"Generate a detailed, plausible paragraph that directly answers the following question as if it were extracted from a relevant document or webpage. Focus on providing factual information that would typically be found in documentation, articles, or informational content. Question: {query}"
        
        try:
            hypothetical_answer = await run_agent_with_fallback(
                prompt=hyde_prompt,
                deps=deps,
                system_prompt_override="You generate hypothetical answers for RAG retrieval. Create realistic content that could be found in documentation, articles, or informational websites. Be factual and relevant to the specific question asked. For simple definition questions, keep answers focused and avoid expanding scope."
            )
            return query, hypothetical_answer
        except Exception as e:
            print(f"Error generating hypothetical answer for '{query}': {e}")
            return query, query  # Fallback to original query
    
    # Process all queries in parallel
    query_hyde_pairs = await asyncio.gather(*[process_single_query(q) for q in query_variations])
    
    for original_q, hyde_answer in query_hyde_pairs:
        print(f"---> HyDE for '{original_q[:50]}...': '{hyde_answer[:100]}...'")
    
    # --- Parallel Retrieval for all HyDE answers ---
    print("🔍 Searching database with all variations...")
    
    # --- Batch Embedding Processing ---
    async def batch_generate_embeddings(hyde_answers: List[str]) -> Dict[str, List[float]]:
        """Generate embeddings for multiple HyDE answers in parallel with caching."""
        print(f"⚡ Batch processing {len(hyde_answers)} embeddings...")
        
        # Check cache for existing embeddings
        cached_embeddings = {}
        uncached_answers = []
        
        for answer in hyde_answers:
            cached_embedding = embedding_cache.get(answer)
            if cached_embedding is not None:
                cached_embeddings[answer] = cached_embedding
            else:
                uncached_answers.append(answer)
        
        print(f"---> Cache hits: {len(cached_embeddings)}, Cache misses: {len(uncached_answers)}")
        
        # Generate embeddings for uncached answers in parallel
        async def generate_single_embedding(text: str) -> Tuple[str, List[float] | None]:
            embedding = get_vertex_text_embedding(
                text=text,
                model_name=deps.embedding_model_name,
                task_type="RETRIEVAL_QUERY",
                project_id=deps.vertex_project_id,
                location=deps.vertex_location
            )
            return text, embedding
        
        # Process uncached embeddings in parallel
        if uncached_answers:
            embedding_results = await asyncio.gather(*[
                generate_single_embedding(answer) for answer in uncached_answers
            ], return_exceptions=True)
            
            # Store new embeddings in cache
            for result in embedding_results:
                if isinstance(result, tuple) and result[1] is not None:
                    text, embedding = result
                    cached_embeddings[text] = embedding
                    embedding_cache.store(text, embedding)
        
        return cached_embeddings
    
    async def search_single_hyde(hyde_answer: str, embedding_dict: Dict[str, List[float]]) -> List[Dict]:
        """Search ChromaDB with a single HyDE answer using pre-computed embeddings."""
        try:
            collection = get_or_create_collection(
                client=deps.chroma_client,
                collection_name=deps.collection_name
            )
            
            # Detect embedding dimension
            collection_embedding_dim = None
            try:
                sample = collection.get(limit=1, include=["embeddings"])
                if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
                    collection_embedding_dim = len(sample["embeddings"][0])
            except Exception:
                pass
            
            # Query based on embedding type
            if collection_embedding_dim == 384:
                # Use text-based query for sentence-transformers
                results = collection.query(
                    query_texts=[hyde_answer],
                    n_results=20,  # Weniger pro Query, da wir mehrere haben
                    include=['metadatas', 'documents']
                )
            else:
                # Use pre-computed Vertex AI embedding
                query_embedding = embedding_dict.get(hyde_answer)
                if query_embedding is None:
                    return []
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=20,
                    include=['metadatas', 'documents']
                )
            
            if not results or not results.get('ids') or not results['ids'][0]:
                return []
            
            # Convert to list of dicts
            docs_with_meta = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                docs_with_meta.append({"document": doc, "metadata": meta})
            
            return docs_with_meta
            
        except Exception as e:
            print(f"Error in search_single_hyde: {e}")
            return []
    
    # Generate embeddings in batch for all HyDE answers
    hyde_answers = [hyde_answer for _, hyde_answer in query_hyde_pairs]
    embedding_dict = await batch_generate_embeddings(hyde_answers)
    
    # Execute all searches in parallel with pre-computed embeddings
    all_search_results = await asyncio.gather(*[
        search_single_hyde(hyde_answer, embedding_dict) 
        for _, hyde_answer in query_hyde_pairs
    ])
    
    # Combine and deduplicate results
    print("🔗 Combining and deduplicating results...")
    combined_docs = []
    seen_docs = set()
    
    for search_results in all_search_results:
        for doc_meta in search_results:
            doc_text = doc_meta['document']
            # Simple deduplication based on first 100 characters
            doc_signature = doc_text[:100]
            if doc_signature not in seen_docs:
                seen_docs.add(doc_signature)
                combined_docs.append(doc_meta)
    
    print(f"---> Combined {len(combined_docs)} unique documents from {len(query_variations)} queries")
    
    if not combined_docs:
        return "No relevant context found."
    
    # --- Re-ranking Step ---
    if deps.use_vertex_reranker and deps.vertex_reranker_model:
        print("⚡ Re-ranking with Vertex AI...")
        reranked_results = await rerank_with_vertex_ai(
            query=question,
            documents=combined_docs,
            model_name=deps.vertex_reranker_model,
            top_n=15
        )
        final_docs = [item['document'] for item in reranked_results]
        final_metadatas = [item['metadata'] for item in reranked_results]
    else:
        print("INFO: Verwende erweiterte Relevanz-Filterung (ohne Vertex AI Reranker)")
        # Enhanced fallback filtering with better semantic matching
        filtered_docs = []
        query_words = set(question.lower().split())
        
        # Remove stop words
        stop_words = {'der', 'die', 'das', 'und', 'oder', 'aber', 'ist', 'sind', 'was', 'wie', 'wo', 'wann', 'warum', 'ein', 'eine', 'einen'}
        query_words = query_words - stop_words
        
        for doc_meta in combined_docs:
            doc_text = doc_meta['document'].lower()
            doc_words = set(doc_text.split()) - stop_words
            
            # 1. Base word overlap score (most important)
            word_overlap = len(query_words.intersection(doc_words))
            if len(query_words) > 0:
                word_overlap_ratio = word_overlap / len(query_words)
            else:
                word_overlap_ratio = 0
            
            # 2. Exact phrase matching in document
            phrase_score = 0
            for word in query_words:
                if word in doc_text:
                    phrase_score += 1
            
            # 3. Question type bonus (generic approach)
            question_type_bonus = 0
            if any(q_word in question.lower() for q_word in ['was ist', 'was bedeutet', 'definition', 'erkläre']):
                # Definition questions: look for explanatory language
                if any(def_word in doc_text for def_word in ['ist', 'bedeutet', 'bezeichnet', 'definition', 'erklärung']):
                    question_type_bonus += 2
            
            # 4. Document quality indicators
            quality_score = 0
            # Prefer documents with reasonable length (not too short, not too long)
            doc_length = len(doc_text)
            if 200 <= doc_length <= 2000:  # Sweet spot for informative chunks
                quality_score += 1
            elif doc_length < 50:  # Very short docs often lack context
                quality_score -= 2
            
            # 5. Calculate final score
            total_score = (
                word_overlap_ratio * 10 +  # Most important: word overlap ratio
                phrase_score * 2 +         # Exact word matches
                question_type_bonus +      # Question type relevance
                quality_score              # Document quality
            )
            
            # Only include docs with meaningful relevance
            if total_score > 1.0:  # Reasonable threshold
                filtered_docs.append((doc_meta, total_score))
        
        # Sort by score and take top 15
        filtered_docs.sort(key=lambda x: x[1], reverse=True)
        top_filtered = filtered_docs[:15]
        
        final_docs = [item[0]['document'] for item in top_filtered]
        final_metadatas = [item[0]['metadata'] for item in top_filtered]
        
        print(f"---> Enhanced filtering: {len(final_docs)} most relevant chunks")
        print(f"---> Top 3 scores: {[f'{item[1]:.1f}' for item in top_filtered[:3]]}")

    print("--- Context Provided to Gemini ---")
    return _format_context_parts(final_docs, final_metadatas)


def main():
    """Main function to parse arguments, set up dependencies, and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using ChromaDB and Vertex AI.")
    parser.add_argument("--question", required=True, help="The question to answer.")
    parser.add_argument("--collection", default="docs", help="Name of the ChromaDB collection.")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory for ChromaDB data.")
    
    # Vertex AI arguments (now mandatory)
    parser.add_argument("--vertex-project-id", default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="Google Cloud Project ID.")
    parser.add_argument("--vertex-location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"), help="Google Cloud Location.")
    parser.add_argument("--embedding-model-name", default="text-multilingual-embedding-002", help="Vertex AI embedding model name.")
    
    # Reranker arguments
    parser.add_argument("--use-vertex-reranker", action='store_true', help="Enable Vertex AI reranker.")
    parser.add_argument(
        "--vertex-reranker-model", 
        type=str, 
        default="projects/{project_id}/locations/global/rankingConfigs/default_ranking_config",
        help="The ranking config for Vertex AI Reranker. Must include {project_id} placeholder and uses 'global' location."
    )

    parser.add_argument('--n-results', type=int, default=10, help='Number of final results to pass to LLM.')
    parser.add_argument('--llm-model', type=str, default='gemini-2.5-flash', help='LLM model for generation (default: Gemini 2.5 Flash, fallback: OpenAI GPT-4.1-mini).')
    
    args = parser.parse_args()

    if not args.vertex_project_id:
        parser.error("--vertex-project-id is required.")

    print(f"CLI: Using Vertex AI: Project '{args.vertex_project_id}', Location '{args.vertex_location}'")
    init_vertex_ai(project_id=args.vertex_project_id, location=args.vertex_location)

    reranker_model_name_formatted = None
    if args.use_vertex_reranker:
        reranker_model_name_formatted = args.vertex_reranker_model.format(
            project_id=args.vertex_project_id
        )
        print(f"CLI: Vertex AI Reranker enabled. Model path: {reranker_model_name_formatted}")

    deps = RAGDeps(
        chroma_client=get_chroma_client(args.db_dir),
        collection_name=args.collection,
        embedding_model_name=args.embedding_model_name,
        embedding_provider="vertex_ai", # Hardcoded as it's the only option now
        vertex_project_id=args.vertex_project_id,
        vertex_location=args.vertex_location,
        use_vertex_reranker=args.use_vertex_reranker,

        vertex_reranker_model=reranker_model_name_formatted
    )

    response = asyncio.run(run_rag_agent_entrypoint(
        question=args.question,
        deps=deps,
        llm_model=args.llm_model
    ))
    
    print("\n--- Agent Response ---")
    print(response)

if __name__ == "__main__":
    main()