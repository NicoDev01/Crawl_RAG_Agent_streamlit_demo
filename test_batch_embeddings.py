#!/usr/bin/env python3
"""
Test script for the new batch embedding functionality.
"""

import asyncio
import time
from rag_agent import batch_generate_document_embeddings, RAGDeps, PydanticAICacheConfig
from utils import get_chroma_client

async def test_batch_embeddings():
    """Test the new batch embedding generation."""
    print("🧪 Testing batch embedding generation...")
    
    # Create test dependencies
    deps = RAGDeps(
        chroma_client=get_chroma_client("./chroma_db"),
        collection_name="test_collection",
        embedding_model_name="text-embedding-004",
        embedding_provider="vertex_ai",
        vertex_project_id="your-project-id",  # Replace with actual project ID
        vertex_location="us-central1",
        cache_config=PydanticAICacheConfig(
            enable_embedding_cache=True,
            cache_hit_logging=True
        )
    )
    
    # Test documents
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties."
    ]
    
    print(f"📝 Testing with {len(test_texts)} documents...")
    
    # Test batch generation
    start_time = time.time()
    
    embeddings = await batch_generate_document_embeddings(
        texts=test_texts,
        deps=deps,
        task_type="RETRIEVAL_DOCUMENT",
        max_concurrency=3,
        timeout_seconds=10.0
    )
    
    end_time = time.time()
    
    # Analyze results
    successful_embeddings = [e for e in embeddings if e is not None]
    success_rate = len(successful_embeddings) / len(test_texts) * 100
    
    print(f"\n📊 Test Results:")
    print(f"   Total time: {end_time - start_time:.2f}s")
    print(f"   Success rate: {success_rate:.1f}% ({len(successful_embeddings)}/{len(test_texts)})")
    print(f"   Average time per embedding: {(end_time - start_time)/len(test_texts):.3f}s")
    
    if successful_embeddings:
        embedding_dim = len(successful_embeddings[0])
        print(f"   Embedding dimension: {embedding_dim}")
        print(f"   Sample embedding (first 5 values): {successful_embeddings[0][:5]}")
    
    # Test caching by running again
    print(f"\n🔄 Testing cache performance...")
    cache_start_time = time.time()
    
    cached_embeddings = await batch_generate_document_embeddings(
        texts=test_texts,
        deps=deps,
        task_type="RETRIEVAL_DOCUMENT",
        max_concurrency=3,
        timeout_seconds=10.0
    )
    
    cache_end_time = time.time()
    cache_time = cache_end_time - cache_start_time
    
    print(f"   Cache run time: {cache_time:.2f}s")
    print(f"   Speedup: {((end_time - start_time) / cache_time):.1f}x faster")
    
    return embeddings

if __name__ == "__main__":
    # Run the test
    try:
        embeddings = asyncio.run(test_batch_embeddings())
        print("\n✅ Batch embedding test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()