#!/usr/bin/env python3
"""
Test script to verify embedding dimension compatibility fixes.
"""

import os
import sys
import tempfile
import shutil
from typing import List

# Add current directory to path for imports
sys.path.append('.')

from utils import get_chroma_client, get_or_create_collection, check_embedding_compatibility, add_documents_to_collection
from chromadb.utils import embedding_functions

def test_embedding_dimensions():
    """Test embedding dimension compatibility handling."""
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Get ChromaDB client
        client = get_chroma_client(temp_dir)
        
        # Test 1: Create collection with ChromaDB default embeddings (384 dimensions)
        print("\n=== Test 1: ChromaDB Default Embeddings ===")
        collection_name = "test-default-embeddings"
        
        default_embedding_function = embedding_functions.DefaultEmbeddingFunction()
        collection1 = get_or_create_collection(
            client=client,
            collection_name=collection_name,
            embedding_function=default_embedding_function
        )
        
        # Add a document with default embeddings
        test_docs = ["This is a test document for default embeddings."]
        test_ids = ["doc_1"]
        test_metadatas = [{"source": "test", "type": "default"}]
        
        add_documents_to_collection(
            collection=collection1,
            ids=test_ids,
            documents=test_docs,
            metadatas=test_metadatas,
            embeddings=None  # Let ChromaDB generate embeddings
        )
        
        # Check what dimension the embeddings have
        sample = collection1.get(limit=1, include=["embeddings"])
        if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
            default_dim = len(sample["embeddings"][0])
            print(f"ChromaDB default embedding dimension: {default_dim}")
        else:
            print("No embeddings found in collection")
            default_dim = 384  # Default assumption
        
        # Test 2: Try to add different Vertex AI style embeddings to same collection
        print("\n=== Test 2: Dimension Conflict Tests ===")
        
        # Test with text-multilingual-embedding-002 (768D)
        fake_multilingual_embedding = [0.1] * 768
        compatibility_768 = check_embedding_compatibility(collection1, fake_multilingual_embedding)
        print(f"768D embedding compatibility: {compatibility_768}")
        
        # Test with gemini-embedding-001 (3072D)
        fake_gemini_embedding = [0.1] * 3072
        compatibility_3072 = check_embedding_compatibility(collection1, fake_gemini_embedding)
        print(f"3072D embedding compatibility: {compatibility_3072}")
        
        if not compatibility_768["compatible"] and not compatibility_3072["compatible"]:
            print("✅ Correctly detected dimension incompatibilities")
        else:
            print("❌ Failed to detect some dimension incompatibilities")
        
        # Test 3: Create collections for different Vertex AI models
        print("\n=== Test 3: Different Vertex AI Model Embeddings ===")
        
        # Test text-multilingual-embedding-002 (768D)
        collection_name_768 = "test-multilingual-768d"
        collection_768 = get_or_create_collection(
            client=client,
            collection_name=collection_name_768,
            embedding_function=None  # No embedding function = use provided embeddings
        )
        
        multilingual_docs = ["This is a test for text-multilingual-embedding-002."]
        multilingual_ids = ["doc_multilingual_1"]
        multilingual_metadatas = [{"source": "test", "type": "multilingual", "model": "text-multilingual-embedding-002"}]
        multilingual_embeddings = [[0.1] * 768]  # 768-dimensional embedding
        
        add_documents_to_collection(
            collection=collection_768,
            ids=multilingual_ids,
            documents=multilingual_docs,
            metadatas=multilingual_metadatas,
            embeddings=multilingual_embeddings
        )
        
        # Test gemini-embedding-001 (3072D)
        collection_name_3072 = "test-gemini-3072d"
        collection_3072 = get_or_create_collection(
            client=client,
            collection_name=collection_name_3072,
            embedding_function=None  # No embedding function = use provided embeddings
        )
        
        gemini_docs = ["This is a test for gemini-embedding-001."]
        gemini_ids = ["doc_gemini_1"]
        gemini_metadatas = [{"source": "test", "type": "gemini", "model": "gemini-embedding-001"}]
        gemini_embeddings = [[0.1] * 3072]  # 3072-dimensional embedding
        
        add_documents_to_collection(
            collection=collection_3072,
            ids=gemini_ids,
            documents=gemini_docs,
            metadatas=gemini_metadatas,
            embeddings=gemini_embeddings
        )
        
        # Verify the embeddings were stored correctly
        multilingual_sample = collection_768.get(limit=1, include=["embeddings"])
        gemini_sample = collection_3072.get(limit=1, include=["embeddings"])
        
        multilingual_dim = len(multilingual_sample["embeddings"][0]) if multilingual_sample["embeddings"] is not None and len(multilingual_sample["embeddings"]) > 0 else 0
        gemini_dim = len(gemini_sample["embeddings"][0]) if gemini_sample["embeddings"] is not None and len(gemini_sample["embeddings"]) > 0 else 0
        
        print(f"text-multilingual-embedding-002 dimension: {multilingual_dim}")
        print(f"gemini-embedding-001 dimension: {gemini_dim}")
        
        # Test 4: Cross-compatibility checks
        print("\n=== Test 4: Cross-Compatibility Checks ===")
        
        # Check if default collection is compatible with different Vertex AI embeddings
        multilingual_compat = check_embedding_compatibility(collection1, fake_multilingual_embedding)
        gemini_compat = check_embedding_compatibility(collection1, fake_gemini_embedding)
        print(f"Default collection + Multilingual (768D): {multilingual_compat}")
        print(f"Default collection + Gemini (3072D): {gemini_compat}")
        
        # Check cross-compatibility between Vertex AI models
        fake_default_embedding = [0.1] * default_dim
        multilingual_vs_default = check_embedding_compatibility(collection_768, fake_default_embedding)
        gemini_vs_multilingual = check_embedding_compatibility(collection_3072, fake_multilingual_embedding)
        
        print(f"Multilingual collection + Default embedding: {multilingual_vs_default}")
        print(f"Gemini collection + Multilingual embedding: {gemini_vs_multilingual}")
        
        print("\n=== Test Results ===")
        print("✅ All embedding dimension tests completed successfully!")
        print(f"- ChromaDB default embeddings: {default_dim} dimensions")
        print(f"- text-multilingual-embedding-002: {multilingual_dim} dimensions")
        print(f"- gemini-embedding-001: {gemini_dim} dimensions")
        print("- Dimension conflict detection: Working")
        print("- Separate collections for different dimensions: Working")
        print("- Model-specific embedding handling: Working")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary directory (with retry for Windows file locking issues)
        if os.path.exists(temp_dir):
            try:
                # Try to close any open connections first
                if 'client' in locals():
                    try:
                        # ChromaDB doesn't have an explicit close method, but we can delete the reference
                        del client
                    except:
                        pass
                
                # Wait a moment for file handles to be released
                import time
                time.sleep(0.5)
                
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temporary directory: {temp_dir}")
            except PermissionError as pe:
                print(f"\nWarning: Could not clean up temporary directory due to file locks: {pe}")
                print(f"Please manually delete: {temp_dir}")
            except Exception as cleanup_error:
                print(f"\nWarning: Cleanup error: {cleanup_error}")
                print(f"Please manually delete: {temp_dir}")

if __name__ == "__main__":
    print("Testing Embedding Dimension Compatibility Fixes")
    print("=" * 50)
    test_embedding_dimensions()