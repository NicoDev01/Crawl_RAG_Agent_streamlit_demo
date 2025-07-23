"""Utility functions for text processing and ChromaDB operations."""

import os
from typing import List, Dict, Any, Optional

import chromadb



def get_chroma_client(persist_directory: str) -> chromadb.PersistentClient:
    """Get a ChromaDB client with the specified persistence directory."""
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)


def get_or_create_collection(
    client,  # Kann PersistentClient oder Client sein
    collection_name: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    distance_function: str = "cosine",
    embedding_function: Optional[Any] = None
) -> chromadb.Collection:
    """Get an existing collection or create a new one if it doesn't exist.
    
    Args:
        client: ChromaDB client (PersistentClient or Client)
        collection_name: Name of the collection
        embedding_model_name: Name of the embedding model (legacy parameter)
        distance_function: Distance function for similarity search
        embedding_function: ChromaDB embedding function to use. If None, ChromaDB will
                          expect embeddings to be provided when adding documents.
    
    Returns:
        ChromaDB Collection object
    """
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Successfully loaded existing collection: '{collection_name}'.")
        return collection
    except Exception:
        print(f"Collection '{collection_name}' not found. Creating a new one.")
        
        # Prepare collection creation parameters
        create_params = {
            "name": collection_name,
            "metadata": {"hnsw:space": distance_function}
        }
        
        # Only add embedding_function if it's not None
        if embedding_function is not None:
            create_params["embedding_function"] = embedding_function
            print(f"Creating collection with embedding function: {type(embedding_function).__name__}")
        else:
            print("Creating collection without embedding function (will use provided embeddings)")
        
        collection = client.create_collection(**create_params)
        print(f"Successfully created new collection: '{collection_name}'.")
        return collection


def add_documents_to_collection(
    collection: chromadb.Collection,
    ids: List[str],  # Changed from generating IDs to accepting them
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
    batch_size: int = 100
):
    """Add documents to a ChromaDB collection in batches."""
    if not documents:
        print("No documents to add.")
        return

    total_docs = len(documents)

    for i, batch_start in enumerate(range(0, total_docs, batch_size)):
        batch_end = min(batch_start + batch_size, total_docs)
        print(f"Adding batch {i+1}: documents {batch_start} to {batch_end-1}")

        batch_docs = documents[batch_start:batch_end]
        batch_metadatas = metadatas[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]
        
        batch_embeddings = None
        if embeddings:
            batch_embeddings = embeddings[batch_start:batch_end]

        collection.add(
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids,
            embeddings=batch_embeddings
        )
    print(f"Successfully added {total_docs} documents to the collection.")


def query_collection(
    collection: chromadb.Collection,
    query_texts: List[str],
    n_results: int = 5,
    query_embeddings: Optional[List[List[float]]] = None
) -> Dict[str, Any]:
    """Query a ChromaDB collection."""
    results = collection.query(
        query_texts=query_texts,
        n_results=n_results,
        query_embeddings=query_embeddings
    )
    return results

import shutil

def check_embedding_compatibility(collection: chromadb.Collection, test_embedding: Optional[List[float]] = None) -> Dict[str, Any]:
    """Check if a collection is compatible with given embedding dimensions.
    
    Args:
        collection: ChromaDB collection to test
        test_embedding: Optional test embedding to check compatibility
    
    Returns:
        Dictionary with compatibility info
    """
    try:
        # Get collection info
        collection_count = collection.count()
        
        if collection_count == 0:
            return {
                "compatible": True,
                "reason": "Empty collection, no dimension constraints",
                "collection_count": 0,
                "expected_dimension": None
            }
        
        # Get a sample document to check dimensions
        sample = collection.get(limit=1, include=["embeddings"])
        
        if sample["embeddings"] is None or len(sample["embeddings"]) == 0:
            return {
                "compatible": True,
                "reason": "No embeddings in collection",
                "collection_count": collection_count,
                "expected_dimension": None
            }
        
        existing_dimension = len(sample["embeddings"][0])
        
        if test_embedding is None:
            return {
                "compatible": True,
                "reason": "No test embedding provided",
                "collection_count": collection_count,
                "expected_dimension": existing_dimension
            }
        
        test_dimension = len(test_embedding)
        compatible = existing_dimension == test_dimension
        
        return {
            "compatible": compatible,
            "reason": f"Existing: {existing_dimension}D, Test: {test_dimension}D",
            "collection_count": collection_count,
            "expected_dimension": existing_dimension,
            "test_dimension": test_dimension
        }
        
    except Exception as e:
        return {
            "compatible": False,
            "reason": f"Error checking compatibility: {str(e)}",
            "collection_count": 0,
            "expected_dimension": None
        }

def delete_collection(client: chromadb.PersistentClient, collection_name: str, db_dir: str):
    """Deletes a collection from ChromaDB and its corresponding data directory."""
    try:
        # Get collection to find its UUID, which corresponds to the folder name
        collection = client.get_collection(name=collection_name)
        collection_uuid = collection.id
        
        # Construct the physical directory path
        collection_dir = os.path.join(db_dir, str(collection_uuid))
        
        # First, delete the physical directory
        if os.path.exists(collection_dir):
            shutil.rmtree(collection_dir)
            print(f"Successfully deleted data directory: {collection_dir}")
        else:
            print(f"WARN: Data directory not found for collection '{collection_name}' at {collection_dir}")

        # Then, delete the collection from Chroma's metadata
        client.delete_collection(name=collection_name)
        print(f"Successfully deleted collection '{collection_name}' from ChromaDB metadata.")

    except Exception as e:
        print(f"Error deleting collection '{collection_name}': {e}")
        raise e

def rename_collection(client: chromadb.PersistentClient, old_name: str, new_name: str):
    """Renames a collection by copying data to a new one and deleting the old one."""
    try:
        # Validate that the new name doesn't already exist
        existing_collections = [c.name for c in client.list_collections()]
        if new_name in existing_collections:
            raise ValueError(f"Collection with name '{new_name}' already exists.")

        # 1. Get the old collection
        old_collection = client.get_collection(name=old_name)
        
        # 2. Get all data from the old collection
        data = old_collection.get(include=["documents", "metadatas", "embeddings"])
        
        # 3. Create the new collection
        new_collection = client.create_collection(name=new_name)
        
        # 4. Add data to the new collection
        if data and data['ids']:
            # Convert numpy arrays to lists if necessary
            embeddings_list = data['embeddings']
            if embeddings_list and hasattr(embeddings_list, 'tolist'):
                embeddings_list = embeddings_list.tolist()
            elif embeddings_list:
                embeddings_list = [list(e) for e in embeddings_list]

            add_documents_to_collection(
                collection=new_collection,
                ids=data['ids'],
                documents=data['documents'],
                embeddings=embeddings_list,
                metadatas=data['metadatas']
            )
        
        # 5. Delete the old collection
        client.delete_collection(name=old_name)
        print(f"Successfully renamed collection from '{old_name}' to '{new_name}'.")
        
    except Exception as e:
        print(f"Error renaming collection from '{old_name}' to '{new_name}': {e}")
        # Clean up the new collection if it was created but the process failed
        try:
            client.delete_collection(name=new_name)
            print(f"Cleaned up partially created collection '{new_name}'.")
        except Exception as cleanup_e:
            print(f"Error during cleanup of '{new_name}': {cleanup_e}")
        raise e
