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


import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

async def add_documents_to_collection_async(
    collection: chromadb.Collection,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
    initial_batch_size: int = 100,
    max_parallel_batches: int = 2,  # CLOUD-OPTIMIERT: Reduziert für Streamlit Cloud Stabilität
    turbo_mode: bool = True  # NEU: Turbo-Mode für maximale Performance
):
    """Add documents to ChromaDB collection with async parallel processing and adaptive batch sizing."""
    if not documents:
        print("No documents to add.")
        return

    total_docs = len(documents)
    successfully_added = 0
    failed_batches = []
    current_batch_size = initial_batch_size

    print(f"🚀 Starting ASYNC batch insertion: {total_docs} documents")
    print(f"📊 Initial batch size: {current_batch_size}, Max parallel: {max_parallel_batches}")

    async def process_single_batch(batch_start: int, batch_size: int, batch_num: int) -> tuple:
        """Process a single batch asynchronously."""
        batch_end = min(batch_start + batch_size, total_docs)
        batch_size_actual = batch_end - batch_start
        
        try:
            # Prepare batch data
            batch_docs = documents[batch_start:batch_end]
            batch_metadatas = metadatas[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end] if embeddings else None
            
            # Validate batch data
            if len(batch_docs) != len(batch_metadatas) or len(batch_docs) != len(batch_ids):
                raise ValueError(f"Batch {batch_num}: Data length mismatch")
            
            # Use ThreadPoolExecutor to run ChromaDB operation in thread
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                await loop.run_in_executor(
                    executor,
                    lambda: collection.add(
                        documents=batch_docs,
                        metadatas=batch_metadatas,
                        ids=batch_ids,
                        embeddings=batch_embeddings
                    )
                )
            
            return batch_num, batch_size_actual, None  # Success
            
        except Exception as e:
            return batch_num, batch_size_actual, str(e)  # Error

    # Adaptive batch processing with automatic size reduction
    batch_start = 0
    batch_num = 1
    
    while batch_start < total_docs:
        # Create batch tasks for parallel processing
        batch_tasks = []
        current_parallel_count = 0
        temp_batch_start = batch_start
        
        # Create up to max_parallel_batches tasks
        while (temp_batch_start < total_docs and 
               current_parallel_count < max_parallel_batches):
            
            batch_end = min(temp_batch_start + current_batch_size, total_docs)
            if batch_end > temp_batch_start:  # Only create task if there are documents
                task = process_single_batch(temp_batch_start, current_batch_size, batch_num)
                batch_tasks.append(task)
                
                temp_batch_start = batch_end
                batch_num += 1
                current_parallel_count += 1
        
        if not batch_tasks:
            break
            
        # Execute batches in parallel with MINIMAL logging
        if batch_num <= 4 or batch_num % 25 == 0:
            print(f"🔄 Processing {len(batch_tasks)} parallel batches starting from batch {batch_num - len(batch_tasks)}")
        
        start_time = time.time()
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        batch_time = time.time() - start_time
        
        # Process results
        batch_errors = []
        for result in results:
            if isinstance(result, Exception):
                batch_errors.append(str(result))
                continue
                
            batch_num_result, batch_size_actual, error = result
            if error:
                batch_errors.append(f"Batch {batch_num_result}: {error}")
                failed_batches.append({
                    'batch_num': batch_num_result,
                    'error': error
                })
            else:
                successfully_added += batch_size_actual
                # MINIMAL Logging für maximale Performance
                if batch_num_result <= 3 or batch_num_result % 50 == 0:
                    print(f"✅ Batch {batch_num_result} completed ({batch_size_actual} docs)")
        
        # Health-Check-freundliche Pause alle 5 Batch-Gruppen
        if batch_num % 5 == 0:
            time.sleep(0.1)  # Kurze Pause für Streamlit Health-Checks
        
        # CLOUD-OPTIMIZED: Conservative batch size adjustment für Streamlit Cloud
        if batch_errors:
            if current_batch_size > 50:  # ERHÖHT: Minimum 50 statt 25
                current_batch_size = max(50, current_batch_size // 2)
                print(f"⚠️ Errors detected, reducing batch size to {current_batch_size}")
            else:
                print(f"❌ Errors persist with minimum batch size: {batch_errors[:2]}")
        elif turbo_mode and batch_time < 0.8 and current_batch_size < initial_batch_size * 1.5:
            # CLOUD-OPTIMIZED: Moderatere Erhöhung für Health-Check-Kompatibilität
            current_batch_size = min(initial_batch_size * 1.5, int(current_batch_size * 1.3))
            print(f"⚡ CLOUD-OPTIMIZED: Fast processing detected, increasing batch size to {current_batch_size}")
        elif batch_time < 1.5 and current_batch_size < initial_batch_size:
            # Standard: Moderate Erhöhung
            current_batch_size = min(initial_batch_size, int(current_batch_size * 1.2))
            print(f"⚡ Fast processing detected, increasing batch size to {current_batch_size}")
        
        # Update batch_start for next iteration
        batch_start = temp_batch_start
        
        # MINIMAL Progress updates für maximale Performance
        if batch_num % 100 == 0:
            progress_pct = (successfully_added / total_docs) * 100
            print(f"📊 Progress: {successfully_added}/{total_docs} documents ({progress_pct:.1f}%) - {batch_time:.2f}s for {len(batch_tasks)} batches")

def add_documents_to_collection(
    collection: chromadb.Collection,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
    batch_size: int = 100
):
    """Synchronous wrapper for async batch insertion (for backward compatibility)."""
    return asyncio.run(add_documents_to_collection_async(
        collection, ids, documents, metadatas, embeddings, batch_size, max_parallel_batches=3
    ))

# Legacy function for compatibility (keeping the old implementation as fallback)
def add_documents_to_collection_sync(
    collection: chromadb.Collection,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
    batch_size: int = 100
):
    """Original synchronous batch insertion (fallback)."""
    if not documents:
        print("No documents to add.")
        return

    total_docs = len(documents)
    successfully_added = 0
    failed_batches = []

    print(f"📦 Starting SYNC batch insertion: {total_docs} documents in batches of {batch_size}")

    for i, batch_start in enumerate(range(0, total_docs, batch_size)):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_num = i + 1
        batch_size_actual = batch_end - batch_start
        
        # ULTRA-REDUZIERTE Logging-Frequenz für maximale Performance
        if batch_num <= 3 or batch_num % 25 == 0 or batch_num == total_docs // batch_size:
            print(f"📥 Adding batch {batch_num}: documents {batch_start} to {batch_end-1} ({batch_size_actual} docs)")
        elif batch_num % 100 == 0:
            print(f"📊 Progress: Batch {batch_num} - {successfully_added + batch_size_actual}/{total_docs} documents processed")
        
        # Health-Check-freundliche Pause alle 10 Batches
        if batch_num % 10 == 0:
            time.sleep(0.05)  # Sehr kurze Pause für Streamlit Health-Checks

        try:
            batch_docs = documents[batch_start:batch_end]
            batch_metadatas = metadatas[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            
            # Validate batch data
            if len(batch_docs) != len(batch_metadatas) or len(batch_docs) != len(batch_ids):
                raise ValueError(f"Batch {batch_num}: Mismatched lengths - docs: {len(batch_docs)}, metadatas: {len(batch_metadatas)}, ids: {len(batch_ids)}")
            
            batch_embeddings = None
            if embeddings:
                batch_embeddings = embeddings[batch_start:batch_end]
                if len(batch_embeddings) != len(batch_docs):
                    raise ValueError(f"Batch {batch_num}: Embedding count mismatch - embeddings: {len(batch_embeddings)}, docs: {len(batch_docs)}")

            # Attempt to add the batch
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=batch_embeddings
            )
            
            successfully_added += batch_size_actual
            
            # Reduzierte Logging und Collection-Count-Checks für Performance
            if batch_num <= 5 or batch_num % 20 == 0 or batch_num == total_docs // batch_size:
                print(f"✅ Batch {batch_num} added successfully ({batch_size_actual} docs)")
                # Verify the addition by checking collection count (nur gelegentlich)
                current_count = collection.count()
                print(f"📊 Collection now contains {current_count} documents")
            
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {str(e)}")
            failed_batches.append({
                'batch_num': batch_num,
                'start': batch_start,
                'end': batch_end,
                'error': str(e)
            })
            
            # Try smaller sub-batches for failed batch
            if batch_size_actual > 1:
                print(f"🔄 Retrying batch {batch_num} with smaller sub-batches...")
                sub_batch_size = max(1, batch_size_actual // 4)  # Quarter size
                
                for sub_start in range(batch_start, batch_end, sub_batch_size):
                    sub_end = min(sub_start + sub_batch_size, batch_end)
                    
                    try:
                        sub_docs = documents[sub_start:sub_end]
                        sub_metadatas = metadatas[sub_start:sub_end]
                        sub_ids = ids[sub_start:sub_end]
                        sub_embeddings = embeddings[sub_start:sub_end] if embeddings else None
                        
                        collection.add(
                            documents=sub_docs,
                            metadatas=sub_metadatas,
                            ids=sub_ids,
                            embeddings=sub_embeddings
                        )
                        
                        successfully_added += len(sub_docs)
                        print(f"✅ Sub-batch {sub_start}-{sub_end-1} added successfully")
                        
                    except Exception as sub_e:
                        print(f"❌ Sub-batch {sub_start}-{sub_end-1} also failed: {str(sub_e)}")
                        continue

    # Final verification
    final_count = collection.count()
    print(f"🎯 === BATCH INSERTION SUMMARY ===")
    print(f"📊 Total documents processed: {total_docs}")
    print(f"✅ Successfully added: {successfully_added}")
    print(f"❌ Failed batches: {len(failed_batches)}")
    print(f"📈 Final collection count: {final_count}")
    print(f"🎯 Success rate: {(successfully_added/total_docs*100):.1f}%")
    
    if failed_batches:
        print(f"⚠️ Failed batch details:")
        for failed in failed_batches:
            print(f"  • Batch {failed['batch_num']} ({failed['start']}-{failed['end']}): {failed['error']}")
    
    if successfully_added < total_docs:
        print(f"⚠️ WARNING: Only {successfully_added}/{total_docs} documents were successfully added!")
    else:
        print(f"🎉 All {total_docs} documents successfully added to the collection!")


def verify_collection_integrity(collection: chromadb.Collection, expected_count: int) -> bool:
    """Verify that the collection contains the expected number of documents."""
    try:
        actual_count = collection.count()
        print(f"🔍 Collection integrity check:")
        print(f"  • Expected documents: {expected_count}")
        print(f"  • Actual documents: {actual_count}")
        
        if actual_count == expected_count:
            print(f"✅ Collection integrity verified!")
            return True
        else:
            print(f"⚠️ Collection integrity issue: {actual_count}/{expected_count} documents")
            
            # Try to get sample documents to diagnose
            try:
                sample = collection.get(limit=5, include=["documents", "metadatas"])
                if sample and sample.get("documents"):
                    print(f"📄 Sample documents found: {len(sample['documents'])}")
                    for i, doc in enumerate(sample["documents"][:3]):
                        print(f"  • Doc {i+1}: {doc[:100]}...")
                else:
                    print(f"❌ No documents found in collection!")
            except Exception as e:
                print(f"❌ Error sampling collection: {e}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error checking collection integrity: {e}")
        return False

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
