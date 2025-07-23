"""Pydantic AI agent that leverages RAG with a local ChromaDB for crawled website content."""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List
from pydantic import BaseModel, Field

# Define structured output model for RAG responses
class StructuredRagAnswer(BaseModel):
    """
    Ein Modell für eine strukturierte Antwort aus dem RAG-System.
    """
    summary: str = Field(..., description="Eine prägnante Zusammenfassung der Antwort in 2-3 Sätzen.")
    key_details: List[str] = Field(..., description="Eine Liste der wichtigsten Fakten, Details oder Eigenschaften.")
    contact_info: Optional[str] = Field(None, description="Kontaktinformationen falls verfügbar (E-Mail, Telefon).")
    sources: List[str] = Field(..., description="Liste der verwendeten Quellen-URLs.")
    confidence_score: float = Field(..., ge=0, le=1, description="Konfidenzwert (0.0 bis 1.0) basierend auf der Qualität der gefundenen Dokumente.")
import asyncio
import chromadb
import logfire
import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
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

# Initialize Gemini client using google.generativeai
import google.generativeai as genai

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    print("Gemini API configured successfully")
else:
    print("Warning: GEMINI_API_KEY not found. Gemini will not be available.")

async def generate_with_gemini(prompt: str, system_prompt: str = "", project_id: str = None, location: str = "us-central1") -> str:
    """Generate text using Gemini 2.5 Flash with fallback to OpenAI."""
    try:
        # Check if Gemini API key is available
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        # Create Gemini model
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        
        # Combine system prompt and user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # Generate response (synchronous call in async wrapper)
        response = await asyncio.to_thread(model.generate_content, full_prompt)
        return response.text
        
    except Exception as e:
        print(f"Gemini generation failed: {e}. Falling back to OpenAI GPT-4.1-mini...")
        # Fallback to OpenAI
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            completion = await aclient.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0
            )
            return completion.choices[0].message.content or ""
        except Exception as fallback_error:
            print(f"OpenAI fallback also failed: {fallback_error}")
            raise RuntimeError(f"Both Gemini and OpenAI failed: {e}, {fallback_error}")

# Custom Exceptions
class RetrievalError(Exception):
    """Custom exception for errors during document retrieval."""
    pass

class ReRankingError(Exception):
    """Custom exception for errors during re-ranking."""
    pass

# Define Dependencies
@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    chroma_client: chromadb.Client
    collection_name: str
    embedding_model_name: str
    embedding_provider: str
    vertex_project_id: Optional[str] = None
    vertex_location: Optional[str] = None
    # New fields for Vertex AI Reranker
    use_vertex_reranker: bool = False
    vertex_reranker_model: Optional[str] = None

# Define the system prompt as a separate variable
SYSTEM_PROMPT_TEMPLATE = (
    "## Deine Rolle: Spezialisierter und auf Quellen basis arbeitender Q&A-Assistent\n"
    "Du bist ein hochspezialisierter KI-Assistent. Deine EINZIGE Aufgabe ist es, Fragen präzise und ausschließlich auf Basis des unten bereitgestellten Kontexts aus Webseiten oder Dokumenten zu beantworten. Du darfst unter KEINEN Umständen externes Wissen verwenden.\n\n"
    
    "## Kernanweisungen (Regeln):\n"
    "1.  **IMMER zuerst den `retrieve` Befehl nutzen:** Bevor du eine Frage beantwortest, die Informationen erfordert, MUSST du das `retrieve`-Tool verwenden, um den relevanten Kontext abzurufen. Antworte niemals aus dem Gedächtnis. Die einzige Ausnahme sind rein konversationelle Fragen (z.B. 'Hallo').\n"
    "2.  **100% kontextbasiert arbeiten:** Deine Antwort MUSS zu 100% auf den Informationen im `--- KONTEXT ---`-Block basieren. Füge keine Informationen hinzu, interpretiere nichts hinein und mache keine Annahmen.\n"
    "3.  **Umgang mit fehlenden Informationen:** Wenn der Kontext die Antwort nicht enthält, MUSST du das klar sagen. Beispiel: 'Die bereitgestellten Dokumente enthalten keine Informationen zu diesem Thema.'\n"
    "4.  **Sparsame Zitationen:** Verwende klickbare Links [(1)](URL) nur EINMAL pro Person/Thema, nicht nach jedem Satz. Platziere sie strategisch nach Namen oder wichtigen Fakten. Verwende die URLs aus dem URL_MAPPING.\n"
    "5.  **Quellen formatieren:** Am Ende MUSST du unter '**Quellen:**' (mit Zeilenumbruch) jede Quelle in einer SEPARATEN Zeile auflisten:\n"
    "**Quellen:**\n"
    "(1): https://example.com\n"
    "(2): https://example.com\n"
    "NIEMALS alle Quellen in einer Zeile!\n\n"
    
    "## Dein Arbeitsprozess:\n"
    "1.  Analysiere die Nutzerfrage.\n"
    "2.  Rufe mit dem `retrieve`-Tool den Kontext ab.\n"
    "3.  Formuliere eine Antwort *ausschließlich* basierend auf dem `--- KONTEXT ---`. Verwende für jeden Fakt nur EINE Quellenangabe (die erste relevante).\n"
    "4.  Füge am Ende unter '**Quellen:**\\n' nur die tatsächlich verwendeten Quellen als vollständige URLs auf.\n\n"
    
    "## Beispiel für korrekte Formatierung:\n"
    "**Person Name** [(1)](https://example.com/person):\n"
    "- **Position:** Anwendungsentwickler\n"
    "- **Aufgaben:** Webentwicklung, PHP, JavaScript\n"
    "- **Besonderheiten:** Lieblings-Emoji 😎, typischer Satz: 'Muss ja...'\n"
    "- **Kontakt:** email@company.com, +49 123 456789\n\n"
    "**Quellen:**\n"
    "(1): https://example.com/person\n\n"
    
    "## KRITISCH WICHTIG für Quellenangaben:\n"
    "- Verwende jede URL nur EINMAL, auch wenn sie in mehreren Dokumenten vorkommt\n"
    "- NIEMALS alle Quellen in einer Zeile schreiben!\n"
    
    "--- KONTEXT ---\n"
    "{context}\n"
    "--- END KONTEXT ---\n\n"
    
    "Beantworte nun die folgende Frage *ausschließlich* auf Basis des oben stehenden Kontexts.\n"
    "Frage: {question}"
)

# Define the Pydantic AI Agent (structured output will be handled separately)
agent = Agent(
    name="RAGAgent",
    description="Answers questions about crawled website content using Retrieval-Augmented Generation (RAG).",
    system_prompt=SYSTEM_PROMPT_TEMPLATE,
    dependencies=RAGDeps,
    model="gpt-4.1-mini",
    llm=aclient
)


def _format_context_parts(docs_texts: list[str], metadatas: list[dict]) -> str:
    """Formats the retrieved documents into a single context string with deduplicated numbered references."""
    if len(docs_texts) != len(metadatas):
        raise ValueError("docs_texts and metadatas must have the same length")
    
    # Create a mapping of unique URLs to reference numbers and store URLs for inline links
    url_to_ref_num = {}
    unique_sources = []
    context_parts = []
    
    for doc_text, metadata in zip(docs_texts, metadatas):
        source_url = metadata.get('url', 'Unknown Source')
        
        # Assign reference number (deduplicate URLs)
        if source_url not in url_to_ref_num:
            ref_num = len(url_to_ref_num) + 1
            url_to_ref_num[source_url] = ref_num
            unique_sources.append(f"({ref_num}): {source_url}")
        else:
            ref_num = url_to_ref_num[source_url]
        
        # Add numbered reference to the document text
        context_parts.append(f"[Quelle {ref_num}] {doc_text}")
    
    if not context_parts:
        return "No relevant context found."
    
    # Combine context with deduplicated source references at the end
    context_text = "\n\n---\n\n".join(context_parts)
    references_text = "\n".join(unique_sources)
    
    # Also provide URL mapping for inline citations
    url_mapping_text = "\n".join([f"QUELLE_{ref_num}_URL: {url}" for url, ref_num in url_to_ref_num.items()])
    
    return f"{context_text}\n\n--- QUELLENVERZEICHNIS ---\n{references_text}\n\n--- URL_MAPPING ---\n{url_mapping_text}"

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
        # Use Gemini 2.5 Flash for HyDE with fallback to OpenAI
        hypothetical_answer = await generate_with_gemini(
            prompt=hyde_prompt,
            system_prompt="You generate hypothetical answers for RAG retrieval. Create realistic content that could be found in documentation, articles, or informational websites. Be factual and relevant to the specific question asked.",
            project_id=context.deps.vertex_project_id,
            location=context.deps.vertex_location
        )
        print(f"---> Generated Hypothetical Answer: '{hypothetical_answer}'")
    except Exception as e:
        print(f"Error generating hypothetical answer: {e}. Falling back to original query.")

    # --- Initial Retrieval Step --- 
    initial_n_results = max(25, n_results * 2)
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
            query_embedding_for_chroma = get_vertex_text_embedding(
                text=hypothetical_answer,
                model_name=context.deps.embedding_model_name,
                task_type="RETRIEVAL_QUERY",
                project_id=context.deps.vertex_project_id,
                location=context.deps.vertex_location
            )
            if query_embedding_for_chroma is None:
                return "Error generating query embedding with Vertex AI."
            
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
        print("WARNUNG: Vertex AI Re-ranker nicht konfiguriert. Überspringe Re-Ranking.")
        # Fallback: Use the top N results from the initial search
        top_n_initial = initial_docs_with_meta[:n_results]
        final_docs = [item['document'] for item in top_n_initial]
        final_metadatas = [item['metadata'] for item in top_n_initial]

    print("--- Context Provided to LLM ---")
    return _format_context_parts(final_docs, final_metadatas)

async def run_rag_agent_entrypoint(
    question: str,
    deps: RAGDeps,
    llm_model: str,
) -> str:
    """Main entry point for running the RAG agent."""
    try:
        # Use Gemini 2.5 Flash for the main response
        if llm_model.startswith("gemini"):
            # Get context using retrieve tool manually
            from pydantic_ai import RunContext
            
            # Create a mock RunContext for the retrieve function
            # We'll call retrieve directly with the search query
            print("--- Using Gemini 2.5 Flash for main response ---")
            
            # Step 1: Get context using HyDE + retrieval
            context_result = await retrieve_context_for_gemini(question, deps)
            
            # Step 2: Format the system prompt with context
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_result, question=question)
            
            # Step 3: Generate normal response with Gemini (no JSON)
            response = await generate_with_gemini(
                prompt=question,
                system_prompt=system_prompt,
                project_id=deps.vertex_project_id,
                location=deps.vertex_location
            )
            return response
        else:
            # Use the original Pydantic AI agent for non-Gemini models
            result = await agent.run(question, deps=deps, model=llm_model)
            return result.data
    except Exception as e:
        logfire.exception("Agent execution failed", question=question, model=llm_model)
        raise RuntimeError(f"Agent execution failed: {e}") from e

async def retrieve_context_for_gemini(question: str, deps: RAGDeps) -> str:
    """Helper function to retrieve context for Gemini without RunContext."""
    print("--- Retrieve Context for Gemini ---")
    print(f"Original Query: '{question}'")
    
    # --- HyDE Step --- 
    hyde_prompt = f"Generate a detailed, plausible paragraph that directly answers the following question as if it were extracted from a relevant document or webpage. Focus on providing factual information that would typically be found in documentation, articles, or informational content. Question: {question}"
    hypothetical_answer = question
    try:
        # Use Gemini 2.5 Flash for HyDE
        hypothetical_answer = await generate_with_gemini(
            prompt=hyde_prompt,
            system_prompt="You generate hypothetical answers for RAG retrieval. Create realistic content that could be found in documentation, articles, or informational websites. Be factual and relevant to the specific question asked.",
            project_id=deps.vertex_project_id,
            location=deps.vertex_location
        )
        print(f"---> Generated Hypothetical Answer: '{hypothetical_answer}'")
    except Exception as e:
        print(f"Error generating hypothetical answer: {e}. Falling back to original query.")

    # --- Initial Retrieval Step --- 
    initial_n_results = 25
    print(f"---> Querying ChromaDB for {initial_n_results} initial candidates...")

    # Intelligente Embedding-Auswahl basierend auf Collection-Typ
    try:
        collection = get_or_create_collection(
            client=deps.chroma_client,
            collection_name=deps.collection_name
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
            query_embedding_for_chroma = None  # Signal für query_texts usage
        else:
            # Collection verwendet wahrscheinlich Vertex AI Embeddings
            print("---> Using Vertex AI embeddings for query")
            query_embedding_for_chroma = get_vertex_text_embedding(
                text=hypothetical_answer,
                model_name=deps.embedding_model_name,
                task_type="RETRIEVAL_QUERY",
                project_id=deps.vertex_project_id,
                location=deps.vertex_location
            )
            if query_embedding_for_chroma is None:
                return "Error generating query embedding with Vertex AI."
        
    except Exception as e:
        raise RetrievalError(f"Failed to access collection for embedding detection: {e}") from e

    try:
        # Query-Ausführung basierend auf Embedding-Typ
        if query_embedding_for_chroma is None:
            # Verwende ChromaDB Default Embeddings
            results = collection.query(
                query_texts=[hypothetical_answer],
                n_results=initial_n_results,
                include=['metadatas', 'documents']
            )
            print("---> Query executed with ChromaDB default embeddings")
        else:
            # Verwende Vertex AI Embeddings
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

    if deps.use_vertex_reranker and deps.vertex_reranker_model:
        reranked_results = await rerank_with_vertex_ai(
            query=question,
            documents=initial_docs_with_meta,
            model_name=deps.vertex_reranker_model,
            top_n=10
        )
        final_docs = [item['document'] for item in reranked_results]
        final_metadatas = [item['metadata'] for item in reranked_results]
    else:
        print("WARNUNG: Vertex AI Re-ranker nicht konfiguriert. Überspringe Re-Ranking.")
        # Fallback: Use the top N results from the initial search
        top_n_initial = initial_docs_with_meta[:10]
        final_docs = [item['document'] for item in top_n_initial]
        final_metadatas = [item['metadata'] for item in top_n_initial]

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