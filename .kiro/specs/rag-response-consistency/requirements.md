1. Grundlegende Struktur und Imports
# Core-Bibliotheken
import os, sys, argparse, hashlib, time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from pydantic import BaseModel, Field
Was passiert: Grundlegende Python-Bibliotheken für Dateisystem, Kommandozeile, Hashing und Typisierung.

2. Datenmodelle (Pydantic)
class StructuredRagAnswer(BaseModel):
    summary: str = Field(..., description="Eine prägnante Zusammenfassung der Antwort in 2-3 Sätzen.")
    key_details: List[str] = Field(..., description="Eine Liste der wichtigsten Fakten, Details oder Eigenschaften.")
    contact_info: Optional[str] = Field(None, description="Kontaktinformationen falls verfügbar (E-Mail, Telefon).")
    sources: List[str] = Field(..., description="Liste der verwendeten Quellen-URLs.")
    confidence_score: float = Field(..., ge=0, le=1, description="Konfidenzwert (0.0 bis 1.0) basierend auf der Qualität der gefundenen Dokumente.")
Zweck: Strukturierte Ausgabe für RAG-Antworten mit Validierung durch Pydantic.

3. KI-Bibliotheken und Clients
import asyncio
import chromadb  # Vektor-Datenbank
import logfire   # Logging
import dotenv    # Environment Variables
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI  # OpenAI Client
import google.generativeai as genai  # Gemini Client
Technologien:

ChromaDB: Vektor-Datenbank für Embeddings
Pydantic AI: Framework für KI-Agenten
OpenAI: GPT-Modelle als Fallback
Google Gemini: Primäres LLM (2.5 Flash)
4. Vertex AI Integration
from google.cloud.discoveryengine_v1.services.rank_service import RankServiceAsyncClient
from google.cloud.discoveryengine_v1.types import RankRequest, RankingRecord
from vertex_ai_utils import get_vertex_text_embedding, init_vertex_ai
Zweck: Google Cloud Vertex AI für Embeddings und Re-Ranking.

5. Cache-Systeme (Optimierung)
LLM Response Cache
class LLMResponseCache:
    def __init__(self, ttl_hours: int = 24, max_size: int = 500):
        self.cache = {}  # content_hash -> (response, timestamp, access_count)
        self.ttl = ttl_hours * 3600
        self.max_size = max_size
Logik:

Cached LLM-Antworten basierend auf Frage + Kontext
TTL (Time-To-Live) von 24 Stunden
LRU (Least Recently Used) Eviction
Normalisiert Kontext (entfernt URLs, Timestamps)
Query Embedding Cache
class QueryEmbeddingCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}  # query_hash -> (embedding, timestamp, access_count)
Zweck: Vermeidet redundante Embedding-API-Calls für gleiche Queries.

6. Gemini Integration mit Fallback
async def generate_with_gemini(prompt: str, system_prompt: str = "", project_id: str = None, location: str = "us-central1") -> str:
    try:
        # Gemini 2.5 Flash Konfiguration
        generation_config = genai.types.GenerationConfig(
            temperature=0.4,
            max_output_tokens=2048,
            top_p=0.85,
            top_k=40
        )
        
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        response = await asyncio.to_thread(model.generate_content, full_prompt)
        return response.text
        
    except Exception as e:
        # Fallback zu OpenAI GPT-4.1-mini
        completion = await aclient.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0
        )
        return completion.choices[0].message.content
Strategie: Primär Gemini, bei Fehlern automatischer Fallback zu OpenAI.

7. Dependencies und Agent Setup
@dataclass
class RAGDeps:
    chroma_client: chromadb.Client
    collection_name: str
    embedding_model_name: str
    embedding_provider: str
    vertex_project_id: Optional[str] = None
    vertex_location: Optional[str] = None
    use_vertex_reranker: bool = False
    vertex_reranker_model: Optional[str] = None

agent = Agent(
    name="RAGAgent",
    description="Answers questions about crawled website content using Retrieval-Augmented Generation (RAG).",
    system_prompt=SYSTEM_PROMPT_TEMPLATE,
    dependencies=RAGDeps,
    model="gpt-4.1-mini",
    llm=aclient
)
Zweck: Dependency Injection für den Pydantic AI Agent.

8. Relevanz-Filterung
def _unified_relevance_filter(docs_with_meta: list[dict], question: str, max_results: int = 15) -> tuple[list[str], list[dict]]:
    query_words = set(question.lower().split())
    stop_words = {'der', 'die', 'das', 'und', 'oder', 'aber', 'ist', 'sind', 'was', 'wie', 'wo', 'wann', 'warum'}
    query_words = query_words - stop_words
    
    for doc_meta in docs_with_meta:
        # 1. Word overlap ratio (most important)
        word_overlap = len(query_words.intersection(doc_words))
        word_overlap_ratio = word_overlap / max(len(query_words), 1)
        
        # 2. Exact phrase matching
        # 3. Question type bonus
        # 4. Document quality score
        # 5. Calculate unified score
        
        total_score = (
            word_overlap_ratio * 20 +
            phrase_bonus * 5 +
            keyword_bonus * 3 +
            length_bonus * 2
        )
Algorithmus: Multi-Kriterien-Scoring für Dokumentrelevanz.

9. Vertex AI Re-Ranking
async def rerank_with_vertex_ai(query: str, documents: List[dict], model_name: str, top_n: int) -> List[dict]:
    req_documents = [
        RankingRecord(id=str(i), content=doc['document']) 
        for i, doc in enumerate(documents)
    ]

    request = RankRequest(
        ranking_config=model_name, 
        query=query,
        records=req_documents,
        top_n=top_n,
    )
    
    async with RankServiceAsyncClient() as client:
        response = await client.rank(request=request)
Zweck: Professionelles Re-Ranking durch Google Cloud Discovery Engine.

10. HyDE (Hypothetical Document Embeddings)
# HyDE Step
hyde_prompt = f"Generate a detailed, plausible paragraph that directly answers the following question as if it were extracted from a relevant document or webpage. Question: {search_query}"

hypothetical_answer = await generate_with_gemini(
    prompt=hyde_prompt,
    system_prompt="You generate hypothetical answers for RAG retrieval..."
)
Konzept: Generiert hypothetische Antworten, um bessere Embeddings für die Suche zu erstellen.

11. Multi-Query Retrieval
async def generate_query_variations(question: str, deps: RAGDeps) -> List[str]:
    complexity = analyze_question_complexity(question)
    
    if complexity == "simple":
        max_variations = 1
    elif complexity == "moderate":
        max_variations = 2
    else:  # complex
        max_variations = 3
Strategie: Adaptive Query-Expansion basierend auf Fragenkomplexität.

12. Hauptretrieval-Funktion
@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 10) -> str:
    # 1. HyDE Generation
    # 2. Initial Retrieval (50+ Kandidaten)
    # 3. Vertex AI Re-Ranking oder Fallback-Filterung
    # 4. Context Formatting mit klickbaren Links
Pipeline: HyDE → Retrieval → Re-Ranking → Formatierung.

13. Context Formatierung
def _format_context_parts(docs_texts: list[str], metadatas: list[dict]) -> str:
    superscript_map = {1: '¹', 2: '²', 3: '³', ...}
    
    # Deduplizierte URL-Referenzen
    url_to_ref_num = {}
    
    # Klickbare Links: [¹](URL), [²](URL)
    clickable_references = []
    for url, ref_num in url_to_ref_num.items():
        superscript = superscript_map.get(ref_num, f"^{ref_num}")
        clickable_references.append(f"[{superscript}]({url}) {url}")
Output: Formatierter Kontext mit hochgestellten, klickbaren Quellenverweisen.

14. Haupteinstiegspunkt
async def run_rag_agent_entrypoint(question: str, deps: RAGDeps, llm_model: str) -> str:
    if llm_model.startswith("gemini"):
        # Gemini-spezifischer Pfad mit Multi-Query
        context_result = await retrieve_context_for_gemini(question, deps)
        
        # Cache-Check
        cached_response = llm_cache.get(question, context_result)
        if cached_response:
            return cached_response
        
        # Gemini-Generation
        response = await generate_with_gemini(...)
        
        # Cache-Speicherung
        llm_cache.store(question, context_result, response)
        
    else:
        # Pydantic AI Agent für andere Modelle
        result = await agent.run(question, deps=deps, model=llm_model)
Architektur: Zwei Pfade - Gemini-optimiert vs. Standard Pydantic AI.

15. CLI Interface
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--question", required=True)
    parser.add_argument("--collection", default="docs")
    parser.add_argument("--vertex-project-id", ...)
    parser.add_argument("--use-vertex-reranker", action='store_true')
    
    deps = RAGDeps(...)
    response = asyncio.run(run_rag_agent_entrypoint(...))
Gesamtarchitektur-Flow:
Input: Benutzerfrage
Query Expansion: HyDE + Multi-Query basierend auf Komplexität
Embedding: Vertex AI Embeddings (mit Cache)
Retrieval: ChromaDB Vektorsuche
Re-Ranking: Vertex AI Reranker oder Fallback-Filterung
Context: Formatierung mit klickbaren Quellenverweisen
Generation: Gemini 2.5 Flash (mit OpenAI Fallback)
Caching: LLM Response Cache für Performance
Output: Strukturierte Antwort mit Quellen
Optimierungen:

Dual-Cache-System (LLM + Embeddings)
Adaptive Query-Strategien
Professionelles Re-Ranking
Fallback-Mechanismen
Batch-Processing für Embeddings


1. Pydantic BaseModel für strukturierte Ausgaben
from pydantic import BaseModel, Field

class StructuredRagAnswer(BaseModel):
    """Ein Modell für eine strukturierte Antwort aus dem RAG-System."""
    summary: str = Field(..., description="Eine prägnante Zusammenfassung der Antwort in 2-3 Sätzen.")
    key_details: List[str] = Field(..., description="Eine Liste der wichtigsten Fakten, Details oder Eigenschaften.")
    contact_info: Optional[str] = Field(None, description="Kontaktinformationen falls verfügbar (E-Mail, Telefon).")
    sources: List[str] = Field(..., description="Liste der verwendeten Quellen-URLs.")
    confidence_score: float = Field(..., ge=0, le=1, description="Konfidenzwert (0.0 bis 1.0) basierend auf der Qualität der gefundenen Dokumente.")
Zweck: Validierte, strukturierte Datenmodelle mit automatischer Typenprüfung und Beschreibungen.

2. Pydantic AI Framework
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

agent = Agent(
    name="RAGAgent",
    description="Answers questions about crawled website content using Retrieval-Augmented Generation (RAG).",
    system_prompt=SYSTEM_PROMPT_TEMPLATE,
    dependencies=RAGDeps,
    model="gpt-4.1-mini",
    llm=aclient
)
Zweck: Das gesamte Agent-Framework basiert auf Pydantic AI, einem modernen Framework für KI-Agenten.

3. Dataclass für Dependencies
from dataclasses import dataclass

@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    chroma_client: chromadb.Client
    collection_name: str
    embedding_model_name: str
    embedding_provider: str
    vertex_project_id: Optional[str] = None
    vertex_location: Optional[str] = None
    use_vertex_reranker: bool = False
    vertex_reranker_model: Optional[str] = None
Hinweis: Hier wird @dataclass statt Pydantic verwendet, aber es arbeitet mit dem Pydantic AI System zusammen.

4. Agent Tool mit Pydantic AI
@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 10) -> str:
    """Retrieve relevant documents from ChromaDB using HyDE, then re-rank them."""
Zweck: Der @agent.tool Decorator ist Teil des Pydantic AI Frameworks und ermöglicht typisierte Tool-Funktionen.

5. Formatierungsfunktion für strukturierte Ausgaben
def format_structured_answer(structured_answer: StructuredRagAnswer) -> str:
    """Format a structured answer into a readable text format."""
    formatted_response = f"""**Zusammenfassung:**
{structured_answer.summary}

**Wichtige Details:**
"""
    
    for detail in structured_answer.key_details:
        formatted_response += f"- {detail}\n"
Zweck: Konvertiert das Pydantic-Modell in lesbaren Text.

Zusammenfassung der Pydantic-Nutzung:
Datenvalidierung: StructuredRagAnswer für typisierte Ausgaben
Agent Framework: Komplettes Pydantic AI System für den RAG-Agent
Tool-System: Typisierte Tools mit automatischer Validierung
Type Safety: Vollständige Typenprüfung zur Laufzeit
Dokumentation: Automatische API-Dokumentation durch Field-Beschreibungen
Aber: Im aktuellen Code wird hauptsächlich der Gemini-Pfad verwendet, der das Pydantic AI Framework umgeht:

if llm_model.startswith("gemini"):
    # Direkter Gemini-Aufruf ohne Pydantic AI Agent
    context_result = await retrieve_context_for_gemini(question, deps)
    response = await generate_with_gemini(...)
else:
    # Pydantic AI Agent für andere Modelle
    result = await agent.run(question, deps=deps, model=llm_model)
Das bedeutet: Pydantic ist vorhanden und konfiguriert, aber wird nur für Non-Gemini-Modelle aktiv genutzt. Der Hauptpfad (Gemini) umgeht das Pydantic AI System für bessere Performance und Kontrolle.