"""
Semantic Chunking Implementation for better text splitting.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class SemanticChunk:
    """Represents a semantically coherent text chunk."""
    text: str
    start_pos: int
    end_pos: int
    chunk_type: str  # 'paragraph', 'section', 'sentence'
    metadata: Dict = None

class SemanticChunker:
    """Intelligent text chunking based on semantic boundaries."""
    
    def __init__(self, 
                 target_chunk_size: int = 1200,
                 max_chunk_size: int = 1800,
                 min_chunk_size: int = 300,
                 overlap_size: int = 150):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[SemanticChunk]:
        """Split text into semantic chunks."""
        if len(text) <= self.target_chunk_size:
            return [SemanticChunk(
                text=text,
                start_pos=0,
                end_pos=len(text),
                chunk_type='complete',
                metadata=metadata
            )]
        
        # Try different splitting strategies in order of preference
        chunks = self._split_by_sections(text, metadata)
        if not chunks:
            chunks = self._split_by_paragraphs(text, metadata)
        if not chunks:
            chunks = self._split_by_sentences(text, metadata)
        if not chunks:
            chunks = self._split_by_characters(text, metadata)
        
        # Add overlap between chunks
        chunks = self._add_overlap(chunks, text)
        
        return chunks
    
    def _split_by_sections(self, text: str, metadata: Dict) -> List[SemanticChunk]:
        """Split by sections (headers, major breaks)."""
        # Look for section markers
        section_patterns = [
            r'\n#{1,6}\s+.+\n',  # Markdown headers
            r'\n[A-Z][A-Z\s]{10,}\n',  # ALL CAPS headers
            r'\n\d+\.\s+[A-Z].+\n',  # Numbered sections
            r'\n[A-Z][^.!?]*:[\s]*\n',  # Title: format
        ]
        
        sections = []
        current_pos = 0
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) >= 2:  # Need at least 2 sections
                for i, match in enumerate(matches):
                    start = current_pos if i == 0 else matches[i-1].end()
                    end = match.start() if i < len(matches) - 1 else len(text)
                    
                    section_text = text[start:end].strip()
                    if self.min_chunk_size <= len(section_text) <= self.max_chunk_size:
                        sections.append(SemanticChunk(
                            text=section_text,
                            start_pos=start,
                            end_pos=end,
                            chunk_type='section',
                            metadata=metadata
                        ))
                
                if sections:
                    return sections
        
        return []
    
    def _split_by_paragraphs(self, text: str, metadata: Dict) -> List[SemanticChunk]:
        """Split by paragraphs."""
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed target size
            if len(current_chunk) + len(para) > self.target_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(SemanticChunk(
                    text=current_chunk.strip(),
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    chunk_type='paragraph_group',
                    metadata=metadata
                ))
                
                # Start new chunk
                current_chunk = para
                current_start = text.find(para, current_start + len(current_chunk))
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = text.find(para)
        
        # Add final chunk
        if current_chunk:
            chunks.append(SemanticChunk(
                text=current_chunk.strip(),
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                chunk_type='paragraph_group',
                metadata=metadata
            ))
        
        return chunks if chunks else []
    
    def _split_by_sentences(self, text: str, metadata: Dict) -> List[SemanticChunk]:
        """Split by sentences."""
        # Improved sentence splitting
        sentence_endings = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed target size
            if len(current_chunk) + len(sentence) > self.target_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(SemanticChunk(
                    text=current_chunk.strip(),
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    chunk_type='sentence_group',
                    metadata=metadata
                ))
                
                # Start new chunk
                current_chunk = sentence
                current_start = text.find(sentence, current_start + len(current_chunk))
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                    current_start = text.find(sentence)
        
        # Add final chunk
        if current_chunk:
            chunks.append(SemanticChunk(
                text=current_chunk.strip(),
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                chunk_type='sentence_group',
                metadata=metadata
            ))
        
        return chunks if chunks else []
    
    def _split_by_characters(self, text: str, metadata: Dict) -> List[SemanticChunk]:
        """Fallback: split by characters at word boundaries."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.target_chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk_text = text[start:]
            else:
                # Find word boundary
                while end > start and text[end] not in ' \n\t':
                    end -= 1
                
                if end == start:  # No word boundary found
                    end = start + self.target_chunk_size
                
                chunk_text = text[start:end]
            
            chunks.append(SemanticChunk(
                text=chunk_text.strip(),
                start_pos=start,
                end_pos=end,
                chunk_type='character_split',
                metadata=metadata
            ))
            
            start = end
        
        return chunks
    
    def _add_overlap(self, chunks: List[SemanticChunk], original_text: str) -> List[SemanticChunk]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.text
            
            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i-1]
                overlap_start = max(0, len(prev_chunk.text) - self.overlap_size)
                overlap_text = prev_chunk.text[overlap_start:]
                
                # Find word boundary for clean overlap
                if overlap_text and not overlap_text[0].isspace():
                    space_pos = overlap_text.find(' ')
                    if space_pos > 0:
                        overlap_text = overlap_text[space_pos+1:]
                
                chunk_text = overlap_text + " " + chunk_text
            
            overlapped_chunks.append(SemanticChunk(
                text=chunk_text.strip(),
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                chunk_type=chunk.chunk_type,
                metadata=chunk.metadata
            ))
        
        return overlapped_chunks

# Convenience function for easy integration
def semantic_chunk_text(text: str, 
                       chunk_size: int = 1200, 
                       chunk_overlap: int = 150,
                       metadata: Dict = None) -> List[str]:
    """
    Simple function to chunk text semantically.
    Returns list of text strings.
    """
    chunker = SemanticChunker(
        target_chunk_size=chunk_size,
        overlap_size=chunk_overlap
    )
    
    chunks = chunker.chunk_text(text, metadata)
    return [chunk.text for chunk in chunks]