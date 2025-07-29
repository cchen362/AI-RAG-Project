from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
import math
from datetime import datetime

@dataclass
class TextChunk:
    """A piece of text with metadata - like a labeled container"""
    content: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any]

    def __len__(self):
        return len(self.content.split())
    
    def __str__(self):
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"Chunk {self.chunk_id}: {preview}"
    
class TextChunker:
    """
    Splits documents into manageable pieces using various strategies.
    
    Think of thi as a smart document slicer that can adapt its cutting 
    style based on the type of content and your needs.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200, strategy: str = "sentences"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy= strategy

        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        
        if self.overlap < 0:
            raise ValueError("Overlap cannot be negative")
        
    def chunk_text(self, text: str, source_metadata: Dict = None) -> List[TextChunk]:
        """Main chunking method - automatically selects strategy."""

        if not text or not text.strip():
            return []
        
        source_metadata = source_metadata or {}

        # Choose chunking method based on strategy
        if self.strategy == "fixed":
            return self._chunk_by_tokens(text, source_metadata)
        elif self.strategy == "sentences":
            return self._chunk_by_sentences(text, source_metadata)
        elif self.strategy == "paragraphs":
            return self._chunk_by_paragraphs(text, source_metadata)
        elif self.strategy == "semantic":
            return self._chunk_by_semantic_sections(text, source_metadata)
        elif self.strategy == "structured_data":
            return self._chunk_by_structured_data(text, source_metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
        
    def _chunk_by_tokens(self, text: str, source_metadata: Dict) -> List[TextChunk]:
        """Fixed-size chunking - like cutting bread into equal slices."""

        words = text.split()
        chunks = []

        if not words:
            return chunks
        
        start_idx = 0
        chunk_num = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)

            # Calculate character positions (approximate)
            char_start = len(' '.join(words[:start_idx]))
            char_end = char_start + len(chunk_text)

            chunk = TextChunk(
                content=chunk_text,
                start_index=char_start,
                end_index=char_end,
                chunk_id=f"token_chunk_{chunk_num:03d}",
                metadata={
                    'chunk_method': 'fixed_tokens',
                    'chunk_size': len(chunk_words),
                    'overlap_tokens': self.overlap,
                    'chunk_number': chunk_num,
                    'total_words': len(words),
                    **source_metadata
                }
            )

            chunks.append(chunk)

            if end_idx >= len(words):
                break

            start_idx = max(end_idx - self.overlap, start_idx + 1)
            chunk_num += 1

        return chunks
    
    def _chunk_by_sentences(self, text: str, source_metadata: Dict) -> List[TextChunk]:
        """Sentence-aware chunking - like cutting at natural pauses."""

        # Split into sentences (improved regex)
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern,text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return self._chunk_by_tokens(text, source_metadata)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_num = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())

            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    start_index=chunk_num * self.chunk_size,
                    end_index=(chunk_num + 1) * self.chunk_size,
                    chunk_id=f"sentence_chunk_{chunk_num:03d}",
                    metadata={
                        'chunk_method': 'sentences',
                        'word_count': len(current_chunk.split()),
                        'chunk_number': chunk_num,
                        **source_metadata
                    }
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + ' ' + sentence if overlap_sentences else sentence
                current_size = len(current_chunk.split())
                chunk_num += 1
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
                current_size += sentence_size

        # Final chunk
        if current_chunk:
            chunk = TextChunk(
                content=current_chunk.strip(),
                start_index=chunk_num * self.chunk_size,
                end_index=(chunk_num + 1) * self.chunk_size,
                chunk_id=f"sentence_chunk_{chunk_num:03d}",
                metadata={
                        'chunk_method': 'sentences',
                        'word_count': len(current_chunk.split()),
                        'chunk_number': chunk_num,
                        **source_metadata
                }
            )
            chunks.append(chunk)

        return chunks
    
    def _get_overlap_sentences(self, chunk_text: str) -> str:
        """Get the last few sentences for overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
        overlap_words = self.overlap

        overlap_text = ""
        words_collected = 0

        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if words_collected + sentence_words <= overlap_words:
                overlap_text = sentence + ' ' + overlap_text if overlap_text else sentence
                words_collected += sentence_words
            else:
                break

        return overlap_text.strip()
    
    def _chunk_by_paragraphs(self, text: str, source_metadata: Dict) -> List[TextChunk]:
        """Paragraph-aware chunking - like cutting at topic changes."""

        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return self._chunk_by_sentences(text, source_metadata)

        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_num = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph.split())

            if paragraph_size > self.chunk_size:
                # Split large paragraph using sentence chunking
                if current_chunk:
                    chunk = self._create_paragraph_chunk(current_chunk, chunk_num, source_metadata)
                    chunks.append(chunk)
                    chunk_num += 1

                temp_chunker = TextChunker(self.chunk_size, self.overlap, "sentences")
                para_chunks = temp_chunker._chunk_by_sentences(paragraph, source_metadata)

                for para_chunk in para_chunks:
                    para_chunk.chunk_id = f"paragraph_chunk_{chunk_num:03d}"
                    para_chunk.metadata['chunk_method'] = 'paragraphs_split'
                    chunks.append(para_chunk)
                    chunk_num += 1

                current_chunk = ""
                current_size = 0

            elif current_size + paragraph_size > self.chunk_size and current_chunk:
                chunk = self._create_paragraph_chunk(current_chunk, chunk_num, source_metadata)
                chunks.append(chunk)

                current_chunk = paragraph
                current_size = paragraph_size
                chunk_num += 1

            else:
                current_chunk += '\n\n' + paragraph if current_chunk else paragraph
                current_size += paragraph_size

        if current_chunk:
            chunk = self._create_paragraph_chunk(current_chunk, chunk_num, source_metadata)
            chunks.append(chunk)

        return chunks
    
    def _create_paragraph_chunk(self, content: str, chunk_num: int, source_metadata: Dict) -> TextChunk:
        """Helper to create a paragraph-based chunk."""

        return TextChunk(
            content=content.strip(),
            start_index=0,
            end_index=len(content),
            chunk_id=f"paragraph_chunk_{chunk_num:03d}",
            metadata={
                'chunk_method': 'paragraphs',
                'word_count': len(content.split()),
                'chunk_number': chunk_num,
                **source_metadata
            }
        )

    def _chunk_by_semantic_sections(self, text: str, source_metadata: Dict) -> List[TextChunk]:
        """Semantic section chunking - simplified approach."""

        # Look for section markers
        section_patterns = [
            r'^#{1,6}\s+.+$',           # Markdown headings
            r'^\d+\.\s+.+$',            # Numbered sections
            r'^[A-Z][A-Z\s]{3,}:?\s*$', # ALL CAPS headings
            r'^.{1,50}:$',              # Short lines ending with colon
        ]

        lines = text.split('\n')
        section_breaks = [0]

        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                for pattern in section_patterns:
                    if re.match(pattern, line, re.MULTILINE):
                        section_breaks.append(i)
                        break

        section_breaks.append(len(lines))

        # Extract sections
        sections = []
        for i in range(len(section_breaks) - 1):
            start_line = section_breaks[i]
            end_line = section_breaks[i + 1]
            section_text = '\n'.join(lines[start_line:end_line]).strip()

            if section_text:
                sections.append(section_text)

        # If no clear sections found, fall back to paragraph chunking
        if len(sections) <= 1:
            return self._chunk_by_paragraphs(text, source_metadata)
        
        # Process each section
        chunks = []
        chunk_num = 0

        for section in sections:
            section_size = len(section.split())

            if section_size <= self.chunk_size:
                # Section fits in one chunk
                chunk = TextChunk(
                    content=section,
                    start_index=0,
                    end_index=len(section),
                    chunk_id=f"semantic_chunk_{chunk_num:03d}",
                    metadata={
                        'chunk_method': 'semantic',
                        'word_count': section_size,
                        'chunk_number': chunk_num,
                        **source_metadata
                    }
                )
                chunks.append(chunk)
                chunk_num += 1
            else:
                # Section too large, split further
                temp_chunker = TextChunker(self.chunk_size, self.overlap, "sentences")
                section_chunks = temp_chunker._chunk_by_sentences(section, source_metadata)

                for section_chunk in section_chunks:
                    section_chunk.chunk_id = f"semantic_chunk_{chunk_num:03d}"
                    section_chunk.metadata['chunk_method'] = 'semantic_split'
                    chunks.append(section_chunk)
                    chunk_num += 1

        return chunks
    
    def _chunk_by_structured_data(self, text: str, source_metadata: Dict) -> List[TextChunk]:
        """
        Structured data chunking - specifically for CSV/Excel data entries.
        
        Creates individual chunks for each data entry to improve retrieval granularity.
        """
        chunks = []
        
        # Look for individual data entries marked by "### Sheet: SheetName - Entry N"
        entry_pattern = r'### Sheet: .+ - Entry \d+\n\n.+?(?=### Sheet: .+ - Entry \d+|\Z)'
        entries = re.findall(entry_pattern, text, re.DOTALL)
        
        if entries:
            # Process individual entries
            start_pos = 0
            for i, entry in enumerate(entries):
                entry = entry.strip()
                if entry:
                    end_pos = start_pos + len(entry)
                    
                    chunk = TextChunk(
                        content=entry,
                        start_index=start_pos,
                        end_index=end_pos,
                        chunk_id=f"data_entry_{i+1:03d}",
                        metadata={
                            'chunk_method': 'structured_data_entry',
                            'entry_number': i + 1,
                            'word_count': len(entry.split()),
                            **source_metadata
                        }
                    )
                    chunks.append(chunk)
                    start_pos = end_pos
        else:
            # Fallback to semantic chunking if no structured pattern found
            return self._chunk_by_semantic_sections(text, source_metadata)
        
        return chunks
    
    def analyze_chunking_quality(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Analyze the quality of chunking results.
        
        Like a quality inspector checking if the pizza slices
        are cut properly - consistent size, complete pieces, etc.
        """

        if not chunks:
            return {"error": "No chunks to analyze"}
        
        # Calculate statistics
        chunk_sizes = [len(chunk.content.split()) for chunk in chunks]

        analysis = {
            'total_chunks': len(chunks),
            'chunk_size_stats': {
                'min': min(chunk_sizes),
                'max': max(chunk_sizes),
                'mean': sum(chunk_sizes) / len(chunk_sizes),
                'target': self.chunk_size
            },
            'size_distribution': {
                'under_50%': sum(1 for size in chunk_sizes if size < self.chunk_size * 0.5),
                'optimal_50-150%': sum(1 for size in chunk_sizes if self.chunk_size * 0.5 <= size <= self.chunk_size * 1.5),
                'over_150%': sum(1 for size in chunk_sizes if size > self.chunk_size * 1.5)
            },
            'content_quality': {
                'empty_chunks': sum(1 for chunk in chunks if not chunk.content.strip()),
                'very_short_chunks': sum(1 for chunk in chunks if len(chunk.content.split()) < 10),
                'average_content_length': sum(len(chunk.content) for chunk in chunks) / len(chunks)
            }
        }

        # Quality score (0-100)
        optimal_count = analysis['size_distribution']['optimal_50-150%']
        quality_score = (optimal_count / len(chunks)) * 100
        analysis['quality_score'] = round(quality_score, 1)

        return analysis
    
# Chunking Strategy Selector
class ChunkingStrategySelector:
    """
    Automatically selects the best chunking strategy based on content type.
    
    Like having a smart assistant who knows whether to slice bread,
    cut pizza, or dice vegetables based on what you're preparing.
    """

    @staticmethod
    def analyze_content(text: str) -> Dict[str, Any]:
        """Analyze text to determine characteristics."""

        lines = text.split('\n')
        paragraphs = re.split(r'\n\s*\n', text)
        sentences = re.split(r'[.!?]+', text)

        # Count structural elements
        heading_count = sum(1 for line in lines if re.match(r'^#+\s|^\d+\.\s|^[A-Z\s]{5,}:?$', line.strip()))

        return {
            'total_length': len(text),
            'word_count': len(text.split()),
            'line_count': len(lines),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'heading_count': heading_count,
            'avg_paragraph_length': sum(len(p.split()) for p in paragraphs if p.strip()) / max(len(paragraphs), 1),
            'avg_sentence_length': sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1),
        }
    
    @staticmethod
    def recommend_strategy(text: str, target_chunk_size: int = 1000) -> str:
        """Recommend the best chunking strategy for given text."""

        analysis = ChunkingStrategySelector.analyze_content(text)

        #Decision logic
        if analysis['heading_count'] > 3:
            return "semantic" # Has clear sections
        elif analysis['avg_paragraph_length'] > target_chunk_size * 0.8:
            return "sentences" # Long paragraphs, break by sentences
        elif analysis['paragraph_count'] > 5 and analysis['avg_paragraph_length'] < target_chunk_size * 0.3:
            return "paragraphs" # Many short paragraphs
        else: 
            return "sentences" # Default to sentence-aware