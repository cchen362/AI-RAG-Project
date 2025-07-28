import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

# Document processing libraries
import pypdf
import pdfplumber
from docx import Document
import pandas as pd

# Text processing
import re
from datetime import datetime

class DocumentProcessor:
    """Like a universal translator documents.
    Takes any document type and converts it to clean, structured text.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the document processor"""
        self.config = config or {}
        self.supported_formats = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.txt', '.csv']

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'total_pages': 0,
            'errors': 0,
            'processing_time': 0
        }

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and extract its content."""
        start_time = datetime.now()

        try:
            # Get file information
            file_info = self._get_file_info(file_path)

            # Extract content based on file type
            content = self._extract_content(file_path, file_info['extension'])

            # Clean and normalize the content
            clean_content = self._clean_content(content)

            # Extract metadata
            metadata = self._extract_metadata(file_path, content)

            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                'file_path': file_path,
                'file_info': file_info,
                'content': clean_content,
                'metadata': metadata,
                'processing_time': processing_time,
                'status': 'success',
                'success': True,  # Add this for RAG system compatibility
                'error': None
            }

            self.stats['files_processed'] += 1
            self.stats['processing_time'] += processing_time

            self.logger.info(f"✅ Processed: {Path(file_path).name}")
            return result
        
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"❌ Error processing {file_path}: {str(e)}")

            return {
                'file_path': file_path,
                'content': None,
                'status': 'error',
                'success': False,  # Add this for RAG system compatibility
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic file information."""
        path_obj = Path(file_path)
        stat = path_obj.stat()

        return {
            'filename': path_obj.name,
            'extension': path_obj.suffix.lower(),
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created_date': datetime.fromtimestamp(stat.st_ctime),
            'modified_date': datetime.fromtimestamp(stat.st_mtime),
            'file_hash': self._calculate_file_hash(file_path)
        }
        
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for duplicate detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _extract_content(self, file_path: str, extension: str) -> str:
        """Extract text content based on file type."""
        if extension == '.pdf':
            return self._extract_pdf_content(file_path)
        elif extension in ['.docx', '.doc']:
            return self._extract_word_content(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self._extract_excel_content(file_path)
        elif extension == '.txt':
            return self._extract_text_content(file_path)
        elif extension == '.csv':
            return self._extract_csv_content(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
    def _extract_pdf_content(self, file_path: str) -> str:
        """Extract text from PDF files."""
        content = ""

        try: 
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    self.stats['total_pages'] += 1

        except Exception as e:
            self.logger.warning(f"pdfplumber failed for {file_path}, trying pypdf: {e}")

            # Fallback to pypdf
            try:
                pdf_reader = pypdf.PdfReader(file_path)

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    self.stats['total_pages'] += 1

            except Exception as e2:
                raise Exception(f"Both PDF extraction methods failed: {e2}")
            
        return content
    
    def _extract_word_content(self, file_path: str) -> str:
        """Extract text from Word documents."""
        try:
            doc = Document(file_path)
            content = ""

            # Extract paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content += paragraph.text + "\n"

            # Extract tables
            for table in doc.tables:
                content += "\n--- Table ---\n"
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    content += row_text + "\n"

            return content

        except Exception as e:
            raise Exception(f"Failed to extract Word content: {e}")
        
    def _extract_excel_content(self, file_path: str) -> str:
        """Extract content from Excel files."""
        try:
            excel_file = pd.ExcelFile(file_path)
            content = "## Excel Workbook Analysis\n\n"

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                content += f"### Sheet: {sheet_name}\n"
                content += f"This sheet contains {len(df)} rows and {len(df.columns)} columns.\n\n"
                
                # Add column definitions for better semantic understanding
                content += "#### Column Structure:\n"
                for col in df.columns:
                    # Get sample non-null values to understand data type
                    sample_vals = df[col].dropna().head(3).tolist()
                    if sample_vals:
                        sample_str = ", ".join([str(v)[:40] for v in sample_vals])
                        content += f"- {col}: {sample_str}...\n"
                    else:
                        content += f"- {col}: (empty column)\n"
                content += "\n"

                # Add data in semantic sections (grouped for better chunking)
                content += "#### Sheet Data:\n"
                max_rows = self.config.get('excel_max_rows', 100)
                
                # Group rows for better chunking (every 15 rows)
                for group_start in range(0, min(len(df), max_rows), 15):
                    group_end = min(group_start + 15, len(df), max_rows)
                    content += f"\n##### Rows {group_start + 1}-{group_end}:\n"
                    
                    for idx in range(group_start, group_end):
                        row = df.iloc[idx]
                        row_text = " | ".join([f"{col}: {val}" for col, val in zip(df.columns, row.values)])
                        content += f"Row {idx + 1}: {row_text}\n"

                if len(df) > max_rows:
                    content += f"\n... ({len(df) - max_rows} additional rows not shown)\n"
                    
                content += "\n---\n\n"  # Separator between sheets

            return content
    
        except Exception as e:
            raise Exception(f"failed to extract Excel content: {e}")
        
    def _extract_text_content(self, file_path: str) -> str:
        """Extract plaint text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
            
    def _extract_csv_content(self, file_path: str) -> str:
        """Extract CSV content."""
        try:
            df = pd.read_csv(file_path)

            # Create semantic-friendly CSV content
            content = f"## CSV Data Summary\n"
            content += f"This CSV file contains {len(df)} rows and {len(df.columns)} columns.\n\n"
            
            # Add column definitions for better semantic understanding
            content += "### Column Definitions:\n"
            for col in df.columns:
                # Get sample non-null values to understand data type
                sample_vals = df[col].dropna().head(3).tolist()
                sample_str = ", ".join([str(v)[:50] for v in sample_vals])
                content += f"- {col}: {sample_str}...\n"
            content += "\n"
            
            # Add data in semantic sections
            content += "### Data Content:\n"
            max_rows = self.config.get('csv_max_rows', 50)
            
            # Group rows for better chunking (every 10 rows)
            for group_start in range(0, min(len(df), max_rows), 10):
                group_end = min(group_start + 10, len(df), max_rows)
                content += f"\n#### Rows {group_start + 1}-{group_end}:\n"
                
                for idx in range(group_start, group_end):
                    row = df.iloc[idx]
                    row_text = " | ".join([f"{col}: {val}" for col, val in zip(df.columns, row.values)])
                    content += f"Row {idx + 1}: {row_text}\n"

            if len(df) > max_rows:
                content += f"\n... ({len(df) - max_rows} additional rows not shown)\n"

            return content
        
        except Exception as e:
            raise Exception(f"Failed to extract CSV content: {e}")
        
    def _clean_content(self, content: str) -> str:
        """Clean and normalize extracted content."""
        if not content:
            return ""

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)

        # Remove special characters that might interfere
        content = re.sub(r'[^\w\s\-.,!?:;()\[\]{}"\'/\\@#$%^&*+=<>|~`]', '', content)   
        
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Remove extremely long lines (likely formatting artifacts)
        lines = content.split('\n')
        cleaned_lines = []

        for line in lines:
            if len(line) < 1000:
                cleaned_lines.append(line.strip())

        return '\n'.join(cleaned_lines).strip()
    
    def _extract_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """Extract useful metadata from the document"""
        metadata = {
            'word_count': len(content.split()) if content else 0,
            'char_count': len(content) if content else 0,
            'line_count': len(content.split('\n')) if content else 0,
        }

        # Try to extract title (first meaningful line)
        if content:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if lines:
                for line in lines[:5]:
                    if len(line) > 10 and len(line) < 200:
                        metadata['title'] = line
                        break

                # Extract potential keywords
                words = content.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 4:
                        word_freq[word] = word_freq.get(word, 0) + 1

                top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                metadata['keywords'] = [word for word, freq in top_keywords]

        return metadata
    
        
