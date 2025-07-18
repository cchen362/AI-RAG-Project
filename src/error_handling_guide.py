# Enhanced Error Handling for Document Processing
# 
# This file shows WHERE and HOW to add robust error handling to your document processor

import logging
import traceback
from typing import Dict, Any, List
from pathlib import Path
import time

class ErrorHandler:
    """
    Like a medical emergency room for your document processing.
    Handles different types of errors gracefully and provides useful feedback.
    """
    
    def __init__(self):
        self.error_log = []
        self.setup_logging()
    
    def setup_logging(self):
        """Set up detailed logging for errors."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/processing_errors.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def handle_file_error(self, file_path: str, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle file-specific errors.
        Like a doctor diagnosing what went wrong with a specific patient.
        """
        error_info = {
            'file_path': file_path,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': time.time(),
            'traceback': traceback.format_exc()
        }
        
        # Log the error
        self.logger.error(f"File processing failed: {file_path}")
        self.logger.error(f"Error: {error}")
        self.logger.error(f"Context: {context}")
        
        # Store for analysis
        self.error_log.append(error_info)
        
        return error_info
    
    def retry_with_fallback(self, func, *args, max_retries=3, **kwargs):
        """
        Try an operation multiple times before giving up.
        Like trying different keys when one doesn't work.
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                time.sleep(1)  # Wait before retry


# HOW TO INTEGRATE ERROR HANDLING INTO YOUR EXISTING CODE:
"""
1. ADD THESE IMPORTS to your document_processor.py:
   import time
   import traceback

2. ADD THIS METHOD to your DocumentProcessor class:
   (Replace your existing process_file method)
"""

def enhanced_process_file(self, file_path: str) -> Dict[str, Any]:
    """
    REPLACE your existing process_file method with this enhanced version.
    This goes INSIDE your DocumentProcessor class.
    """
    start_time = time.time()
    
    try:
        # Step 1: Validate file exists and is readable
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not Path(file_path).is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Step 2: Check file size (prevent memory issues)
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        max_size_mb = self.config.get('max_file_size_mb', 100)
        
        if file_size_mb > max_size_mb:
            raise ValueError(f"File too large ({file_size_mb:.1f}MB). Max allowed: {max_size_mb}MB")
        
        # Step 3: Validate file format
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Step 4: Your original processing logic (with try-catch around each step)
        try:
            file_info = self._get_file_info(file_path)
        except Exception as e:
            raise Exception(f"Failed to get file info: {e}")
        
        try:
            content = self._extract_content(file_path, file_info['extension'])
        except Exception as e:
            raise Exception(f"Failed to extract content: {e}")
        
        try:
            clean_content = self._clean_content(content)
        except Exception as e:
            raise Exception(f"Failed to clean content: {e}")
        
        try:
            metadata = self._extract_metadata(file_path, content)
        except Exception as e:
            raise Exception(f"Failed to extract metadata: {e}")
        
        processing_time = time.time() - start_time
        
        result = {
            'file_path': file_path,
            'file_info': file_info,
            'content': clean_content,
            'metadata': metadata,
            'processing_time': processing_time,
            'status': 'success',
            'error': None
        }
        
        self.stats['files_processed'] += 1
        self.stats['processing_time'] += processing_time
        
        self.logger.info(f"✅ Processed: {Path(file_path).name} ({processing_time:.3f}s)")
        return result
        
    except FileNotFoundError as e:
        return self._create_error_result(file_path, e, "File not found", start_time)
    
    except PermissionError as e:
        return self._create_error_result(file_path, e, "Permission denied", start_time)
    
    except MemoryError as e:
        return self._create_error_result(file_path, e, "Out of memory", start_time)
    
    except ValueError as e:
        return self._create_error_result(file_path, e, "Validation error", start_time)
    
    except Exception as e:
        return self._create_error_result(file_path, e, "Unexpected error", start_time)


"""
3. ADD THIS HELPER METHOD to your DocumentProcessor class:
"""
def _create_error_result(self, file_path: str, error: Exception, context: str, start_time: float) -> Dict[str, Any]:
    """Create standardized error result."""
    processing_time = time.time() - start_time
    
    # Log detailed error information
    self.logger.error(f"❌ Error processing {Path(file_path).name}: {error}")
    self.logger.error(f"Context: {context}")
    self.logger.error(f"Full traceback: {traceback.format_exc()}")
    
    # Update statistics
    self.stats['errors'] += 1
    self.stats['processing_time'] += processing_time
    
    return {
        'file_path': file_path,
        'content': None,
        'status': 'error',
        'error': str(error),
        'error_type': type(error).__name__,
        'context': context,
        'processing_time': processing_time,
        'file_info': None,
        'metadata': None
    }


"""
4. ENHANCED PDF EXTRACTION with fallback:
   Replace your _extract_pdf_content method with this:
"""
def enhanced_extract_pdf_content(self, file_path: str) -> str:
    """Extract text from PDF files with multiple fallback methods."""
    content = ""
    extraction_methods = [
        ('pdfplumber', self._try_pdfplumber),
        ('pypdf', self._try_pypdf)
    ]
    
    for method_name, method_func in extraction_methods:
        try:
            self.logger.info(f"Trying PDF extraction with {method_name}")
            content = method_func(file_path)
            
            if content and len(content.strip()) > 10:  # Minimum content check
                self.logger.info(f"✅ PDF extracted successfully with {method_name}")
                return content
            else:
                self.logger.warning(f"⚠️ {method_name} returned insufficient content")
                
        except Exception as e:
            self.logger.warning(f"⚠️ {method_name} extraction failed: {e}")
            continue
    
    # If we get here, all methods failed
    raise Exception("All PDF extraction methods failed. File may be corrupted or encrypted.")


def _try_pdfplumber(self, file_path: str) -> str:
    """Try extracting with pdfplumber."""
    import pdfplumber
    content = ""
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            self.stats['total_pages'] += 1
    
    return content


def _try_pypdf(self, file_path: str) -> str:
    """Try extracting with pypdf."""
    import pypdf
    content = ""
    
    with open(file_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            self.stats['total_pages'] += 1
    
    return content


"""
SUMMARY - WHERE TO ADD ERROR HANDLING:

1. ✅ REPLACE process_file() with enhanced_process_file() 
2. ✅ ADD _create_error_result() method
3. ✅ REPLACE _extract_pdf_content() with enhanced_extract_pdf_content()
4. ✅ ADD the helper methods _try_pdfplumber() and _try_pypdf()
5. ✅ ADD imports: time, traceback

This gives you:
- File validation before processing
- Memory protection (file size limits)
- Detailed error logging
- Graceful error recovery
- Multiple PDF extraction methods
- Comprehensive error reporting
"""
