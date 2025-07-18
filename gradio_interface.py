import gradio as gr
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_path = current_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_system import create_rag_system

class  RAGInterface:
    def __init__(self):
        """Initialize RAG interface with improved configuration."""
        config = {
            'chunk_size': 800,
            'chunk_overlap': 150,
            'embedding_model': 'local', # Updated to use local by default
            'max_retrieved_chunks': 5
        }
        self.rag_system = create_rag_system(config)
        self.chat_history = []
        self.processed_files = []

    def process_document(self, file):
        """Process uploaded document with improved error handling."""
        if file is None:
            return "⚠️ No file uploaded"
    
        try:
            # Handle the file object from Gradio
            if hasattr(file, 'name'):
                # Normal file object with file path
                file_name = os.path.basename(file.name)
                file_path = file.name
            else:
                # Handle other cases
                file_name = "uploaded_document"
                file_path = file
        
            # Process document using the updated API
            result = self.rag_system.add_documents([file_path])
        
            # Check results using the new response format
            if result['successful']:
                doc_info = result['successful'][0]
                self.processed_files.append({
                    'name': file_name,
                    'chunks': doc_info['chunks'],
                    'processing_time': result['processing_time']
                })
                return f"✅ Successfully processed **{file_name}**!\n\n📄 Created **{doc_info['chunks']}** searchable chunks\n⏱️ Processing time: **{result['processing_time']:.2f}s**"
            else:
                error_info = result['failed'][0] if result['failed'] else {'error': 'Unknown error'}
                return f"❌ Error processing **{file_name}**: {error_info.get('error', 'Unknown error')}"
    
        except Exception as e:
            return f"❌ Error: {str(e)}"
        
    def query_documents(self, question, history):
        """Query the RAG system with improved response handling."""
        if not question.strip():
            return history, ""
        
        try:
            # Check if documents are loaded
            if not self.processed_files:
                error_response = "⚠️ Please upload and process some documents first!"
                history.append([question, error_response])
                return history, ""
            
            # Query the system
            result = self.rag_system.query(question)

            # Handle the new response format
            if result.get('success', False):
                response = f"**🤖 Answer:**\n{result.get('answer', 'No answer available')}\n\n"
                response += f"**📊 Confidence:** {result.get('confidence', 0.0):.2f}\n"
                response += f"**📚 Sources Used:** {len(result.get('sources', []))}\n\n"

                if result.get('sources'):
                    response += "**📋 Source Details:**\n"
                    for i, source in enumerate(result['sources'][:3], 1):
                        # Handle different source formats
                        if isinstance(source, dict):
                            filename = source.get('filename', source.get('source_number', f'Source {i}'))
                            score = source.get('relevance_score', source.get('score', 0.0))
                            response += f"{i}. **{filename}** (relevance: {score:.3f})\n"
                        else:
                            response += f"{i}. Source information available\n"
            else:
                response = f"❌ Error: {result.get('error', 'Unknown error occurred')}"

            # Update history
            history.append([question, response])

            return history, ""
        
        except Exception as e:
            error_response = f"❌ Error: {str(e)}"
            history.append([question, error_response])
            return history, ""
        
    def get_stats(self):
        """Get system statistics with improved formatting."""
        try:
            stats = self.rag_system.get_system_info()

            status_emoji = "✅" if stats.get('is_initialized', False) else "❌"

            return f"""
            ## 📊 System Statistics
            
            **System Status:** {status_emoji} {'Ready' if stats.get('is_initialized', False) else 'Not Ready'}
            
            **📚 Document Statistics:**
            - Total Documents: **{stats.get('total_documents', 0)}**
            - Total Searchable Chunks: **{stats.get('total_chunks', 0)}**
            
            **⚙️ Configuration:**
            - Embedding Model: **{stats.get('config', {}).get('embedding_model', 'Unknown')}**
            - Chunk Size: **{stats.get('config', {}).get('chunk_size', 'Unknown')}**
            - Chunk Overlap: **{stats.get('config', {}).get('chunk_overlap', 'Unknown')}**
            
            **📄 Processed Files:**
            {chr(10).join([f'- {doc}' for doc in stats.get('processed_documents', [])]) if stats.get('processed_documents') else '- No documents processed yet'}
            
            ---
            
            **💡 Tip:** Upload documents in the Document Upload tab to start asking questions!
            """
        except Exception as e:
            return f"❌ Error retrieving stats: {str(e)}"
        
    def clear_all_data(self):
        """Clear all data from the system."""
        try:
            # Clear the RAG system
            self.rag_system.clear_documents()

            # Clear local tracking
            self.processed_files = []

            return "✅ All data cleared successfully!"
        except Exception as e:
            return f"❌ Error clearing data: {str(e)}"
        
def create_interface():
    """Create Gradio interface with improved layout and features."""

    rag_interface = RAGInterface()

    with gr.Blocks(title="🤖 Smart Document Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Smart Document Assistant")
        gr.Markdown("**Transform your documents into a conversational knowledge base!**")
        
        with gr.Tab("📄 Document Upload"):
            gr.Markdown("### Upload and Process Documents")
            gr.Markdown("Supported formats: TXT, PDF, DOCX, CSV, XLSX")
            
            with gr.Row():
                file_input = gr.File(
                    label="📁 Choose Document", 
                    file_types=[".txt", ".pdf", ".docx", ".csv", ".xlsx"]
                )
                upload_btn = gr.Button("📤 Process Document", variant="primary")
            
            upload_output = gr.Markdown(label="Processing Status")
            
            upload_btn.click(
                rag_interface.process_document,
                inputs=[file_input],
                outputs=[upload_output]
            )
        
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Ask Questions About Your Documents")
            
            chatbot = gr.Chatbot(
                label="Conversation", 
                height=500,
                show_label=False,
                avatar_images=None
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question", 
                    placeholder="e.g., What is the main topic of the document?",
                    scale=4
                )
                submit_btn = gr.Button("🚀 Submit", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                example_btn = gr.Button("💡 Example Questions", variant="secondary")

            # Example questions display
            examples = gr.Markdown(visible=False)

            # Chat functionality
            submit_btn.click(
                rag_interface.query_documents,
                inputs = [msg, chatbot],
                outputs=[chatbot, msg]
            )

            clear_btn.click(
                lambda: ([], ""),
                outputs=[chatbot, msg]
            )

            # Example questions functionality
            def show_examples():
                return gr.Markdown(
                    """
                    ### 💡 Example Questions to Try:
                    
                    - "What is the main topic of this document?"
                    - "Can you summarize the key points?"
                    - "What are the benefits mentioned?"
                    - "Who are the main stakeholders?"
                    - "What recommendations are provided?"
                    - "Are there any risks or challenges discussed?"
                    """,
                    visible=True
                )
            
            example_btn.click(
                show_examples,
                outputs=[examples]
            )

        with gr.Tab("📊 Statistics"):
            gr.Markdown("### System Information and Statistics")
            
            with gr.Row():
                stats_btn = gr.Button("🔄 Refresh Stats", variant="primary")
                clear_all_btn = gr.Button("🗑️ Clear All Data", variant="secondary")

            stats_output = gr.Markdown()
            clear_output = gr.Markdown()

            stats_btn.click(
                rag_interface.get_stats,
                outputs=[stats_output]
            )

            clear_all_btn.click(
                rag_interface.clear_all_data,
                outputs=[clear_output]
            )

            # Load stats on startup
            demo.load(
                rag_interface.get_stats,
                outputs=[stats_output]
            )

    return demo

def launch_gradio_interface():
    """Launch the Gradio interface with proper error handling."""
    try:
        print("🚀 Starting Gradio RAG Interface...")
        print("="*50)
        
        # Create and launch interface
        interface = create_interface()
        
        print("🌐 Starting web interface...")
        print("📱 Access the interface at: http://localhost:7860")
        print("💡 Press Ctrl+C to stop the application")
        print("-" * 50)
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Error launching Gradio interface: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure gradio is installed: pip install gradio")
        print("2. Check that all component files exist in the src folder")
        print("3. Verify your Python environment and dependencies")

if __name__ == "__main__":
    launch_gradio_interface()
        
    
