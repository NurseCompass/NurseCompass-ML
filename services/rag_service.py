import os
import config
import pdfplumber
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

from langchain_ollama import OllamaEmbeddings


class RAGService:
    def __init__(self):
        self.vector_db = None
        try:
            self.embeddings = OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url="https://swept-uncurled-darkroom.ngrok-free.dev"
            )
            self.initialize_rag()
        except Exception as e:
            print(f"⚠️ Embeddings initialization deferred: {e}")

    def get_absolute_path(self, path_name):
        """Helper to guarantee an absolute path based on the project root."""
        if os.path.isabs(path_name):
            return path_name
        # Resolves to the NurseCompass-ML root directory (one level up from /services)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, path_name)

    def initialize_rag(self):
        # 1. Force absolute paths for storage
        training_dir = self.get_absolute_path(config.TRAINING_FOLDER)
        chroma_dir = self.get_absolute_path(config.CHROMA_DB_DIR)

        # 2. Explicitly create directories BEFORE SQLite tries to open them
        os.makedirs(training_dir, exist_ok=True)
        os.makedirs(chroma_dir, exist_ok=True)

        # 3. Initialize Chroma with absolute path
        if os.listdir(chroma_dir):
            print(f"⚡ Loading existing study materials from {chroma_dir}...")
            try:
                self.vector_db = Chroma(
                    persist_directory=chroma_dir,
                    embedding_function=self.embeddings
                )
                print("✅ RAG Pipeline Ready!")
            except Exception as e:
                print(f"⚠️ Error loading Chroma DB: {e}")
        else:
            print(f"⚠️ No existing ChromaDB found. Creating an empty vector store at {chroma_dir}...")
            self.vector_db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=chroma_dir
            )

    def add_document(self, file_path, filename, owner_id, section_id):
        if self.vector_db is None:
            self.initialize_rag()

        print(f"📚 Scanning '{filename}' for study materials...")
        docs = []
        try:
            extracted_text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        extracted_text += page_text + "\n\n"
            
            if extracted_text:
                metadata = {
                    "source": filename,
                    "owner_id": str(owner_id),
                    "section_id": str(section_id)
                }
                docs.append(Document(page_content=extracted_text, metadata=metadata))
        except Exception as e:
            print(f"⚠️ Failed to load {filename}: {e}")
            return False

        if not docs:
            print(f"⚠️ No text could be extracted from {filename}.")
            return False

        print(f"📄 Splitting text for {filename}...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        print(f"✂️ Indexing {len(chunks)} chunks into vector database with privacy controls...")
        self.vector_db.add_documents(chunks)
        print(f"✅ Successfully indexed {filename} [Owner: {owner_id} | Section: {section_id}]")
        return True

    def get_context(self, query, user_id="unknown", section_id="none", k=12):
        if self.vector_db is None:
            self.initialize_rag()

        if self.vector_db is None:
            return ""

        try:
            # 1. Pull a massive net of 60 chunks to ensure we don't miss the answer
            results = self.vector_db.similarity_search(query, k=60)
            
            valid_chunks = []
            
            # 2. Aggressively clean the user's sections (lowercase, remove all spaces)
            raw_user_sections = str(section_id).split(",")
            user_sections = [s.strip().lower().replace(" ", "") for s in raw_user_sections if s.strip()]
            
            print(f"\n🔍 --- RAG PRIVACY FILTER DEBUG ---")
            print(f"👤 Asking User ID: {user_id} | Allowed Sections: {user_sections}")

            for doc in results:
                doc_owner = str(doc.metadata.get("owner_id", "unknown"))
                doc_sections_str = str(doc.metadata.get("section_id", "none"))
                
                # Clean the document's metadata exactly the same way
                doc_sections = [s.strip().lower().replace(" ", "") for s in doc_sections_str.split(",") if s.strip()]

                is_allowed = False
                
                # Check Ownership
                if doc_owner == str(user_id):
                    is_allowed = True
                else:
                    # Check Section Match
                    for sec in user_sections:
                        if sec != 'none' and sec in doc_sections:
                            is_allowed = True
                            break
                
                if is_allowed:
                    valid_chunks.append(doc)
                    if len(valid_chunks) >= k:
                        break
            
            print(f"✅ Context Assembly Complete: Kept {len(valid_chunks)} authorized chunks.")
            print(f"----------------------------------\n")

            context_text = "\n\n".join([doc.page_content for doc in valid_chunks])
            return context_text.strip()
        except Exception as e:
            print(f"❌ Error performing similarity search: {e}")
            return ""

# Singleton instance
rag_service = RAGService()