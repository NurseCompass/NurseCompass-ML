import os
import requests
import config

# Safe LangChain Imports
try:
    from langchain_community.document_loaders import PDFPlumberLoader
except ImportError:
    from langchain.document_loaders import PDFPlumberLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    from langchain.embeddings import OllamaEmbeddings


class RAGService:
    def __init__(self):
        self.vector_db = None
        try:
            self.embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
            self.initialize_rag()
        except Exception as e:
            print(f"⚠️ Embeddings initialization deferred: {e}")

    def initialize_rag(self):
        os.makedirs(config.TRAINING_FOLDER, exist_ok=True)

        if os.path.exists(config.CHROMA_DB_DIR) and os.listdir(config.CHROMA_DB_DIR):
            print("⚡ Loading existing study materials from vector database...")
            try:
                self.vector_db = Chroma(
                    persist_directory=config.CHROMA_DB_DIR,
                    embedding_function=self.embeddings
                )
                print("✅ RAG Pipeline Ready!")
                return
            except Exception as e:
                print(f"⚠️ Error loading Chroma DB: {e}. Re-indexing PDFs...")

        print("📚 Scanning 'training_data' for study materials using PDFPlumberLoader...")
        docs = []
        for filename in os.listdir(config.TRAINING_FOLDER):
            if filename.endswith(".pdf"):
                file_path = os.path.join(config.TRAINING_FOLDER, filename)
                try:
                    loader = PDFPlumberLoader(file_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Failed to load {filename}: {e}")

        if not docs:
            print("⚠️ No PDFs found in training_data directory.")
            self.vector_db = None
            return

        print(f"📄 Loaded {len(docs)} pages. Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=60)
        chunks = text_splitter.split_documents(docs)

        print(f"✂️ Created {len(chunks)} text chunks. Indexing vector database...")
        batch_size = 50
        self.vector_db = Chroma.from_documents(
            documents=chunks[:batch_size],
            embedding=self.embeddings,
            persist_directory=config.CHROMA_DB_DIR
        )

        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.vector_db.add_documents(batch)

        print("✅ RAG Pipeline successfully indexed study materials!")

    def get_context(self, query, k=6):
        if self.vector_db is None:
            self.initialize_rag()

        if self.vector_db is None:
            return ""

        try:
            results = self.vector_db.similarity_search(query, k=k)
            context_text = "\n\n".join([doc.page_content for doc in results])
            return context_text.strip()
        except Exception as e:
            print(f"❌ Error performing similarity search: {e}")
            return ""

# Singleton instance
rag_service = RAGService()