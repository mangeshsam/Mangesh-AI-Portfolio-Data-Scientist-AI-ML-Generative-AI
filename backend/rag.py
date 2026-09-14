import os
import re
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import pypdf

# Initialize ChromaDB client (local file-based)
CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "chroma_data")
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

def get_or_create_collection(name="portfolio_collection"):
    return client.get_or_create_collection(name=name)

def get_or_create_pdf_collection():
    return client.get_or_create_collection(name="uploaded_pdf_collection")

def extract_text_from_ts(filepath):
    """Simple extractor to grab strings from the TS data files."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove interface definitions safely (non-greedy)
    content = re.sub(r'export interface .*?\}', '', content, flags=re.DOTALL | re.MULTILINE)
    
    # Remove export const ... = 
    content = re.sub(r'export const \w+(:\s*[^=]+)?\s*=\s*', '', content)
    
    return content.strip()

def initialize_rag():
    try:
        client.delete_collection("portfolio_collection")
        print("Deleted old ChromaDB collection to rebuild index.")
    except ValueError:
        pass
        
    collection = get_or_create_collection()
        
    txt_path = os.path.join(os.path.dirname(__file__), "mangesh_portfolio.txt")
    
    docs = []
    ids = []
    metadatas = []
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into logical sections by double newlines
        raw_chunks = content.split('\n\n')
        
        for i, chunk in enumerate(raw_chunks):
            if len(chunk.strip()) > 10:
                docs.append(chunk.strip())
                ids.append(f"portfolio_{i}")
                metadatas.append({"source": "mangesh_portfolio.txt"})
            
    if docs:
        print(f"Embedding {len(docs)} chunks into ChromaDB from text file natively...")
        try:
            collection.add(
                documents=docs,
                metadatas=metadatas,
                ids=ids
            )
            print("RAG content successfully loaded into ChromaDB from text file!")
        except Exception as e:
            print(f"WARNING: Hugging Face API blocked locally. Backend will boot but RAG may fail. Error: {e}")

def query_rag(query_text: str, n_results: int = 2):
    collection = get_or_create_collection()
    
    try:
        # Return 2 most relevant chunks
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
    except Exception as e:
        print(f"WARNING: Query failed due to API block. Error: {e}")
        return ""
    
    # Combine retrieved documents
    context = ""
    if results and 'documents' in results and results['documents']:
        for doc_list in results['documents']:
            for doc in doc_list:
                context += doc + "\n---\n"
                
    return context

def ingest_pdf(pdf_path: str):
    """Extracts text from a PDF, chunks it, and stores it in a temporary collection."""
    try:
        client.delete_collection("uploaded_pdf_collection")
    except ValueError:
        pass
    collection = get_or_create_pdf_collection()
    
    docs = []
    ids = []
    
    reader = pypdf.PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Simple chunking by paragraphs/newlines
            paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 20]
            for i, p in enumerate(paragraphs):
                docs.append(p)
                ids.append(f"page{page_num}_{i}")
                
    if docs:
        try:
            collection.add(
                documents=docs,
                ids=ids
            )
            return len(docs)
        except Exception as e:
            print(f"WARNING: PDF upload embeddings failed. Error: {e}")
            return 0
    return 0

def query_pdf_rag(query_text: str, n_results: int = 3):
    collection = get_or_create_pdf_collection()
    if collection.count() == 0:
        return ""
    
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
    except Exception as e:
        print(f"WARNING: Query failed due to API block. Error: {e}")
        return ""
    
    context = ""
    if results and 'documents' in results and results['documents']:
        for doc_list in results['documents']:
            for doc in doc_list:
                context += doc + "\n---\n"
                
    return context
