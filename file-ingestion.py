import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def ingest(pdf_path: str, persist_dir: str = "chroma_db"):
    # 1️⃣ Load the PDF page by page
    docs_per_page = PyPDFLoader(pdf_path).load()
    
    # Extract source filename for metadata
    source_filename = os.path.basename(pdf_path)
    
    # 2️⃣ Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=50,
    )
    
    # 3️⃣ Split each page into chunks and enrich with metadata
    all_chunks = []
    for doc in docs_per_page:
        # Split the content of the page
        chunks_text = text_splitter.split_text(doc.page_content)
        
        # Add page-specific metadata to each chunk
        for chunk_text in chunks_text:
            all_chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "source": source_filename,
                    "page": doc.metadata.get("page", 0) + 1,
                }
            ))

    # 4️⃣ Embed and store with metadata
    embeddings = FastEmbedEmbeddings()
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"✅ Ingested '{pdf_path}' -> vector DB '{persist_dir}'.")
    print(f"📄 Processed {len(all_chunks)} chunks with metadata:")
    print(f"   - Source: {source_filename}")
    if all_chunks:
        pages = [chunk.metadata.get('page', 1) for chunk in all_chunks]
        print(f"   - Pages: {min(pages)}-{max(pages)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python file-ingestion.py <path/to/your.pdf>")
        sys.exit(1)
    print(f"Processing: {sys.argv[1]}")
    ingest(sys.argv[1])
