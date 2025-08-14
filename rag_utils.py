import os
import shutil
from typing import List, Optional

import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Optional imports for vector stores
try:
    from langchain_community.vectorstores import Chroma  # type: ignore
    _HAS_CHROMA = True
except Exception:
    Chroma = None  # type: ignore
    _HAS_CHROMA = False

try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    _HAS_FAISS = True
except Exception:
    FAISS = None  # type: ignore
    _HAS_FAISS = False


@st.cache_resource(show_spinner=False)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def extract_documents_from_pdfs(pdf_files) -> List[Document]:
    """Extract per-page Documents with metadata from uploaded PDF files.

    Each Document contains page content and metadata: {'source': filename, 'page': page_number}.
    """
    documents: List[Document] = []
    for uploaded in pdf_files:
        try:
            reader = PdfReader(uploaded)
            source_name = getattr(uploaded, "name", "uploaded.pdf")
            for page_index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                metadata = {"source": source_name, "page": page_index + 1}
                documents.append(Document(page_content=text, metadata=metadata))
        except Exception:
            # Skip problematic files/pages but continue processing others
            continue
    return documents


def split_documents(documents: List[Document], chunk_size: int = 1500, chunk_overlap: int = 200) -> List[Document]:
    """Split Documents into overlapping chunks while preserving metadata."""
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def _reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def build_chroma_index(chunks: List[Document], persist_dir: str = "chroma_db", collection_name: str = "pdf_chunks"):
    """Build and persist a Chroma index from chunks. Returns the vectorstore.

    If Chroma is not available, returns None.
    """
    if not _HAS_CHROMA:
        return None
    if not chunks:
        return None
    _reset_dir(persist_dir)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    try:
        vectorstore.persist()
    except Exception:
        pass
    return vectorstore


def load_chroma_index(persist_dir: str = "chroma_db", collection_name: str = "pdf_chunks"):
    """Load a persisted Chroma index if present; else return None."""
    if not _HAS_CHROMA:
        return None
    if not os.path.isdir(persist_dir):
        return None
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def build_faiss_index(chunks: List[Document], faiss_dir: str = "faiss_index"):
    """Build and persist a FAISS index from chunks. Returns the vectorstore."""
    if not _HAS_FAISS:
        return None
    if not chunks:
        return None
    _reset_dir(faiss_dir)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(faiss_dir)
    return vectorstore


def load_faiss_index(faiss_dir: str = "faiss_index"):
    """Load a persisted FAISS index if present; else return None."""
    if not _HAS_FAISS:
        return None
    if not os.path.isdir(faiss_dir):
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)


def build_vectorstore(chunks: List[Document], prefer: str = "chroma"):
    """Build a vectorstore with preference for Chroma then FAISS."""
    prefer = (prefer or "").lower()
    if prefer == "chroma" and _HAS_CHROMA:
        vs = build_chroma_index(chunks)
        if vs is not None:
            return vs
    # Fallback to FAISS when available
    if _HAS_FAISS:
        return build_faiss_index(chunks)
    return None


def load_vectorstore(prefer: str = "chroma"):
    """Load an existing vectorstore with preference for Chroma then FAISS."""
    prefer = (prefer or "").lower()
    if prefer == "chroma" and _HAS_CHROMA:
        vs = load_chroma_index()
        if vs is not None:
            return vs
    if _HAS_FAISS:
        return load_faiss_index()
    return None


def get_retriever(vectorstore, k: int = 6):
    """Return an optimized retriever using MMR for better diversity."""
    if vectorstore is None:
        return None
    fetch_k = max(12, int(k * 4))
    try:
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
        )
    except Exception:
        # Some stores might not support MMR; fallback to simple k
        return vectorstore.as_retriever(search_kwargs={"k": k})