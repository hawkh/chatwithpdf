import os
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
VECTOR_DB = os.getenv("VECTOR_DB", "CHROMA").upper()
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")
TOP_K = int(os.getenv("TOP_K", "5"))


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def build_or_update_vector_store(text_chunks: List[str], metadatas: Optional[List[dict]] = None) -> None:
    embeddings = get_embeddings()
    if VECTOR_DB == "CHROMA":
        from langchain_community.vectorstores import Chroma
        _ = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            collection_name="pdf_chunks",
            persist_directory=PERSIST_DIR,
            metadatas=metadatas,
        )
        _.persist()
    else:
        from langchain_community.vectorstores import FAISS
        index_dir = "faiss_index"
        if os.path.exists(index_dir):
            vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            vs.add_texts(text_chunks, metadatas=metadatas)
            vs.save_local(index_dir)
        else:
            vs = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
            vs.save_local(index_dir)


def load_vector_store():
    embeddings = get_embeddings()
    if VECTOR_DB == "CHROMA":
        from langchain_community.vectorstores import Chroma
        return Chroma(
            collection_name="pdf_chunks",
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )
    else:
        from langchain_community.vectorstores import FAISS
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)