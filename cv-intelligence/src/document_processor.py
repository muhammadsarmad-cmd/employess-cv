"""Document processing pipeline for CVs."""

import os
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredImageLoader,
    DirectoryLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS
from .metadata_extractor import extract_cv_metadata


class CVDocumentProcessor:
    """Process CV documents from various formats."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )

    def load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document based on its extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            print(f"Unsupported file type: {ext}")
            return []

        doc_type = SUPPORTED_EXTENSIONS[ext]

        try:
            if doc_type == "pdf":
                loader = PyPDFLoader(str(path))
            elif doc_type == "word":
                loader = Docx2txtLoader(str(path))
            elif doc_type == "text":
                loader = TextLoader(str(path), encoding="utf-8")
            elif doc_type == "image":
                loader = UnstructuredImageLoader(str(path))
            else:
                return []

            docs = loader.load()

            # Add source metadata
            for doc in docs:
                doc.metadata["source_file"] = path.name
                doc.metadata["file_type"] = doc_type
                doc.metadata["file_path"] = str(path)

            return docs

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def load_directory(
        self,
        directory: str,
        show_progress: bool = True
    ) -> List[Document]:
        """Load all supported documents from a directory."""
        path = Path(directory)
        all_docs = []

        # Get all supported files
        files = []
        for ext in SUPPORTED_EXTENSIONS.keys():
            files.extend(path.glob(f"**/*{ext}"))

        # Load each file
        iterator = tqdm(files, desc="Loading documents") if show_progress else files
        for file_path in iterator:
            docs = self.load_single_document(str(file_path))
            all_docs.extend(docs)

        print(f"Loaded {len(all_docs)} documents from {len(files)} files")
        return all_docs

    def process_documents(
        self,
        documents: List[Document],
        extract_metadata: bool = True,
        show_progress: bool = True
    ) -> List[Document]:
        """Process documents: split and extract metadata."""

        # Combine all page content per source file for metadata extraction
        if extract_metadata:
            # Group by source file
            file_docs = {}
            for doc in documents:
                source = doc.metadata.get("source_file", "unknown")
                if source not in file_docs:
                    file_docs[source] = []
                file_docs[source].append(doc)

            # Extract metadata per file
            iterator = tqdm(file_docs.items(), desc="Extracting metadata") if show_progress else file_docs.items()
            for source_file, docs in iterator:
                full_text = "\n\n".join(doc.page_content for doc in docs)
                cv_metadata = extract_cv_metadata(full_text)

                # Apply metadata to all docs from this file
                for doc in docs:
                    doc.metadata.update(cv_metadata)

        # Split documents
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk index
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def process_directory(
        self,
        directory: str,
        extract_metadata: bool = True
    ) -> List[Document]:
        """Load and process all documents from a directory."""
        documents = self.load_directory(directory)
        chunks = self.process_documents(documents, extract_metadata)
        return chunks


# Convenience function
def process_cv_directory(directory: str) -> List[Document]:
    """Process all CVs in a directory."""
    processor = CVDocumentProcessor()
    return processor.process_directory(directory)
