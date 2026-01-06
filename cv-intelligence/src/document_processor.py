"""Document processing pipeline for CVs."""

import os
from pathlib import Path
from typing import List, Optional, Iterator
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
    """Process CV documents from various formats.

    Supports both batch loading (prototype) and streaming (production).
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.data_dir = Path(data_dir) if data_dir else None
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

    # === NEW METHODS FOR STREAMING/BATCH PROCESSING ===

    def iter_files(self, directory: Optional[Path] = None) -> Iterator[Path]:
        """Iterate over files without loading into memory.

        Args:
            directory: Directory to scan (uses self.data_dir if None)

        Yields:
            Path objects for each supported file
        """
        scan_dir = Path(directory) if directory else self.data_dir
        if not scan_dir:
            raise ValueError("No directory specified")

        for ext in SUPPORTED_EXTENSIONS.keys():
            for file_path in scan_dir.rglob(f"*{ext}"):
                if file_path.is_file():
                    yield file_path

    def count_files(self, directory: Optional[Path] = None) -> int:
        """Count total files to process.

        Args:
            directory: Directory to scan (uses self.data_dir if None)

        Returns:
            Number of supported files found
        """
        return sum(1 for _ in self.iter_files(directory))

    def _load_file(self, file_path: Path) -> List[Document]:
        """Internal method to load a single file.

        Args:
            file_path: Path to the file

        Returns:
            List of documents (usually 1, but PDFs may have multiple pages)
        """
        return self.load_single_document(str(file_path))

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk index
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks

    def process_file(
        self,
        file_path: Path,
        extract_metadata: bool = True
    ) -> List[Document]:
        """Process a single file and return chunks.

        Args:
            file_path: Path to the file
            extract_metadata: Whether to extract CV metadata

        Returns:
            List of document chunks
        """
        try:
            # Load document
            documents = self._load_file(file_path)

            if not documents:
                return []

            # Extract metadata if requested
            if extract_metadata:
                full_text = "\n\n".join(doc.page_content for doc in documents)
                cv_metadata = extract_cv_metadata(full_text)

                # Apply metadata to all documents
                for doc in documents:
                    doc.metadata.update(cv_metadata)

            # Chunk documents
            chunks = self.chunk_documents(documents)

            return chunks

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def iter_batches(
        self,
        batch_size: int = 50,
        directory: Optional[Path] = None
    ) -> Iterator[List[Path]]:
        """Yield batches of file paths.

        Args:
            batch_size: Number of files per batch
            directory: Directory to scan (uses self.data_dir if None)

        Yields:
            Lists of Path objects
        """
        batch = []
        for file_path in self.iter_files(directory):
            batch.append(file_path)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


# Convenience function
def process_cv_directory(directory: str) -> List[Document]:
    """Process all CVs in a directory."""
    processor = CVDocumentProcessor()
    return processor.process_directory(directory)
