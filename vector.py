from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
import pandas as pd
from typing import List

# Initialize components
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Patterns for detecting PII
PII_PATTERNS = {
    'cnic': r'\b\d{5}-\d{7}-\d{1}\b',  # Pakistani CNIC format
    'phone': r'\b\d{4}-\d{7}\b',       # Pakistani phone format
    'account_number': r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Account number format
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
}

def remove_pii(text: str) -> str:
    """
    Remove Personally Identifiable Information (PII) from text.
    Replaces detected PII with [REDACTED] tags.
    """
    if not isinstance(text, str):
        return text
        
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[REDACTED_{pii_type.upper()}]', text)
    
    # Additional specific redactions for banking context
    text = re.sub(r'(?i)(account\s*(?:no|number|#)?:\s*)\d+', r'\1[REDACTED_ACCOUNT]', text)
    text = re.sub(r'(?i)(customer\s*id:\s*)\d+', r'\1[REDACTED_CUSTOMER_ID]', text)
    text = re.sub(r'(?i)(transaction\s*id:\s*)\d+', r'\1[REDACTED_TRANSACTION_ID]', text)
    
    return text


def load_or_create_vector_store():
    db_location = "./chroma_langchain_db"
    add_documents = not os.path.exists(db_location)
    
    # Load existing data from JSON if it exists
    if os.path.exists("bank_data.json"):
        df = pd.read_json("bank_data.json")
    else:
        df = pd.DataFrame()

    if add_documents:
        documents = []
        ids = []
        doc_id = 0

        for account_name, account_data in df.items():
            for detail in enumerate(account_data["details"]):
                documents.append(
                    Document(
                        page_content=f"Question: {detail[1]['question']}\nAnswer: {detail[1]['answer']}",
                    )
                )
                ids.append(str(doc_id))
                doc_id += 1

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_location
        )
    else:
        vector_store = Chroma(
            persist_directory=db_location,
            embedding_function=embeddings
        )

    return vector_store

vector_store = load_or_create_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

def add_documents_to_store(files: List[Document]):
    """Add new documents to the vector store"""
    # Process and split documents
    splits = text_splitter.split_documents(files)
    
    # Add to vector store
    vector_store.add_documents(splits)
    
    # Persist changes
    vector_store.persist()
    return len(splits)