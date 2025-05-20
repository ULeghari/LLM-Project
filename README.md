# LLM-Project

## Introduction
This project aims to develop an AI-driven customer service assistant for a fictional bank using Large Language Models (LLMs). The system processes anonymized customer interaction data to provide accurate, context-aware responses while ensuring data privacy and reliability. The implementation involves data preprocessing, embedding generation, LLM integration, and a user-friendly interface for seamless customer interactions.

## Implementation Overview
The system begins by ingesting and cleaning the provided bank dataset (nust_bank_data.json). Text normalization includes lowercasing, whitespace correction, and punctuation handling, while special patterns like currency (e.g., "Rs.100") and percentages are preserved using custom regex rules. The cleaned data is tokenized using NLTK and stored in CSV/JSON formats. For retrieval, the all-MiniLM-L6-v2 model generates 384-dimensional sentence embeddings, enabling efficient similarity searches via cosine similarity.

The LLM component uses T5-small in a retrieval-augmented generation (RAG) pipeline. When a user submits a query, the system retrieves the most relevant pre-existing answer as context, then generates a natural-language response using prompt engineering. Guardrails include similarity thresholding to deflect off-topic questions (e.g., "How to cook pasta?" triggers a polite banking-related redirection). The Gradio-based interface provides a simple chat experience with example queries and real-time responses.

## System Architecture
The architecture follows a modular pipeline: raw JSON data undergoes cleaning and tokenization before embedding storage. User queries are vectorized and matched against stored embeddings; the top result feeds into T5-small for response generation. The Gradio UI wraps this workflow into an interactive chat, ensuring accessibility.

## Conclusion
This project delivers a functional banking assistant that automates customer queries using LLMs. Core features like data processing, embedding retrieval, and RAG are implemented, with a user-friendly interface. While foundational, the system can be expanded for scalability and security, demonstrating the potential of AI in financial customer support.
