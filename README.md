<a href="https://github.com/isltown/Vachnamrut-RAG-LLM">
  <h1 align="center">Vachnamrut RAG LLM API</h1>
</a>

<p align="center">
  An AI-powered Gujarati chatbot that answers questions from the original <em>Vachnamrut</em> scripture using Retrieval-Augmented Generation (RAG) with Open Source Large Language Model (Gemma 2 9B).
</p>

<p align="center">
  <a href="#features"><strong>Features</strong></a> ·
  <a href="#architecture"><strong>Architecture</strong></a> ·
  <a href="#run-locally"><strong>Run Locally</strong></a> ·
  <a href="#api-endpoints"><strong>API Endpoints</strong></a>
</p>

<br/>

##  Features

- **Gujarati Language Support**
  - Understands and responds in Gujarati, based on the authentic *Vachnamrut* text.
- **Retrieval-Augmented Generation (RAG)**
  - Combines semantic search, BM25 ranking, and LLM reasoning for factual answers.
- **Context-Aware Responses**
  - Includes relevant scripture excerpts and chapter references for transparency.
- **FastAPI Backend**
  - Lightweight, production-ready API design.
- **Modular Components**
  - FAISS for semantic retrieval, Hugging Face for inference, Sentence-BERT for embeddings.

---

## Architecture

The system follows a **RAG (Retrieval-Augmented Generation)** pipeline:

1. **Semantic Search** – Encodes the user query using `l3cube-pune/gujarati-sentence-bert-nli`.  
2. **Hybrid Retrieval** – Uses FAISS (semantic) + BM25 (lexical) + MMR selection for top passages.  
3. **Context Expansion** – Adds nearby scripture lines for depth and coherence.  
4. **LLM Generation** – Passes the context to a model like `google/gemma-2-9b-it` for answer generation.  
5. **Response** – Returns a Gujarati answer with (Chapter ID) citations.

## FlowChart
<img width="864" height="480" alt="Screenshot 2025-10-12 at 9 22 33 PM" src="https://github.com/user-attachments/assets/714cab44-7beb-4a41-a580-992d4b13cddf" />


