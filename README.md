<a href="https://github.com/isltown/Vachnamrut-RAG-LLM">
  <h1 align="center">RAG LLM Question Answering System on the Vachanamrut Scripture</h1>
</a>

<p align="center">
  An AI-powered Gujarati chatbot that answers questions from the original <em>Vachanamrut</em> scripture using Retrieval-Augmented Generation (RAG) with Open Source Large Language Model (Gemma-2-9B).
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

<br/>

## API Architecture (Code at /Backend/app.py)

The system follows a **RAG (Retrieval-Augmented Generation)** pipeline:

1. **Semantic Search** – Encodes the user query using `l3cube-pune/gujarati-sentence-bert-nli` encoding model.  
2. **Hybrid Retrieval** – Uses FAISS (Facebook AI Similarity Search) + BM25 (Best Matching 25) + MMR (Maximal Marginal Relevance) selection for top passages.  
3. **Context Expansion** – Adds nearby scripture lines for depth and coherence.  
4. **LLM Generation** – Passes the context to a model like `google/gemma-2-9b-it` for answer generation.  
5. **Response** – Returns a Gujarati answer with (Chapter ID) citations.

API Hosted using Hugging Face Space. (Link - https://huggingface.co/spaces/nk233/vachnamrut_api/tree/main)

## FlowChart
<img width="843" height="463" alt="Screenshot 2025-10-12 at 9 35 59 PM" src="https://github.com/user-attachments/assets/6c22ee6e-1d03-4a23-bcb9-c5061cc36218" />

## Website
Link - https://llm-rag-vachnamrut-project.vercel.app/


