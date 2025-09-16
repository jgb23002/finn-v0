Finn v0 — Privacy-First Wellness Helper

Finn v0 is a lightweight AI wellness assistant that answers basic health & wellness questions from a small curated knowledge base. It respects user consent, includes safety rails to decline out-of-scope medical queries, and is built entirely with open-source components for transparency and portability.



Features:
    -Retrieval-Augmented Generation (RAG): Uses Sentence-Transformers + FAISS to embed and search knowledge base snippets.
    -Knowledge Base (KB): Simple markdown files with hydration, sleep, and stress basics.
    -Safety Layer: Regex-based out-of-scope detection (e.g., dosages, prescriptions, urgent medical terms).
    -Consent-Aware Context: Only includes app context if `{"consented": true}` is provided in the request.
    -Local LLM Adapter: Calls Ollama (`llama3.1:8b-instruct`) for generation; falls back to a safe default if unavailable.
    -Simple API: FastAPI endpoints with CORS enabled for demo.

How to run the code: 
    -python -m venv .venv && source .venv/bin/activate
    -pip install -r requirements.txt
    -python -m scripts.build_index
    -uvicorn app.main:api --reload
    -open http://127.0.0.1:8000/docs
