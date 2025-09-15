from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import AskRequest, AskResponse
from app.rag import VectorStore
from app.prompts import compose_prompt
from app.safety import is_oos, REFUSAL
from app.llm import generate_ollama

api = FastAPI(title="Finn v0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VS = VectorStore()
if not VS.load():
    VS.build()

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = req.query.strip()
    if is_oos(q):
        return AskResponse(answer=REFUSAL, sources=[], oos=True)
    path_chunks = VS.search(q, k=3)
    snippets = [c for _, c in path_chunks]
    prompt = compose_prompt(snippets, q, req.context or {})
    answer = generate_ollama(prompt)
    sources = list(dict.fromkeys([p for p, _ in path_chunks]))
    return AskResponse(answer=answer, sources=sources, oos=False)
