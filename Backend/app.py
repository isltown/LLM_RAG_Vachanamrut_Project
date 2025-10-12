import os, re, json, numpy as np, regex as regex

base = "/home/user/.cache/huggingface"
try:
    os.makedirs(base, exist_ok=True)
except Exception:
    base = "/tmp/huggingface"
    os.makedirs(base, exist_ok=True)
os.environ.setdefault("HF_HOME", base)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(base, "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(base, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(base, "transformers"))

for p in (os.environ["HF_HUB_CACHE"], os.environ["HF_DATASETS_CACHE"], os.environ["TRANSFORMERS_CACHE"]):
    os.makedirs(p, exist_ok=True)
from typing import List, Optional, Dict, Any
from functools import lru_cache
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datasets import load_dataset
from huggingface_hub import hf_hub_download, InferenceClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

REPO_ID          = os.getenv("REPO_ID", "nk233/vachnamrut_dataset")
FAISS_PATHS      = [p.strip() for p in os.getenv("FAISS_PATHS", "indices/emb.index,emb.index").split(",")]
LLM_MODEL_ID     = os.getenv("LLM_MODEL_ID", "google/gemma-2-9b-it")
HF_TOKEN         = os.getenv("HF_TOKEN")  # Secret if model requires auth
ALLOWED_ORIGINS  = os.getenv("ALLOWED_ORIGINS", "*")
SEMANTIC_TOP_K   = int(os.getenv("SEMANTIC_TOP_K", "60"))
BM25_TOP_M       = int(os.getenv("BM25_TOP_M", "30"))
FINAL_K          = int(os.getenv("FINAL_K", "6"))
MMR_LAMBDA       = float(os.getenv("MMR_LAMBDA", "0.65"))
NEIGHBOR_WINDOW  = int(os.getenv("NEIGHBOR_WINDOW", "4"))
MAX_TOTAL_CHARS  = int(os.getenv("MAX_TOTAL_CHARS", "8000"))
EMBED_MODEL_ID   = os.getenv("EMBED_MODEL_ID", "l3cube-pune/gujarati-sentence-bert-nli")
index_path = None

ds = load_dataset(REPO_ID, split="train")

PREFIX_RE = regex.compile(r"^([^_]+(?:_[^_]+)?)__")

if "chapter_id" not in ds.column_names:
    def _derive_chapter(ex):
        _id = ex.get("id", "")
        m = PREFIX_RE.match(_id) if isinstance(_id, str) else None
        ex["chapter_id"] = m.group(1) if m else ""
        return ex
    ds = ds.map(_derive_chapter)

for cand in FAISS_PATHS:
    try:
        index_path = hf_hub_download(repo_id=REPO_ID, filename=cand, repo_type="dataset")
        break
    except Exception:
        pass
if index_path is None:
    raise RuntimeError(f"FAISS index not found. Upload one of: {FAISS_PATHS}")

ds.load_faiss_index("emb", index_path)

HAS_EMB = "emb" in ds.features
EMB_MAT = np.stack(ds["emb"]).astype("float32") if HAS_EMB else None
ID_LIST   = ds["id"]
TEXT_LIST = ds["text"]
CHAP_LIST = ds["chapter_id"]
ID_TO_IDX = {i: idx for idx, i in enumerate(ID_LIST)}
HAS_POS   = "pos" in ds.column_names
ID_NUM_RE = regex.compile(r'__(?:c|s)(\d+)$', regex.IGNORECASE)

def parse_pos_from_id(_id: str):
    if not _id: return None
    m = ID_NUM_RE.search(_id)
    return int(m.group(1)) if m else None

chapter_to_rows = {}

for idx, (cid, _id) in enumerate(zip(CHAP_LIST, ID_LIST)):
    if not cid: 
        continue
    pos = ds["pos"][idx] if HAS_POS else parse_pos_from_id(_id)
    chapter_to_rows.setdefault(cid, []).append((idx, pos))
for ch, items in chapter_to_rows.items():
    items.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else x[0]))
    chapter_to_rows[ch] = [i for i, _ in items]

_model: SentenceTransformer | None = None

def _load_embed_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_ID)
        # warmup
        _ = _model.encode(["ટેસ્ટ"], convert_to_numpy=True, normalize_embeddings=True)
    return _model

@lru_cache(maxsize=256)
def embed_text_cached(q: str) -> bytes:
    """Cache normalized float32 vector as bytes to speed up repeated queries."""
    m = _load_embed_model()
    v = m.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
    return v.tobytes()

def get_query_embedding(query: str) -> np.ndarray:
    v = np.frombuffer(embed_text_cached(query), dtype="float32")
    return v
    
def ensure_unit(v):
    v = np.asarray(v, dtype="float32"); n = np.linalg.norm(v) + 1e-12
    return v / n

def cos_sim_vector_to_matrix(q_vec, cand_matrix):
    q = ensure_unit(q_vec)
    C = cand_matrix / (np.linalg.norm(cand_matrix, axis=1, keepdims=True) + 1e-12)
    return C @ q

def mmr_select(query_vec, cand_embs, pool_indices, k=6, lam=0.65):
    if len(pool_indices) == 0:
        return []
    cand = cand_embs[pool_indices]
    rel = cos_sim_vector_to_matrix(query_vec, cand)
    if cand.shape[0] == 0:
        return []
    selected = [int(np.argmax(rel))]
    while len(selected) < min(k, cand.shape[0]):
        scores = []
        for i in range(cand.shape[0]):
            if i in selected: scores.append(-np.inf); continue
            ci = cand[i] / (np.linalg.norm(cand[i]) + 1e-12)
            max_sim_to_S = 0.0
            for s in selected:
                cs = cand[s] / (np.linalg.norm(cand[s]) + 1e-12)
                max_sim_to_S = max(max_sim_to_S, float(ci @ cs))
            score = lam*rel[i] - (1.0 - lam)*max_sim_to_S
            scores.append(score)
        selected.append(int(np.argmax(scores)))
    return [pool_indices[i] for i in selected]

def neighbor_expand(selected_row_indices: List[int], window: int, max_total_chars: int):
    seen, ordered, total = set(), [], 0
    for idx in selected_row_indices:
        ch = CHAP_LIST[idx]
        cand = []
        if ch in chapter_to_rows:
            ordered_rows = chapter_to_rows[ch]
            if idx in ordered_rows:
                pos = ordered_rows.index(idx)
                cand.append(ordered_rows[pos])
                for off in range(1, window + 1):
                    if pos - off >= 0: cand.append(ordered_rows[pos - off])
                    if pos + off < len(ordered_rows): cand.append(ordered_rows[pos + off])
            else:
                cand.append(idx)
        else:
            cand.append(idx)
        for j in cand:
            if j in seen: continue
            txt = TEXT_LIST[j] or ""
            if total and total + len(txt) > max_total_chars: break
            seen.add(j); ordered.append(j); total += len(txt)
    return [{"id": ID_LIST[j], "chapter_id": CHAP_LIST[j], "text": TEXT_LIST[j]} for j in ordered]

def build_llm_context_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.get('chapter_id','UNKNOWN')}]")
        parts.append((c.get("text") or "").strip())
        parts.append("")
    return "\n".join(parts).strip()

llm_client = InferenceClient(model=LLM_MODEL_ID, token=HF_TOKEN if HF_TOKEN else None)

app = FastAPI(title="Vachnamrut Retrieval + LLM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS.split(",") if ALLOWED_ORIGINS else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class EmbedRequest(BaseModel):
    query: str

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = None
    bm25_m: Optional[int] = None
    mmr_lambda: Optional[float] = None
    neighbor_window: Optional[int] = None
    max_total_chars: Optional[int] = None

class AnswerRequest(SearchRequest):
    max_tokens: Optional[int] = 800
    temperature: Optional[float] = 0.0

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "dataset": REPO_ID,
        "faiss_loaded": True,
        "embed_model": EMBED_MODEL_ID,
        "llm_model": LLM_MODEL_ID,
        "defaults": {
            "semantic_top_k": SEMANTIC_TOP_K,
            "bm25_top_m": BM25_TOP_M,
            "final_k": FINAL_K,
            "mmr_lambda": MMR_LAMBDA,
            "neighbor_window": NEIGHBOR_WINDOW,
            "max_total_chars": MAX_TOTAL_CHARS,
        },
    }

@app.post("/embed")
def api_embed(body: EmbedRequest):
    try:
        v = get_query_embedding(body.query)
        return {"query": body.query, "embedding_dim": int(v.shape[0]), "embedding": v.tolist()}
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

@app.post("/search")
def api_search(body: SearchRequest):
    try:
        qv = get_query_embedding(body.query)
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}
    k_sem   = SEMANTIC_TOP_K
    k_final = FINAL_K if body.k is None else int(body.k)
    m_bm25  = BM25_TOP_M if body.bm25_m is None else int(body.bm25_m)
    lam     = MMR_LAMBDA if body.mmr_lambda is None else float(body.mmr_lambda)
    window  = NEIGHBOR_WINDOW if body.neighbor_window is None else int(body.neighbor_window)
    max_chr = MAX_TOTAL_CHARS if body.max_total_chars is None else int(body.max_total_chars)
    scores, batch = ds.get_nearest_examples("emb", qv, k=k_sem)
    docs = list(batch["text"])
    rows = [ID_TO_IDX[_id] for _id in batch["id"]]
    cand_embs = EMB_MAT[rows] if HAS_EMB else np.stack([ds[i]["emb"] for i in rows]).astype("float32")
    bm25 = BM25Okapi([d.split() for d in docs]) if docs else None
    bm25_scores = bm25.get_scores(body.query.split()) if bm25 else np.array([])
    bm25_top = np.argsort(-bm25_scores)[:m_bm25] if len(docs) else np.array([], dtype=int)
    picked = mmr_select(qv, cand_embs, bm25_top, k=k_final, lam=lam)
    selected_rows = [rows[i] for i in picked]
    selected = [{"id": ID_LIST[r], "chapter_id": CHAP_LIST[r], "text": TEXT_LIST[r]} for r in selected_rows]
    expanded = neighbor_expand(selected_rows, window=window, max_total_chars=max_chr)
    return {"query": body.query, "selected": selected, "expanded": expanded}

@app.post("/answer")
def api_answer(body: AnswerRequest):
    res = api_search(body)
    if isinstance(res, dict) and "error" in res:
        return res
    selected = res["selected"]
    expanded = res["expanded"]
    chunks = expanded if expanded else selected
    llm_context = build_llm_context_from_chunks(chunks)
    messages = [{
        "role": "user",
        "content": (
                        "You are a helpful assistant that answers questions about the Vachnamrut in Gujarati language only.\n\n"
                        "You will be provided with relevant context passages taken directly from the original Gujarati text of the Vachnamrut.\n"
                        "Use this context to answer the user’s question in your own words — do not copy or quote the text verbatim.\n\n"
                        "If the context does not contain enough information to answer accurately, reply exactly with:\n"
                        "\"Sorry, I cannot answer that question due to lack of relevant context.\"\n\n"
                        "Whenever you use information from the context, cite the chapter ID in the format (Chapter <ID>).\n"
                        "Answer in clear and fluent Gujarati.\n\n"
            f"Context:\n{llm_context}\n\n"
            f"Question:\n{body.query}\n\n"
            "Answer:"
        )
    }]

    try:
        resp = llm_client.chat_completion(
            messages=messages,
            max_tokens=int(body.max_tokens or 800),
            temperature=float(body.temperature or 0.0),
        )
        answer_text = resp.choices[0].message.content
    except Exception as e:
        return {"error": f"LLM inference failed: {e}"}

    return {"query": body.query, "answer": answer_text, "selected": selected, "expanded": expanded}
