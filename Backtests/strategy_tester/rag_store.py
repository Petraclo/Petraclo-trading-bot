# rag_store.py
import os, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from models import Run

# Choose a lightweight embedding model
EMB_MODEL = "all-MiniLM-L6-v2"   # 384-dim
INDEX_PATH = "index.faiss"
META_PATH = "index_meta.json"

_model = SentenceTransformer(EMB_MODEL)
_index = None
_meta = []

# ---------------------------
# Helpers
# ---------------------------
def _load_index():
    """Load FAISS + metadata if exists, else initialize new."""
    global _index, _meta
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
        _meta = json.load(open(META_PATH, "r"))
    else:
        _index = faiss.IndexFlatIP(384)  # dimension = 384
        _meta = []

def _save_index():
    faiss.write_index(_index, INDEX_PATH)
    json.dump(_meta, open(META_PATH, "w"))

def _ensure_index():
    if _index is None:
        _load_index()

# ---------------------------
# Convert Run -> Document
# ---------------------------
def doc_from_run(r: Run) -> str:
    return (
        f"RunID: {r.id} | {r.created_at}\n"
        f"Symbol: {r.symbol} | TF: {r.timeframe} | Dates: {r.begin_date} → {r.end_date}\n"
        f"Params: R:R={r.r_r}, ATRx={r.atr_multiplier}, Sens={r.sensitivity}, "
        f"HTF={r.htf_analysis}, ModifiedStop={r.modified_stop}, Adaptive={r.adaptive}, InitialBalance={r.initial_balance}\n"
        f"Results: PnL={r.pnl_usd}, Return%={r.return_pct}, WinRate%={r.win_rate_pct}, "
        f"Trades={r.avg_trades}, MaxDD%={r.max_dd_pct}, AvgWeeklyPnL={r.avg_weekly_pnl_usd}, "
    )

# ---------------------------
# Upsert Run
# ---------------------------
def upsert_run_to_index(r: Run):
    _ensure_index()
    doc = doc_from_run(r)
    vec = _model.encode([doc], normalize_embeddings=True)
    _index.add(vec.astype("float32"))
    _meta.append({"run_id": r.id})
    _save_index()

# ---------------------------
# Search
# ---------------------------
def search(query: str, top_k=8):
    _ensure_index()
    qv = _model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = _index.search(qv, top_k)
    hits = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        hits.append({
            "idx": int(idx),
            "score": float(D[0][rank]),
            "meta": _meta[idx]
        })
    return hits
