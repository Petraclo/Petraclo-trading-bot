from models import SessionLocal, Run
from rag_store import search, doc_from_run

def retrieve_runs(query: str, symbol=None, timeframe=None, top_k=8):
    """Hybrid retrieval: filter in SQL, rank in FAISS."""
    s = SessionLocal()
    q = s.query(Run)
    if symbol:
        q = q.filter(Run.symbol == symbol)
    if timeframe:
        q = q.filter(Run.timeframe == timeframe)
    base_rows = q.order_by(Run.created_at.desc()).limit(500).all()
    s.close()

    if not base_rows:
        return [], []

    # Build docs for candidate runs
    corpus_docs = {r.id: doc_from_run(r) for r in base_rows}

    # Run FAISS search
    hits = search(query, top_k=top_k*2)  # get more, filter later
    picked, docs = [], []
    for h in hits:
        rid = h["meta"]["run_id"]
        if rid in corpus_docs and len(picked) < top_k:
            picked.append(next(r for r in base_rows if r.id == rid))
            docs.append(corpus_docs[rid])
    return picked, docs

def build_context(rows, docs):
    header = "Top results:\nID | Sym | TF | Win% | MaxDD% | PnL\n"
    lines = [
        f"{r.id} | {r.symbol} | {r.timeframe} | {r.win_rate_pct:.1f} | {r.max_dd_pct:.1f} | {r.pnl_usd:.0f}"
        for r in rows[:5]
    ]
    return header + "\n".join(lines) + "\n\n" + "\n---\n".join(docs[:5])
