#!/usr/bin/env python3
"""
check_dupes.py  – screen a pending batch of math problems for duplicates
  • against an “All Problems” master CSV
  • and within the pending batch itself

Differences from the v1 script:
  – Treat blank IDs as missing (never count as dupes)
  – Ignore any ID in the range L1–L2000 when checking exact-ID duplicates
  – Detect duplicates *within* pending (text + IDs)
"""

import argparse, os, re, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# ─────────────────── configuration ────────────────────
PROMPT = """You are a validator ..."""
MODEL = "gpt-4o-mini"
SIM_THRESHOLD = 0.90
EMBED_MODEL = "all-mpnet-base-v2"

# ─────────────────── helpers ───────────────────────────
def canonicalize(text: str | float) -> str:
    if pd.isna(text):
        return ""
    return (
        str(text).strip()
        .lower()
        .replace("'", "'")
        .replace(""", '"').replace(""", '"')
        .replace("\u2212", "-").replace("\xa0", " ")
    )

_blank_re = re.compile(r"^\s*$")
_skip_id_re = re.compile(r"^L([1-9]\d{0,3}|2000)$")  # matches L1 … L2000

def normalize_id(x) -> str | None:
    """Return a clean string ID, or None if blank OR in skip range."""
    if pd.isna(x) or _blank_re.match(str(x)):
        return None
    s = str(x).strip()
    if _skip_id_re.match(s):
        return None             # hard-skip bogus IDs
    return s

def embed_texts(texts):
    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(
        texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True)

# ─────────────────── main ──────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Check pending problems for duplicates (text + IDs).")
    p.add_argument("--all_csv", required=True)
    p.add_argument("--pending_csv", required=True)

    # column mapping (same defaults as before)
    p.add_argument("--all_id_col", default="#")
    p.add_argument("--all_id_col2", default=None)
    p.add_argument("--all_text_col", default="problem")
    p.add_argument("--pending_id_col", default="#")
    p.add_argument("--pending_id_col2", default=None)
    p.add_argument("--pending_text_col", default="problem")
    p.add_argument("--use_second_id_cols", action="store_true")

    p.add_argument("--out_csv", default="pending_checked.csv")
    p.add_argument("--sim_threshold", type=float, default=SIM_THRESHOLD)
    p.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--skip_validation", action="store_true")
    args = p.parse_args()

    if not args.skip_validation and not args.openai_key:
        raise RuntimeError("Provide --openai_key or set OPENAI_API_KEY")
    if not args.skip_validation:
        openai.api_key = args.openai_key

    all_df     = pd.read_csv(args.all_csv)
    pending_df = pd.read_csv(args.pending_csv)
    print(f"Loaded {len(all_df)} rows in ALL, {len(pending_df)} rows pending.")

    # ──── text duplicate vs ALL ────
    print("Embedding ALL + pending text …")
    all_emb     = embed_texts([canonicalize(t) for t in all_df[args.all_text_col]])
    pending_emb = embed_texts([canonicalize(t) for t in pending_df[args.pending_text_col]])

    print("Finding near-duplicates vs ALL …")
    near_dup_flags, near_dup_sim, match_id = [], [], []
    sims_all = cosine_similarity(pending_emb, all_emb)   # (n_pending × n_all)
    max_sims = sims_all.max(axis=1)
    idxs     = sims_all.argmax(axis=1)
    for sim, idx in zip(max_sims, idxs):
        near_dup_flags.append(sim >= args.sim_threshold)
        near_dup_sim  .append(float(sim))
        match_id      .append(all_df.iloc[idx][args.all_id_col] if sim >= args.sim_threshold else None)

    pending_df["is_near_duplicate_all"]        = near_dup_flags
    pending_df["near_duplicate_max_similarity"] = near_dup_sim
    pending_df["near_duplicate_all_id"]        = match_id

    # ──── text duplicate within pending ────
    print("Finding near-duplicates within pending …")
    pen_pen_sim = cosine_similarity(pending_emb)        # n_pending × n_pending
    np.fill_diagonal(pen_pen_sim, 0)                    # ignore self
    pending_df["is_near_duplicate_pending"] = (pen_pen_sim.max(axis=1) >= args.sim_threshold)
    pending_df["near_duplicate_pending_idx"] = pen_pen_sim.argmax(axis=1)

    # ──── exact-ID duplicate checks ────
    all_id_cols     = [args.all_id_col]
    pending_id_cols = [args.pending_id_col]
    if args.use_second_id_cols:
        if args.all_id_col2 and args.all_id_col2 in all_df.columns:
            all_id_cols.append(args.all_id_col2)
        if args.pending_id_col2 and args.pending_id_col2 in pending_df.columns:
            pending_id_cols.append(args.pending_id_col2)

    # Normalize IDs
    for c in all_id_cols:
        all_df[c+"_norm"] = all_df[c].apply(normalize_id)
    for c in pending_id_cols:
        pending_df[c+"_norm"] = pending_df[c].apply(normalize_id)

    all_norm_cols     = [c+"_norm" for c in all_id_cols]
    pending_norm_cols = [c+"_norm" for c in pending_id_cols]

    # ---- exact ID dupes vs ALL ----
    all_ids = set(all_df[all_norm_cols].stack().dropna())
    def row_is_dup_vs_all(row) -> bool:
        return any( (nid in all_ids) for nid in row[pending_norm_cols] if nid is not None )
    pending_df["is_exact_duplicate_id_all"] = pending_df.apply(row_is_dup_vs_all, axis=1)

    # ---- exact ID dupes within pending ----
    #  Build a Series of all non-blank IDs in pending
    pen_ids_long = pending_df[pending_norm_cols].stack().dropna()
    dup_flags = pen_ids_long.duplicated(keep=False)      # True for every duplicate entry
    dup_any   = dup_flags.groupby(level=0).any()         # row-level bool
    pending_df["is_exact_duplicate_id_pending"] = pending_df.index.map(dup_any).fillna(False)

    # ──── output ────
    pending_df.to_csv(args.out_csv, index=False)
    print(f"Wrote results → {args.out_csv}")

if __name__ == "__main__":
    main()
