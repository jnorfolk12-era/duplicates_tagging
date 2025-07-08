#!/usr/bin/env python3

import argparse, os, time
from pathlib import Path
from datetime import datetime

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# -------------------------------- configuration
PROMPT = """You are a validator responsible for determining whether a math problem is CLEAN or FAILS quality standards.

A problem is considered CLEAN only if it does **not** exhibit any of the following issues:
1. Extraction Error – The problem or its solution has been extracted incorrectly...
2. Wrong or Unsolvable Problem – The problem is invalid, has incorrect logic, or cannot be solved as written...
3. Estimation or Multi-Answer Problem – The task has multiple valid answers...
4. Missing Figure or Media – The problem explicitly refers to a figure...
5. Not a Math Problem – The content is unrelated to mathematics...
6. Multiple Problems – There is more than one question included...

Instructions:
1. Carefully analyze the math problem.
2. If any of the five issues above are present, return exactly: FAIL
3. If none of the issues are present and the problem is self-contained, well-defined, and mathematically valid, return exactly: CLEAN

Do not explain your answer. Only return CLEAN or FAIL.

What follows is the math problem:
"""
MODEL = "gpt-4o-mini"
N_THREADS = 8
SIM_THRESHOLD = 0.90
EMBED_MODEL = "all-mpnet-base-v2"

def canonicalize(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("'", "'")
        .replace(""", '"').replace(""", '"')
        .replace("\u2212", "-").replace("\xa0", " ")
    )

def call_openai_once(problem_text: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT + problem_text}],
                temperature=0.0,
                max_tokens=1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

def pass3_vote(problem_text: str) -> tuple[str, list[str]]:
    votes = [call_openai_once(problem_text) for _ in range(3)]
    clean_count = votes.count("CLEAN")
    verdict = "CLEAN" if clean_count >= 2 else "FAIL"
    return verdict, votes

def embed_texts(texts):
    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

def main():
    p = argparse.ArgumentParser(
        description="Screen pending problems for near-duplicates vs all existing problems, allowing for different ID and text column names in each file."
    )
    p.add_argument("--all_csv", required=True, help="CSV with all existing problems")
    p.add_argument("--pending_csv", required=True, help="CSV with new pending problems to check")

    # NEW: Allow different columns for each file
    p.add_argument("--all_id_col", default="#", help="ID column in all_csv")
    p.add_argument("--all_id_col2", default=None, help="Optional second ID column in all_csv")
    p.add_argument("--all_text_col", default="problem", help="Text column in all_csv")
    p.add_argument("--pending_id_col", default="#", help="ID column in pending_csv")
    p.add_argument("--pending_id_col2", default=None, help="Optional second ID column in pending_csv")
    p.add_argument("--pending_text_col", default="problem", help="Text column in pending_csv")

    p.add_argument("--out_csv", default="pending_checked.csv", help="Output CSV with duplicate flags")
    p.add_argument("--sim_threshold", type=float, default=SIM_THRESHOLD,
                   help=f"Cosine similarity threshold for near-duplicate detection (default: {SIM_THRESHOLD})")
    p.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY"),
                   help="OpenAI API key (or set env var)")
    p.add_argument("--skip_validation", action="store_true", help="Skip pass@3 validation")
    args = p.parse_args()

    # Set up OpenAI if needed
    if not args.skip_validation and not args.openai_key:
        raise RuntimeError("Provide --openai_key or set OPENAI_API_KEY (or use --skip_validation)")
    if not args.skip_validation:
        openai.api_key = args.openai_key

    # Load data
    all_df = pd.read_csv(args.all_csv)
    pending_df = pd.read_csv(args.pending_csv)
    print(f"Loaded {len(all_df)} rows from all_csv, {len(pending_df)} from pending_csv.")

    # Get col names
    all_id_col = args.all_id_col
    all_id_col2 = args.all_id_col2
    all_text_col = args.all_text_col
    pending_id_col = args.pending_id_col
    pending_id_col2 = args.pending_id_col2
    pending_text_col = args.pending_text_col

    # Optionally validate pending problems (pass@3)
    if not args.skip_validation:
        print("Running pass@3 validation on pending problems...")
        texts = pending_df[pending_text_col].astype(str).tolist()
        verdicts = []
        votes = []
        for t in texts:
            v, vlist = pass3_vote(t)
            verdicts.append(v)
            votes.append(vlist)
        pending_df["LLM_verdict"] = verdicts
        pending_df["LLM_votes"] = votes
        pending_df = pending_df[pending_df["LLM_verdict"] == "CLEAN"].reset_index(drop=True)
        print(f"Validation kept {len(pending_df)} pending problems.")

    # Deduplication: compare each pending problem against all problems in all_df
    print("Computing embeddings for all problems and pending problems...")
    all_canon = [canonicalize(t) for t in all_df[all_text_col]]
    pending_canon = [canonicalize(t) for t in pending_df[pending_text_col]]
    all_emb = embed_texts(all_canon)
    pending_emb = embed_texts(pending_canon)

    print("Checking pending for near-duplicates...")
    near_dup_flags = []
    near_dup_sim = []
    match_id_in_all = []

    for i, pen_emb in enumerate(pending_emb):
        sims = cosine_similarity([pen_emb], all_emb)[0]
        max_sim = sims.max()
        match_idx = sims.argmax()
        is_near_dupe = max_sim >= args.sim_threshold
        near_dup_flags.append(is_near_dupe)
        near_dup_sim.append(max_sim)
        match_id_in_all.append(all_df.iloc[match_idx][all_id_col] if is_near_dupe else None)

    pending_df["is_near_duplicate"] = near_dup_flags
    pending_df["near_duplicate_max_similarity"] = near_dup_sim
    pending_df["near_duplicate_all_id"] = match_id_in_all

    # ADD: flag exact ID duplicates (now supports 2 columns per file)
    all_id_cols = [all_id_col]
    pending_id_cols = [pending_id_col]
    if all_id_col2 and all_id_col2 in all_df.columns:
        all_id_cols.append(all_id_col2)
    if pending_id_col2 and pending_id_col2 in pending_df.columns:
        pending_id_cols.append(pending_id_col2)

    # Make sets of all existing IDs
    all_ids = set()
    for col in all_id_cols:
        all_ids.update(all_df[col].astype(str))

    # Mark as duplicate if any ID in pending matches any ID in all
    def is_duplicate_row(row):
        return any(str(row[col]) in all_ids for col in pending_id_cols)

    pending_df["is_exact_duplicate_id"] = pending_df.apply(is_duplicate_row, axis=1)

    # Save output
    pending_df.to_csv(args.out_csv, index=False)
    print(f"Wrote pending file with near-duplicate flags to: {args.out_csv}")

if __name__ == "__main__":
    main()
