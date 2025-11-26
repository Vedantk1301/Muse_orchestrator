"""
Debug utility to inspect raw Qdrant search results (no reranker).

Edit the CONFIG below to change queries/filters/limits, then run:
  python debug_qdrant_raw.py

Environment:
  QDRANT_URL, QDRANT_KEY must be set.
  DeepInfra embedding uses services.deepinfra.embed_catalog.
"""

import asyncio
import os
import json
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from services.deepinfra import embed_catalog, rerank_qwen
from responses_agent import Config

# =============================
# EDIT THESE VALUES TO TEST
# =============================
CONFIG = {
    "queries": [
        # Add or change queries here
        "silk saree with blouse",
        "lehenga choli set",
        "anarkali suit with dupatta",
    ],
    "gender": None,  # Set to None to disable gender filter
    "category_filter": None,
    "limit": 20,
    "ef": Config.HNSW_EF,
    # Leave instruction None to use the default from services.deepinfra
    "rerank_instruction": None,
}


def build_filter(category_filter: Optional[str], gender: Optional[str]):
    must_conditions = []
    should_conditions = []

    cat = category_filter.strip().lower() if category_filter else None
    if cat:
        cat_condition = rest.FieldCondition(
            key="category_path",
            match=rest.MatchText(text="ethnic" if "ethnic" in cat else cat),
        )
        # Keep simple: ethnic as must, others as should
        if "ethnic" in cat:
            must_conditions.append(cat_condition)
        else:
            should_conditions.append(cat_condition)

    gender_val = None
    if gender:
        gender_val = {
            "menswear": "men",
            "men": "men",
            "male": "men",
            "womenswear": "women",
            "women": "women",
            "female": "women",
            "neutral": "unisex",
            "unisex": "unisex",
        }.get(gender.lower())

    if gender_val:
        must_conditions.append(
            rest.FieldCondition(
                key="attributes.gender",
                match=rest.MatchValue(value=gender_val),
            )
        )

    if not must_conditions and not should_conditions:
        return None

    return rest.Filter(must=must_conditions or None, should=should_conditions or None)


async def search_queries(
    queries: List[str],
    limit: int,
    category_filter: Optional[str],
    gender: Optional[str],
    ef: int,
    include_unfiltered: bool = False,
):
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    vectors = await embed_catalog(queries)
    q_filter = build_filter(category_filter, gender)

    summary = {}
    if include_unfiltered:
        summary["_unfiltered"] = {}

    async def _query_one(vec, use_filter: bool):
        return await asyncio.to_thread(
            client.query_points,
            collection_name=Config.CATALOG_COLLECTION,
            query=vec,
            limit=limit,
            with_payload=True,
            search_params=rest.SearchParams(hnsw_ef=ef),
            query_filter=q_filter if use_filter else None,
        )

    # Run Qdrant searches in parallel
    filtered_tasks = [
        asyncio.create_task(_query_one(vec, True)) for vec in vectors
    ]
    unfiltered_tasks = (
        [asyncio.create_task(_query_one(vec, False)) for vec in vectors]
        if include_unfiltered
        else []
    )

    filtered_results = await asyncio.gather(*filtered_tasks)
    unfiltered_results = await asyncio.gather(*unfiltered_tasks) if include_unfiltered else []

    # Per-query rerank in parallel
    rerank_tasks = []
    for q, result in zip(queries, filtered_results):
        docs = []
        for point in (result.points or []):
            payload = point.payload or {}
            doc = " ".join(
                str(x)
                for x in [
                    payload.get("title") or "",
                    payload.get("category_path") or "",
                ]
                if x
            )
            docs.append(doc)
        if docs:
            rerank_tasks.append(
                asyncio.create_task(
                    rerank_qwen(
                        q,
                        docs,
                        top_k=len(docs),
                        instruction=CONFIG["rerank_instruction"],
                    )
                )
            )
        else:
            rerank_tasks.append(asyncio.create_task(asyncio.sleep(0, result=[])))

    rerank_indices_list = await asyncio.gather(*rerank_tasks)

    for idx_q, (q, result) in enumerate(zip(queries, filtered_results)):
        raw_list = []
        for point in (result.points or []):
            payload = point.payload or {}
            raw_list.append(
                {
                    "title": payload.get("title"),
                    "brand": payload.get("brand"),
                    "score": float(point.score),
                    "category_path": payload.get("category_path"),
                }
            )
        summary[q] = {"raw": raw_list, "rerank": []}

        rerank_indices = rerank_indices_list[idx_q] or []
        for i in rerank_indices:
            if 0 <= i < len(raw_list):
                summary[q]["rerank"].append(raw_list[i])

        print(f"\n=== Query: {q} | filtered returned {len(raw_list)} ===")
        print("Top 5 RAW:")
        for j, item in enumerate(raw_list[:5]):
            print(f"  {j+1:02d}. {item['brand']} | {item['title']} | {item['category_path']} | {item['score']:.4f}")
        print("Top 5 RERANK:")
        for j, item in enumerate(summary[q]["rerank"][:5]):
            print(f"  {j+1:02d}. {item['brand']} | {item['title']} | {item['category_path']} | {item['score']:.4f}")

    if include_unfiltered:
        for q, result in zip(queries, unfiltered_results):
            raw_list = []
            for point in (result.points or []):
                payload = point.payload or {}
                raw_list.append(
                    {
                        "title": payload.get("title"),
                        "brand": payload.get("brand"),
                        "score": float(point.score),
                        "category_path": payload.get("category_path"),
                    }
                )
            summary["_unfiltered"][q] = raw_list
            print(f"--- Query: {q} | UNFILTERED returned {len(raw_list)} ---")
            for j, item in enumerate(raw_list[:5]):
                print(f"  {j+1:02d}. {item['brand']} | {item['title']} | {item['category_path']} | {item['score']:.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dumps_dir = Path("dumps")
    dumps_dir.mkdir(exist_ok=True)
    out_path = dumps_dir / f"debug_qdrant_raw_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    asyncio.run(
        search_queries(
            queries=CONFIG["queries"],
            limit=CONFIG["limit"],
            category_filter=CONFIG["category_filter"],
            gender=CONFIG["gender"],
            ef=CONFIG["ef"],
            include_unfiltered=True,
        )
    )
