"""
Quick debug runner for Qdrant search with switchable rerank modes (vector / LLM / hybrid / none).

Usage examples:
  python debug_qdrant_runner.py --query "date outfits" --gender Womenswear
  python debug_qdrant_runner.py --query "traditional festive wear" --gender Womenswear --category ethnic
  python debug_qdrant_runner.py --queries "silk saree with blouse,lehenga choli set for party" --gender Womenswear --category ethnic
  python debug_qdrant_runner.py --query "date outfits" --gender Womenswear --rerank-mode llm
"""

import argparse
import asyncio
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import logging
import sys
import os
from qdrant_client import QdrantClient
from services.deepinfra import embed_catalog, rerank_qwen

from qdrant_client.http import models as rest

from responses_agent import (
    Config,
    Services,
    _brand_histogram,
    _brand_key,
    _dedupe_products,
    _interleave_results,
    _product_identity,
    _numeric_rerank_products,
    _rebalance_brand_pool,
    _llm_rerank_products,
)

try:
    from responses_agent import t_classify_intent
except Exception:
    t_classify_intent = None

# Module-level logger placeholder; initialized in __main__
LOG = logging.getLogger("debug_qdrant")
LOG_FILE = "logs/debug_qdrant.log"

# Ensure UTF-8 stdout to avoid emoji encoding errors on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass


async def ensure_loaded_light():
    if getattr(Services, "_loaded", False):
        return
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_KEY")
    Services.qdr = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    Services.embed = embed_catalog
    Services.rerank = rerank_qwen
    Services._loaded = True
    LOG.info("Loaded light services (qdrant + deepinfra)")

def _gender_value(user_gender: Optional[str]) -> Optional[str]:
    if not user_gender:
        return None
    return {
        "menswear": "men",
        "men": "men",
        "male": "men",
        "womenswear": "women",
        "women": "women",
        "female": "women",
        "neutral": "unisex",
        "unisex": "unisex",
    }.get(user_gender.lower())


def _looks_like_filter_error(msg: str) -> bool:
    msg = (msg or "").lower()
    return any(
        kw in msg
        for kw in [
            "index required",
            "wrong input",
            "keyword",
            "field",
            "matchany",
            "match any",
        ]
    )


async def _search_one(
    qdr_client,
    vec,
    gender_filter_value: Optional[str],
    category_filter: Optional[str],
    limit: int,
    include_match_any: bool = True,
):
    def _make_filter(use_match_any: bool, drop_category: bool = False):
        must_conditions = []
        should_conditions = []

        cat = category_filter.strip().lower() if category_filter else None
        if cat and not drop_category:
            should_conditions.append(
                rest.FieldCondition(key="category_path", match=rest.MatchText(text=cat))
            )
            if use_match_any:
                should_conditions.append(
                    rest.FieldCondition(key="category_path", match=rest.MatchAny(any=[cat]))
                )
            should_conditions.append(
                rest.FieldCondition(key="category_leaf", match=rest.MatchText(text=cat))
            )

        if gender_filter_value:
            must_conditions.append(
                rest.FieldCondition(
                    key="attributes.gender", match=rest.MatchValue(value=gender_filter_value)
                )
            )

        if must_conditions or should_conditions:
            LOG.info(
                "Applying qdrant filter (debug runner)",
                extra={
                    "category_filter": cat,
                    "gender_filter": gender_filter_value,
                    "must": len(must_conditions),
                    "should": len(should_conditions),
                    "match_any": use_match_any,
                },
            )
            return rest.Filter(must=must_conditions or None, should=should_conditions or None)
        return None

    # Try sequence:
    # 1) with category + gender, match_any
    # 2) with category + gender, no match_any (if filter error)
    # 3) gender only (if filter error)
    # 4) no filter (last resort)
    try:
        return await asyncio.to_thread(
            qdr_client.query_points,
            collection_name=Config.CATALOG_COLLECTION,
            query=vec,
            limit=limit,
            with_payload=True,
            search_params=rest.SearchParams(hnsw_ef=Config.HNSW_EF),
            query_filter=_make_filter(include_match_any, drop_category=False),
        )
    except Exception as e:
        msg = str(e)
        if include_match_any and _looks_like_filter_error(msg):
            LOG.warning("Retrying qdrant query without MatchAny filter", extra={"error": msg})
            try:
                return await asyncio.to_thread(
                    qdr_client.query_points,
                    collection_name=Config.CATALOG_COLLECTION,
                    query=vec,
                    limit=limit,
                    with_payload=True,
                    search_params=rest.SearchParams(hnsw_ef=Config.HNSW_EF),
                    query_filter=_make_filter(False, drop_category=False),
                )
            except Exception as e2:
                msg2 = str(e2)
                if _looks_like_filter_error(msg2) and category_filter:
                    LOG.warning("Retrying qdrant query with gender only (dropping category)", extra={"error": msg2})
                    try:
                        return await asyncio.to_thread(
                            qdr_client.query_points,
                            collection_name=Config.CATALOG_COLLECTION,
                            query=vec,
                            limit=limit,
                            with_payload=True,
                            search_params=rest.SearchParams(hnsw_ef=Config.HNSW_EF),
                            query_filter=_make_filter(False, drop_category=True),
                        )
                    except Exception as e3:
                        msg3 = str(e3)
                        if _looks_like_filter_error(msg3):
                            LOG.warning("Retrying qdrant query without any filter", extra={"error": msg3})
                            return await asyncio.to_thread(
                                qdr_client.query_points,
                                collection_name=Config.CATALOG_COLLECTION,
                                query=vec,
                                limit=limit,
                                with_payload=True,
                                search_params=rest.SearchParams(hnsw_ef=Config.HNSW_EF),
                                query_filter=None,
                            )
                        raise
                raise
        raise


def _summarize_products(products: List[Dict[str, object]], top_k: int = 8) -> List[Dict[str, object]]:
    summary = []
    for p in products[:top_k]:
        summary.append(
            {
                "title": p.get("title"),
                "brand": p.get("brand"),
                "score": round(float(p.get("score", 0.0)), 4),
                "from_query": p.get("from_query"),
                "category_path": p.get("category_path"),
                "category_leaf": p.get("category_leaf"),
            }
        )
    return summary


def _dump_payloads(products: List[Dict[str, object]], path: str, limit: int = 20):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(products[:limit], f, ensure_ascii=False, indent=2)
        LOG.info("Saved payload dump", extra={"path": path, "count": min(len(products), limit)})
    except Exception as e:
        LOG.error("Failed to save payload dump", extra={"path": path, "error": str(e)})


def _brand_cap(products: List[Dict[str, object]], max_per_brand: int = 4) -> List[Dict[str, object]]:
    if max_per_brand <= 0:
        return products
    seen: Dict[str, int] = {}
    capped = []
    for p in products:
        b = _brand_key(p)
        cnt = seen.get(b, 0)
        if cnt >= max_per_brand:
            continue
        seen[b] = cnt + 1
        capped.append(p)
    return capped


async def _vector_rerank_slim(user_query: str, products: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if len(products) <= 1:
        return products

    pool_limit = min(Config.NUMERIC_RERANK_POOL, len(products))
    pool = products[:pool_limit]
    remainder = products[pool_limit:]

    texts = [
        f"{p.get('title') or ''} {p.get('category') or p.get('category_leaf') or ''}".strip()
        for p in pool
    ]
    rerank_t0 = time.perf_counter()
    try:
        indices = await Services.rerank(
            user_query,
            texts,
            top_k=min(len(pool), Config.NUMERIC_RERANK_POOL),
        )
        ordered: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for idx in indices or []:
            if idx is None or not (0 <= idx < len(pool)):
                continue
            product = pool[idx]
            pid = _product_identity(product)
            if pid and pid in seen:
                continue
            if pid:
                seen.add(pid)
            ordered.append(product)

        for product in pool:
            pid = _product_identity(product)
            if pid and pid in seen:
                continue
            if pid:
                seen.add(pid)
            ordered.append(product)

        ordered.extend(remainder)
        rerank_ms = int((time.perf_counter() - rerank_t0) * 1000)
        LOG.info(
            "rerank_vector_slim",
            extra={"ms": rerank_ms, "pool": len(pool), "total": len(products)},
        )
        return ordered
    except Exception as e:
        LOG.warning("Vector rerank failed (slim)", extra={"error": str(e)})
        return products


async def debug_search(
    query: str,
    user_gender: Optional[str],
    category_filter: Optional[str],
    search_type: str,
    manual_queries: Optional[List[str]],
    top_k: int,
    rebalance: bool = True,
    dump_path: Optional[str] = None,
    rerank_mode: str = "vector",  # vector | llm | hybrid | none
):
    await ensure_loaded_light()

    queries = [q.strip() for q in manual_queries or [] if q.strip()]
    if not queries:
        if search_type in ("auto", "discovery", "pairing") and t_classify_intent:
            intent = await t_classify_intent(query, user_gender, forced_type=None if search_type == "auto" else search_type)
            queries = intent.get("queries") or [query]
            LOG.info("Intent queries", extra={"search_type": intent.get("search_type"), "queries": queries})
        else:
            queries = [query]

    gender_val = _gender_value(user_gender)
    LOG.info(
        "Running debug search",
        extra={
            "query": query,
            "user_gender": user_gender,
            "gender_filter": gender_val,
            "category_filter": category_filter,
            "queries": queries,
        },
    )

    embed_t0 = time.perf_counter()
    vectors = await Services.embed(queries)
    LOG.info(
        "embed_debug",
        extra={"ms": int((time.perf_counter() - embed_t0) * 1000), "num_queries": len(queries)},
    )

    per_query = Config.PRODUCTS_PER_QUERY if len(queries) > 1 else Config.SIMPLE_SEARCH_LIMIT
    search_t0 = time.perf_counter()
    results = await asyncio.gather(
        *[_search_one(Services.qdr, v, gender_val, category_filter, per_query) for v in vectors],
        return_exceptions=True,
    )
    LOG.info(
        "qdrant_search_debug",
        extra={"ms": int((time.perf_counter() - search_t0) * 1000), "num_queries": len(queries)},
    )

    results_lists: List[List[Dict[str, object]]] = []
    for i, (q, res) in enumerate(zip(queries, results)):
        if isinstance(res, Exception):
            LOG.error("Query failed", extra={"idx": i, "error": repr(res)})
            continue
        q_products: List[Dict[str, object]] = []
        for point in (res.points or [])[:per_query]:
            payload = point.payload or {}
            commerce = payload.get("commerce") or {}
            pid = payload.get("product_id") or payload.get("id") or getattr(point, "id", None)
            if not pid:
                continue
            pid = str(pid)
            image_url = payload.get("primary_image") or payload.get("image_url")
            if not image_url:
                images = payload.get("images")
                if isinstance(images, list) and images:
                    image_url = images[0]
            card = {
                "id": pid,
                "product_id": pid,
                "title": payload.get("title"),
                "brand": payload.get("brand"),
                "category": payload.get("category_leaf"),
                "category_leaf": payload.get("category_leaf"),
                "category_path": payload.get("category_path"),
                "image_url": image_url,
                "url": payload.get("url"),
                "price": commerce.get("price"),
                "price_inr": commerce.get("price_inr"),
                "score": float(point.score),
                "from_query": q,
                "source_tags": payload.get("source_tags") or [],
            }
            q_products.append(card)
        LOG.info(
            "Raw Qdrant slice",
            extra={"query": q, "count": len(q_products), "sample": _summarize_products(q_products, min(5, len(q_products)))},
        )
        results_lists.append(q_products)

    interleaved = _interleave_results(results_lists)
    candidates = _dedupe_products(interleaved)
    LOG.info("Candidate pool", extra={"count": len(candidates), "brand_hist": _brand_histogram(candidates, 50)})

    if rebalance and candidates:
        candidates = _rebalance_brand_pool(candidates, Config.BRAND_CAP_PER_WINDOW, Config.BRAND_CAP_WINDOW)
        LOG.info("After rebalance", extra={"brand_hist": _brand_histogram(candidates, 50)})

    if dump_path:
        _dump_payloads(candidates, dump_path, limit=50)

    reranked = candidates
    if candidates:
        if rerank_mode == "vector":
            reranked = await _vector_rerank_slim(query, candidates)
            reranked = _brand_cap(reranked, max_per_brand=4)
        elif rerank_mode == "llm":
            base = _brand_cap(candidates, max_per_brand=4)
            reranked = await _llm_rerank_products(query, base, top_k)
        elif rerank_mode == "hybrid":
            vectored = await _vector_rerank_slim(query, candidates)
            vectored = _brand_cap(vectored, max_per_brand=4)
            reranked = await _llm_rerank_products(query, vectored, top_k)
        elif rerank_mode == "none":
            reranked = candidates
        else:
            LOG.warning("Unknown rerank_mode, defaulting to vector", extra={"rerank_mode": rerank_mode})
            reranked = await _vector_rerank_slim(query, candidates)
            reranked = _brand_cap(reranked, max_per_brand=4)

    LOG.info(
        "Ranked top",
        extra={"mode": rerank_mode, "top": _summarize_products(reranked, top_k)},
    )

    return candidates, reranked


async def _run_scenarios(
    scenarios: List[Dict[str, object]],
    top_k: int,
    rebalance: bool,
):
    all_summaries = {}
    for sc in scenarios:
        LOG.info("=== Running scenario ===", extra={"label": sc.get("label"), "mode": sc.get("rerank_mode")})
        raw, ranked = await debug_search(
            query=sc["query"],
            user_gender=sc.get("gender"),
            category_filter=sc.get("category"),
            search_type=sc.get("search_type", "auto"),
            manual_queries=sc.get("manual_queries"),
            top_k=top_k,
            rebalance=rebalance,
            dump_path=sc.get("dump_path"),
            rerank_mode=sc.get("rerank_mode", "vector"),
        )

        raw_top = _summarize_products(raw, top_k)
        rerank_top = _summarize_products(ranked, top_k)
        base_label = sc.get("label") or sc.get("query") or "run"
        label = f"{base_label}_{sc.get('rerank_mode', 'vector')}"

        print(f"\n=== {label.upper()} RAW (direct qdrant, interleaved) ===")
        print(json.dumps(raw_top, indent=2, ensure_ascii=False))

        print(f"\n=== {label.upper()} RERANK ({sc.get('rerank_mode', 'vector')}) ===")
        print(json.dumps(rerank_top, indent=2, ensure_ascii=False))

        all_summaries[label] = {
            "raw_top": raw_top,
            "rerank_top": rerank_top,
            "rerank_mode": sc.get("rerank_mode", "vector"),
        }
    return all_summaries


async def main():
    # Default no-CLI run: execute three rerank modes across three scenarios.
    if len(sys.argv) == 1:
        modes = ["vector", "llm", "hybrid"]
        base_scenarios = [
            {
                "label": "date_womenswear",
                "query": "date outfits",
                "gender": "Womenswear",
                "category": None,
                "search_type": "discovery",
            },
            {
                "label": "festive_ethnic_womenswear",
                "query": "traditional festive wear",
                "gender": "Womenswear",
                "category": "ethnic",
                "search_type": "discovery",
            },
            {
                "label": "travel_menswear",
                "query": "summer travel outfits",
                "gender": "Menswear",
                "category": None,
                "search_type": "discovery",
            },
        ]

        scenarios = []
        for sc in base_scenarios:
            for mode in modes:
                run = dict(sc)
                run["rerank_mode"] = mode
                run["dump_path"] = f"logs/{sc['label']}_{mode}_dump.json"
                scenarios.append(run)

        all_summaries = await _run_scenarios(scenarios, top_k=10, rebalance=True)
        with open(LOG_FILE.replace(".log", "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
            print(f"\nSaved summaries to {f.name}")
        return

    parser = argparse.ArgumentParser(description="Debug Qdrant search with configurable rerank modes")
    parser.add_argument("--query", required=False, help="Base user query text")
    parser.add_argument("--gender", default=None, help="Menswear, Womenswear, or Neutral")
    parser.add_argument("--category", default=None, help="Category filter text (e.g., ethnic)")
    parser.add_argument(
        "--search-type",
        default="auto",
        choices=["auto", "discovery", "pairing", "specific"],
        help="How to expand queries",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="Comma-separated manual queries (skips intent classifier when set)",
    )
    parser.add_argument("--top", type=int, default=8, help="Number of top products to display in summary")
    parser.add_argument("--no-rebalance", action="store_true", help="Skip brand rebalance step")
    parser.add_argument("--dump-path", default=None, help="Write raw interleaved payloads (JSON) for inspection")
    parser.add_argument(
        "--rerank-mode",
        default="vector",
        choices=["vector", "llm", "hybrid", "none"],
        help="vector: Qwen rerank only; llm: LLM rerank only; hybrid: vector then LLM; none: no rerank",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run three preset scenarios (date, festive ethnic, travel) sequentially. Ignores --query/--queries.",
    )
    args = parser.parse_args()

    scenarios = []
    if args.batch:
        scenarios = [
            {
                "label": "date_womenswear",
                "query": "date outfits",
                "gender": "Womenswear",
                "category": None,
                "search_type": "discovery",
                "rerank_mode": args.rerank_mode,
                "dump_path": "logs/batch_date_dump.json",
            },
            {
                "label": "festive_ethnic_womenswear",
                "query": "traditional festive wear",
                "gender": "Womenswear",
                "category": "ethnic",
                "search_type": "discovery",
                "rerank_mode": args.rerank_mode,
                "dump_path": "logs/batch_festive_dump.json",
            },
            {
                "label": "travel_menswear",
                "query": "summer travel outfits",
                "gender": "Menswear",
                "category": None,
                "search_type": "discovery",
                "rerank_mode": args.rerank_mode,
                "dump_path": "logs/batch_travel_dump.json",
            },
        ]
    else:
        if not args.query and not args.queries:
            parser.error("Must provide --query or --queries unless --batch is set.")
        scenarios = [
            {
                "label": "single_run",
                "query": args.query,
                "gender": args.gender,
                "category": args.category,
                "search_type": args.search_type,
                "manual_queries": [q.strip() for q in args.queries.split(",")] if args.queries else None,
                "rerank_mode": args.rerank_mode,
                "dump_path": args.dump_path,
            }
        ]

    all_summaries = await _run_scenarios(
        scenarios,
        top_k=args.top,
        rebalance=not args.no_rebalance,
    )

    with open(LOG_FILE.replace(".log", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summaries to {f.name}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = f"logs/debug_qdrant_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
    LOG = logging.getLogger("debug_qdrant")
    asyncio.run(main())
