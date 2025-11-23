# Smart Discovery Search Relevance Plan

## Problem Statement
- Multi-query discovery/pairing flows currently interleave raw Qdrant hits and only rerank when scores look weak, so noisy payloads slip into the catalog list.
- GPT’s final copy recommends mostly-good pieces, but the UI shows mismatched ordering and sometimes irrelevant cards because the retrieval output is not aligned with the LLM reasoning chain.
- We need to **always show at least eight relevant products** (with a hard minimum of 8, and never fewer than 6 from Qdrant when possible) and top up with web results when the catalog is sparse.

## Goals & Guardrails
1. **Deterministic multi-query aggregation** – combine all Qdrant hits from discovery runs, dedupe, and run a single numeric reranker before any UI exposure.
2. **LLM aware ordering** – expose a lightweight tool that takes `title + brand + category_leaf` (sample payload provided) and lets the LLM reorder/drop obviously noisy SKUs so that text + cards stay in sync.
3. **Result count floor** – always surface ≥8 cards; if Qdrant only yields ≤6 good ones, top up the remainder via the existing `gpt_based.search_fashion_with_web` web_search helper.
4. **Zero-regret integration** – preserve current streaming UX, don’t regress latency budgets (45s cap), add clear logs around each new stage for debugging.

## Current Flow Snapshot
1. `t_search_fashion_products` (in `responses_agent.py`) expands the user intent into up to 3 short queries, embeds + searches per query, interleaves results, and only calls `Services.rerank` if the best vector score is below `Config.TAU_NO_RERANK`.
2. The assistant later references `budget.last_products`, so whatever ordering leaves this function becomes the gallery order.
3. `gpt_based.py` already wraps the Responses `web_search` tool and can fetch real SKUs, but nothing auto-invokes it as a fallback.

## Proposed Architecture
### 1. Retrieval & Numeric Rerank (Vector Stage)
- Always accumulate **all** hits from every discovery query into a single candidate pool (`max_candidates = DISCOVERY_QUERIES * PRODUCTS_PER_QUERY`).
- Use richer candidate objects: `{"id", "title", "brand", "category_leaf", "category_path", "url", "primary_image", "source_tags", "score", "from_query"}` – this mirrors the provided Qdrant payload.
- Run `Services.rerank` **unconditionally when `len(queries) > 1`** (and optionally when `top_k > 8`), requesting top 12–16 indices to keep enough for later fallback/Llm pass. Keep the early exit for single-query specific searches.
- Persist metadata needed by later stages (especially `category_leaf`, `from_query`, and `score`) so we can explain rejections.

### 2. LLM Rerank Tool (Reasoning Stage)
- Add `t_rerank_products_llm` as a new tool in `responses_agent.py` that accepts:
  ```json
  { "user_query": "crop tops for brunch", "products": [{ "product_id": "...", "title": "...", "brand": "...", "category_leaf": "...", "score": 0.88 }, ...], "min_results": 8 }
  ```
- Prompt instructs GPT to:
  - Drop items that clearly don’t match the vibe.
  - Return ordered IDs + optional justification so we can log why something was trimmed.
  - Guarantee at least 8 outputs when given ≥8 inputs (otherwise return whatever is left).
- Call this tool **inside `t_search_fashion_products`** only for multi-query searches (after numeric rerank). The assistant doesn’t need to manually invoke it; we keep it server-side for deterministic ordering.
- Output back to the agent as `ordered_products`, preserving the exact sorted list for UI + GPT response copy.

### 3. Web Search Top-Up (External Stage)
- When the post-LLM list has `< 8` entries (or `< 6` purely from Qdrant), call `gpt_based.search_fashion_with_web` to fetch `needed = max(8 - current_count, 2)` extra SKUs.
  - Wrap the sync helper with `await asyncio.to_thread(...)` so we do not block the main loop.
  - Map the web result schema into the same product card structure (`id` can be the source URL, `from_query = "web_search"`).
  - Clearly tag them with `source="web"` so downstream UX can badge them or log separately.
- Merge + dedupe by `url/product_id`, rerun the LLM ordering pass with the augmented list to keep copy and cards consistent.

### 4. Agent / UI Touchpoints
- Update `Config` with knobs:
  - `DISCOVERY_CANDIDATES_PER_QUERY`, `LLM_RERANK_MIN_RESULTS`, `WEB_TOPUP_MIN_COUNT`.
- Ensure `Budget.set_products` and streaming metadata take the post-LLM order so nothing reverts downstream.
- Teach the system prompt (short addition) that the search tool already handles reranking + web fallback so the assistant doesn’t double-ask for products.

## Execution Steps
1. **Refactor search aggregator**
   - Update `t_search_fashion_products` to always build a `candidates` list per above.
   - Move interleave logic into a helper that also records `from_query`.
   - Run numeric rerank whenever `len(queries) > 1` or `best_score < TAU_NO_RERANK`.
2. **Implement `t_rerank_products_llm`**
   - Add helper in `responses_agent.py` plus tool schema entry (even if auto-called).
   - Prompt includes the provided sample payload as documentation so GPT knows the fields.
   - Return a dict with `products` already sorted; surface logging for reasoning + fallbacks.
3. **Wire LLM rerank inside search**
   - After numeric rerank, call the helper with `user_query`, top-N candidates, and `min_results=8`.
   - If the call fails or returns nonsense, fall back to numeric order (log warning).
4. **Integrate web search fallback**
   - Import `search_fashion_with_web` from `gpt_based.py`.
   - When the LLM-ranked list has `< 8` products and at least one discovery query ran, fetch extras, normalize, append, and rerun the LLM ranking to keep coherence.
5. **Telemetry & safeguards**
   - Add structured logs: `rerank_vector_ms`, `rerank_llm_ms`, `web_topup_count`.
   - Surface counters inside `logger.perf` to compare latency before/after.
   - Update tests/CLI smoke (`test_cli`, `test_streaming`) to print product counts so we can manual-verify.

## Risks & Mitigations
- **Latency creep** – two rerank stages + possible web search adds seconds. Mitigate by capping candidate size (≤30), running numeric rerank concurrently with other tasks if possible, and only invoking web search when strictly necessary.
- **LLM hallucination in ordering** – enforce strict schema + fallback to numeric order on JSON errors; log bad outputs for inspection.
- **Duplicate SKUs** – dedupe via `product_id`/`url` before LLM rerank and again after web top-up to avoid repeated cards.
- **Web top-up mismatches** – annotate `source="web"` so copywriters / UI can choose to badge them or hide later.

## Definition of Done
1. Discovery search consistently returns ≥8 cards with clearly higher topical relevance in internal smoke tests.
2. Logs show `rerank_vector` and `rerank_llm` timing for every multi-query request plus web fallback usage counts.
3. GPT responses and gallery order stay aligned because both read from the same LLM-ranked list.
4. When Qdrant is sparse, at least two web-search products appear, tagged and deduped, without blocking the stream for >6 seconds.
