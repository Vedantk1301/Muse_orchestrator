# Changes Summary

1. **Catalog Retrieval Pipeline (`responses_agent.py`)**
   - Added config knobs for target result count and rerank pool sizes.
   - Introduced helper utilities for interleaving, deduping, numeric reranking (Qwen), LLM-based reranking, and web-search fallbacks.
   - Refactored `t_search_fashion_products` to always run numeric + LLM reranks, guarantee at least 8 products, and top up with `search_fashion_with_web` when catalog is sparse, with detailed performance logging.

2. **External Web Search Integration**
   - Imported `search_fashion_with_web` so catalog search can invoke the existing Responses web-search helper asynchronously whenever extra products are required.

3. **Planning Artifact**
   - Added `plan.md` describing the strategy, guardrails, and execution steps for the multi-stage reranking pipeline and web fallback logic.

4. **Display & Prompt Alignment**
   - Limited the UI feed to the top 8 ranked products while retaining a deeper ranked pool (`all_products`) for reuse, and updated both the tool schema and system prompt so the LLM makes a single `search_fashion_products` call, relying on the built-in rerank + web fallback.

5. **Discovery Query Fix**
   - Added `_ensure_discovery_queries` so forced discovery/pairing searches break comma-separated user requests into multiple short queries before embedding, ensuring the reranked gallery matches the conversational bullet order.

6. **UI Ordering + Hidden Pool**
   - Updated the budget tracker to keep the UI-bound list (`last_products`) separate from the deeper cache, ensuring the frontend always shows the in-order top 8 while still storing the wider `_all_products` pool internally (stripped before the LLM sees the tool output).

7. **Discovery Debug Logs**
   - Added detailed logging for LLM rerank ordering plus the final display list so we can trace any mismatches between the bullets and gallery, while still letting the classifier decide when a specific vs. discovery search is appropriate (unless explicitly forced).

8. **Discovery Query Expansion**
   - Discovery searches now call a lightweight FAST-model helper when the classifier returns duplicate queries, ensuring we always get 3 diverse short product strings (instead of repeating the same phrase) before embedding.

9. **Order & UX Polish**
   - Added ranked metadata plus stronger prompt guidance so the final assistant response must follow the catalog order, restored occasion-first option chips, improved image fallbacks by pulling from payload `images`, tightened bullet instructions (≤10 words) and reinforced discovery-query diversity via LLM expansion.
