import os
import numpy as np
import httpx
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# DeepInfra / model configuration (env overridable)
DI_OPENAI_BASE = os.getenv("DI_OPENAI_BASE", "https://api.deepinfra.com/v1/openai")
DI_INFER_BASE = os.getenv("DI_INFER_BASE", "https://api.deepinfra.com/v1/inference")
# Default to a lightweight Llama on DeepInfra; override via env if desired.
DI_CHAT_MODEL = os.getenv(
    "DI_CHAT_MODEL",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
)
EMB_MODEL_CATALOG = os.getenv("EMB_MODEL_CATALOG", "Qwen/Qwen3-Embedding-4B")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
EXPECTED_EMBEDDING_DIM = int(os.getenv("EXPECTED_EMBEDDING_DIM", "3840"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "8.0"))

# Lightweight logger helper so we can surface errors to the main agent logger when provided.
def _log_error(logger, message: str, **kwargs):
    if logger:
        try:
            logger.error(message, **kwargs)
            return
        except Exception:
            pass
    print(f"[deepinfra] {message}: {kwargs}")

# You can override with env RERANK_INSTRUCTION if you want to tweak copy.
DEFAULT_RERANK_INSTRUCTION = os.getenv(
    "RERANK_INSTRUCTION",
    (
        """
        You are a fashion product reranker for an Indian shopping assistant.

Input:
- One user query.
- A list of product documents. Each document may include brand, title, category, fabric, fit, price, and occasion tags.

Goal:
Rank the documents so that higher scores mean more relevant to the user query.

Relevance rules:
1) Match product type/category (shirt, kurta, saree, lehenga, anarkali, dress, trousers, co-ord, sneakers, etc.).
2) For clearly ethnic queries (saree, lehenga, anarkali, kurta, sherwani, dupatta, kurta pajama): BOOST ethnic/traditional items; DEMOTE western/athleisure (blazers, polos, joggers, tees) unless explicitly requested.
3) For clearly western queries (blazer, polo, chinos, slip dress, bodycon): avoid returning ethnic/traditional items.
4) Match fabric, fit, colour family, and vibe (casual, work, travel, date, festive, wedding) over exact wording.
5) Respect gender hints (men, women, unisex) when present.
6) If the query mentions budget, prefer products whose INR price is closer to the mentioned range.
7) Exact keyword match is NOT required; semantic, silhouette, and vibe match are more important, but irrelevant categories must be pushed down.

Diversity rules (important for top results):
1) In the top 8 results, prefer at least 3–4 different brands if possible.
2) Avoid placing many items from the same brand consecutively when good alternatives exist (no more than 2 back-to-back from one brand).
3) Prefer mixing silhouettes when multiple types fit the query (e.g., shirts + polos + overshirts) instead of near-duplicates.
4) Diversity should never override clear irrelevance. Always keep unrelated products low.

Scoring guidance:
1) Rank by relevance first, then adjust for diversity/brand spread.
2) Penalize near-duplicate titles from a single brand when alternatives exist.
3) Do not penalize shorter descriptions as long as relevance signals are clear.
4) When the query contains ethnic terms, downrank western/athleisure items even if they have matching words.

Output:
Return scores that, when sorted descending, produce a relevant and diverse ranked list of products.
""".strip()
    )
)

# =========================
# Embeddings
# =========================

async def embed_catalog(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using Qwen3-Embedding-4B via DeepInfra.
    Returns normalized vectors for cosine similarity.
    
    Args:
        texts: List of strings to embed
        
    Returns:
        List of normalized embedding vectors
        
    Raises:
        httpx.HTTPError: If API request fails
        ValueError: If DEEPINFRA_TOKEN not set
    """
    if not texts:
        return []
    
    token = os.getenv("DEEPINFRA_TOKEN")
    if not token:
        raise ValueError("DEEPINFRA_TOKEN not set in environment")
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, trust_env=False) as client:
            response = await client.post(
                f"{DI_OPENAI_BASE}/embeddings",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": EMB_MODEL_CATALOG,
                    "input": texts,
                    "encoding_format": "float"
                }
            )
            response.raise_for_status()
            data = response.json()["data"]
        
        # Extract and normalize embeddings
        embeddings = np.asarray(
            [row["embedding"] for row in data],
            dtype=np.float32
        )
        
        # L2 normalization for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized = embeddings / norms
        
        return normalized.tolist()
    
    except httpx.HTTPStatusError as e:
        print(f"[embed_catalog] HTTP {e.response.status_code} error: {e}")
        raise
    except httpx.TimeoutException as e:
        print(f"[embed_catalog] Timeout error: {e}")
        raise
    except KeyError as e:
        print(f"[embed_catalog] Unexpected API response format: {e}")
        raise
    except Exception as e:
        print(f"[embed_catalog] Unexpected error: {type(e).__name__}: {e}")
        raise


# =========================
# Reranker
# =========================

def _truncate_instruction(text: str, max_len: int = 1900) -> str:
    """
    DeepInfra caps instruction at 2048 chars.
    We keep a little headroom to be safe.
    """
    if not text:
        return text
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len]


async def rerank_qwen(
    query: str,
    documents: List[str],
    top_k: int = 8,
    instruction: Optional[str] = None,
    service_tier: str = "default",
) -> List[int]:
    """
    Rerank documents using Qwen3-Reranker-4B via DeepInfra.
    
    Args:
        query: Search query string
        documents: List of document strings to rerank
        top_k: Number of top results to return
        instruction: Optional reranking instruction to guide the model.
                     If None, uses DEFAULT_RERANK_INSTRUCTION.
        service_tier: DeepInfra service tier ('default' or 'priority')
        
    Returns:
        List of indices in reranked order (best first)
        
    Notes:
        - We send a single query and N documents.
        - DeepInfra's schema says queries/documents should match in length,
          but this API works in practice with [query] + list-of-docs.
    """
    if not documents:
        return []
    
    if len(documents) == 1:
        return [0]  # No reranking needed
    
    token = os.getenv("DEEPINFRA_TOKEN")
    if not token:
        raise ValueError("DEEPINFRA_TOKEN not set in environment")

    # Pick instruction (env override > default) and enforce length
    if instruction is None:
        instruction = DEFAULT_RERANK_INSTRUCTION
    instruction = _truncate_instruction(instruction, max_len=1900)
    
    try:
        payload: dict = {
            "queries": [query],
            "documents": documents,
        }

        # Only attach instruction if it is non-empty
        if instruction:
            payload["instruction"] = instruction

        # Optional: include service_tier if you ever want to use 'priority'
        if service_tier in ("default", "priority"):
            payload["service_tier"] = service_tier
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, trust_env=False) as client:
            response = await client.post(
                f"{DI_INFER_BASE}/{RERANK_MODEL}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
        
        # Extract scores (handle both single query and batch formats)
        scores = result.get("scores", [])
        # Some rerankers return [[...scores per doc...]] for batched queries
        if scores and isinstance(scores[0], list):
            scores = scores[0]
        
        if not scores:
            print("[rerank_qwen] No scores returned, using original order")
            return list(range(len(documents)))
        
        # Sort by score (descending) and return top_k indices
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        return ranked_indices[:min(top_k, len(documents))]
    
    except httpx.HTTPStatusError as e:
        print(f"[rerank_qwen] HTTP {e.response.status_code} error, falling back to original order")
        return list(range(min(top_k, len(documents))))
    except httpx.TimeoutException as e:
        print(f"[rerank_qwen] Timeout error, falling back to original order")
        return list(range(min(top_k, len(documents))))
    except (KeyError, IndexError) as e:
        print(f"[rerank_qwen] Unexpected API response: {e}, falling back")
        return list(range(min(top_k, len(documents))))
    except Exception as e:
        print(f"[rerank_qwen] Unexpected error: {type(e).__name__}: {e}, falling back")
        return list(range(min(top_k, len(documents))))


# =========================
# Utility Functions
# =========================

def validate_embedding_dimension(embeddings: List[List[float]], expected_dim: int = None):
    """
    Validate that embeddings have the expected dimension.
    Qwen3-Embedding-4B produces 3840-dimensional vectors by default.
    """
    expected = expected_dim or EXPECTED_EMBEDDING_DIM
    for i, emb in enumerate(embeddings):
        if len(emb) != expected:
            raise ValueError(
                f"Embedding {i} has dimension {len(emb)}, expected {expected}"
            )


async def batch_embed_catalog(
    texts: List[str],
    batch_size: int = None
) -> List[List[float]]:
    """
    Embed large lists in batches to avoid timeout/memory issues.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts per batch (uses BATCH_SIZE env var if None)
        
    Returns:
        Concatenated list of all embeddings
    """
    batch_size = batch_size or BATCH_SIZE
    all_embeddings: List[List[float]] = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            embeddings = await embed_catalog(batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"[batch_embed_catalog] Batch {i//batch_size} failed: {e}")
            # Add zero vectors as fallback for failed batch
            all_embeddings.extend([[0.0] * EXPECTED_EMBEDDING_DIM for _ in batch])
    
    return all_embeddings


# =========================
# Quick Options Generation
# =========================

async def generate_quick_options(
    prompt: str,
    context: str = "",
    hint: str = None,
    logger=None,
) -> List[str]:
    """
    Generate 3-5 short, clickable options using a fast DeepInfra model.
    
    Args:
        prompt: The bot's message or current context
        context: Additional context (user message, conversation history)
        hint: Optional hint about what kind of options to generate:
            - "product_refinement": After showing products (price filters, colors, pairing)
            - "question": Answering a question (Menswear/Womenswear/Neutral, etc.)
            - None: Auto-detect from prompt
        logger: Optional agent logger for visibility on failures
    """
    token = os.getenv("DEEPINFRA_TOKEN")
    if not token:
        return []

    # Customize system prompt based on hint
    if hint == "product_refinement":
        system_prompt = (
            "You are a UI helper for a fashion chatbot. "
            "The bot just showed products to the user. Generate 3-5 short refinement options. "
            "Rules:\\n"
            "1. Output ONLY a JSON list of strings. Example: [\\\"Under 3k\\\", \\\"Different colors\\\", \\\"What goes well with this\\\"]\\n"
            "2. Keep options very short (2-4 words).\\n"
            "3. Common product refinement options: \\\"Under Xk\\\", \\\"Different colors\\\", \\\"Similar items\\\", \\\"What goes well\\\", \\\"Show more\\\"\\n"
            "4. Do NOT output markdown code blocks, just the raw JSON string.\\n"
            "5. If context unclear, return generic refinements.\\n"
        )
    elif hint == "question":
        system_prompt = (
            "You are a UI helper for a fashion chatbot. "
            "The bot asked the user a question. Generate 3-4 short answer options. "
            "Rules:\\n"
            "1. Output ONLY a JSON list of strings. Example: [\\\"Menswear\\\", \\\"Womenswear\\\", \\\"Neutral\\\"]\\n"
            "2. Keep options very short (1-3 words).\\n"
            "3. Provide direct answers to the question asked.\\n"
            "4. Do NOT output markdown code blocks, just the raw JSON string.\\n"
            "5. If no obvious options, return an empty list [].\\n"
        )
    else:
        # Auto-detect
        system_prompt = (
            "You are a UI helper for a fashion chatbot. "
            "Your job is to read the bot's last message and generate 3-5 short, relevant, clickable options (chips) for the user. "
            "Rules:\\n"
            "1. Output ONLY a JSON list of strings. Example: [\\\"Menswear\\\", \\\"Womenswear\\\", \\\"Neutral\\\"]\\n"
            "2. Keep options very short (1-4 words).\\n"
            "3. If the bot asks a question, provide answer choices.\\n"
            "4. If the bot shows products, provide refinements (e.g. \\\"Under 3k\\\", \\\"Different colors\\\", \\\"What goes well\\\").\\n"
            "5. Do NOT output markdown code blocks, just the raw JSON string.\\n"
            "6. If no obvious options exist, return an empty list [].\\n"
        )

    user_content = f"Bot Message: {prompt}\\n"
    if context:
        user_content += f"Context: {context}\\n"

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, trust_env=False) as client:
            response = await client.post(
                f"{DI_OPENAI_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": DI_CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 100,
                    "response_format": {"type": "json_object"}
                }
            )
            
            if response.status_code != 200:
                _log_error(
                    logger,
                    "generate_quick_options API error",
                    status=response.status_code,
                    response_text=response.text[:500],
                )
                return []
                
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON
            import json
            try:
                # Handle potential wrapping in {"options": [...]} or just [...]
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return parsed[:5]
                if isinstance(parsed, dict):
                    # Look for common keys
                    for key in ["options", "chips", "suggestions", "choices"]:
                        if key in parsed and isinstance(parsed[key], list):
                            return parsed[key][:5]
                    # Fallback: return values if they are strings
                    return [str(v) for v in parsed.values() if isinstance(v, (str, int, float))][:5]
                _log_error(
                    logger,
                    "generate_quick_options unexpected JSON structure",
                    content=str(content)[:500],
                )
                return []
            except json.JSONDecodeError as e:
                _log_error(
                    logger,
                    "generate_quick_options JSON decode failed",
                    error=str(e),
                    content=str(content)[:500],
                )
                return []

    except Exception as e:
        _log_error(logger, "generate_quick_options failed", error=str(e))
        return []
