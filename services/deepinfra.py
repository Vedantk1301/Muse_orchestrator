import os
import numpy as np
import httpx
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# =========================
# Configuration
# =========================
DI_OPENAI_BASE = os.getenv("DI_OPENAI_BASE", "https://api.deepinfra.com/v1/openai")
DI_INFER_BASE = os.getenv("DI_INFER_BASE", "https://api.deepinfra.com/v1/inference")

EMB_MODEL_CATALOG = os.getenv("EMB_MODEL_CATALOG", "Qwen/Qwen3-Embedding-4B")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
REQUEST_TIMEOUT = int(os.getenv("DEEPINFRA_TIMEOUT", "90"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EXPECTED_EMBEDDING_DIM = int(os.getenv("EXPECTED_EMBEDDING_DIM", "3840"))

# Default rerank instruction – short and domain-specific.
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
1. Match product type (shirt, kurta, dress, trousers, co-ord, sneakers, etc.).
2. Match fabric, fit, colour family, and the overall vibe (casual, work, travel, date, festive).
3. Respect gender hints (men, women, unisex) when present.
4. If the query mentions budget, prefer products whose INR price is closer to the mentioned range.
5. Exact keyword match is NOT required; semantic, silhouette, and vibe match are more important.

Diversity rules (important for top results):
1. In the top 8 results, prefer at least 3–4 different brands if possible.
2. Avoid placing many items from the same brand consecutively when good alternatives exist.
3. Prefer mixing silhouettes when multiple types fit the query (e.g., a couple of shirts, a polo, a lightweight overshirt), instead of returning 8 near-duplicates.
4. Diversity should never override clear irrelevance. Always keep unrelated products low.

Scoring guidance:
- Give higher scores to documents that best balance query relevance and diversity.
- Do not overvalue repeated wording or near-duplicate titles from a single brand.
- Do not penalize shorter descriptions as long as relevance signals are clear.

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
