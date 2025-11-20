import os
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PayloadSchemaType
from dotenv import load_dotenv

load_dotenv()

def build_mem0_qdrant():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_KEY")
    mem_collection = os.getenv("MEM_COLLECTION", "mem0_fashion_qdrant")
    mem_emb_model  = os.getenv("MEM_EMB_MODEL", "text-embedding-3-small")
    mem_emb_dim    = int(os.getenv("MEM_EMB_DIM", "1536"))
    openai_key     = os.getenv("OPENAI_API_KEY")

    qdr = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    if not qdr.collection_exists(mem_collection):
        qdr.create_collection(
            collection_name=mem_collection,
            vectors_config=VectorParams(size=mem_emb_dim, distance=Distance.COSINE),
        )
    for fname in ("user_id","domain"):
        try:
            qdr.create_payload_index(mem_collection, field_name=fname, field_schema=PayloadSchemaType.KEYWORD)
        except Exception:
            pass

    # Mem0 uses OpenAI for its memory embeddings (your OpenAI spend).
    mem = Memory.from_config({
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": qdrant_url,
                "api_key": qdrant_key,
                "collection_name": mem_collection,
                "embedding_model_dims": mem_emb_dim
            }
        },
        "llm": {"provider":"openai","config":{"api_key":openai_key,"model":"gpt-5-nano"}},
        "embedder": {"provider":"openai","config":{"api_key":openai_key,"model":mem_emb_model}},
    })
    return mem, qdr
