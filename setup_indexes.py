# setup_catalog_indexes.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType

load_dotenv()

def main():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_KEY")
    catalog_collection = os.getenv("CATALOG_COLLECTION", "fashion_qwen4b_text")
    
    qdr = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    
    if not qdr.collection_exists(catalog_collection):
        print(f"❌ Collection '{catalog_collection}' does not exist!")
        return
    
    print(f"🔧 Setting up indexes for: {catalog_collection}\n")
    
    indexes = [
        ("attributes.gender", PayloadSchemaType.KEYWORD, "Gender filtering"),
        ("commerce.in_stock", PayloadSchemaType.KEYWORD, "Stock filtering"),
        ("category_path", PayloadSchemaType.TEXT, "Category search"),
        ("brand", PayloadSchemaType.KEYWORD, "Brand filtering"),
        ("category_leaf", PayloadSchemaType.KEYWORD, "Leaf category filtering"),
    ]
    
    for field_name, field_type, description in indexes:
        try:
            qdr.create_payload_index(
                collection_name=catalog_collection,
                field_name=field_name,
                field_schema=field_type
            )
            print(f"✅ {field_name:<25} - {description}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️  {field_name:<25} - Already exists")
            else:
                print(f"❌ {field_name:<25} - Error: {e}")
    
    print(f"\n🎉 Setup complete for {catalog_collection}!")

if __name__ == "__main__":
    main()