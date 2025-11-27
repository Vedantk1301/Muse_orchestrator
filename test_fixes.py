import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from responses_agent import t_classify_intent, t_search_catalog_metadata, Services, Config

async def test_fixes():
    print("🚀 Testing Fixes...")
    await Services.ensure_loaded()
    
    # 1. Test "Try Again" Logic
    print("\n1️⃣ Testing 'Try Again' Intent Classification")
    query = "try again"
    last_context = "traveling to Shimla [queries: wool sweater, puffer jacket]"
    intent = await t_classify_intent(query, last_query_context=last_context)
    print(f"Input: Query='{query}', Context='{last_context}'")
    print(f"Output: {intent}")
    
    # 2. Test Brand Search
    print("\n2️⃣ Testing Brand Search Tool")
    # Test general list
    brands = await t_search_catalog_metadata(field="brand", limit=5)
    print(f"General Brands: {brands}")
    
    # Test specific search
    specific_brand = await t_search_catalog_metadata(query="Fabindia", field="brand")
    print(f"Specific Brand 'Fabindia': {specific_brand}")
    
    print("\n✅ Tests Completed")

if __name__ == "__main__":
    asyncio.run(test_fixes())
