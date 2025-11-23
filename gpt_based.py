import os
import json
from typing import List, Dict, Any, Optional
import concurrent.futures

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# Use gpt-5-mini as requested
MODEL_NAME = "gpt-5-mini"


# ===================== DEBUG: web_search usage =====================


def debug_web_search_usage(response: Any) -> None:
    """
    Inspect the Responses API output and print how many web_search_call
    items were used, their status, and queries.
    """
    data = response.model_dump()
    output_items = data.get("output", []) or []

    total_ws_items = 0
    completed_calls = 0
    details = []

    for item in output_items:
        if item.get("type") == "web_search_call":
            total_ws_items += 1
            status = item.get("status")
            action = item.get("action") or {}
            action_type = action.get("type")
            query = action.get("query")
            if status == "completed":
                completed_calls += 1
            details.append((status, action_type, query))

    print(f"\n[DEBUG] web_search_call items: {total_ws_items}, completed: {completed_calls}")
    for i, (status, action_type, query) in enumerate(details, start=1):
        print(
            f"[DEBUG]  {i}. status={status}, action_type={action_type}, query={query!r}"
        )


# ===================== LLM + web_search part =====================


def search_fashion_with_web(
    user_query: str, max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Use the Responses API with the built-in web_search tool to find fashion products.
    Returns a list of product dicts with keys:
      name, description, price, imageUrl, sourceUrl, tone

    NOTE: imageUrl will often be null here; we enrich it via scraping later.
    """

    system_prompt = """
You are a fashion shopping assistant.

You can use the `web_search` tool to find real products on live e-commerce websites.

The user will give a fashion shopping query (e.g., "kalamkari kurti under 2000").
Your job:
  1. Use web_search to find real products.
  2. Prefer individual product detail pages over generic category/search pages.
     Examples of product detail URLs:
       - Amazon: URLs containing "/dp/" or "/gp/product"
       - Flipkart: URLs containing "/p/" or "pid="
       - Myntra: URLs containing "/buy/"
       - Meesho: product URLs (not generic listings)
  3. Only include products that reasonably match the user's intent (type, style, budget).
  4. Return at most N products (N will be provided in the input).
  5. You CANNOT actually click or open pages beyond what web_search returns; rely only on the tool output.

IMAGE URL RULES:
- If the web_search tool output clearly includes an image URL, thumbnail URL, or main product image URL
  (e.g., metadata fields called "image", "thumbnail", "og:image", etc.), you MAY use that as `imageUrl`.
- If you do NOT see any explicit image URL in web_search output, set `imageUrl` to null.
- NEVER invent or guess an image URL.

OUTPUT FORMAT (STRICT):
- Respond with ONLY a single JSON object.
- No explanations, no markdown, no extra keys.
- The JSON MUST have exactly this shape:

{
  "products": [
    {
      "name": string,
      "description": string,
      "price": number | null,
      "imageUrl": string | null,
      "sourceUrl": string,
      "tone": string | null
    }
  ]
}
"""

    user_payload = {
        "user_query": user_query,
        "max_products": max_results,
    }

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            tools=[
                {
                    "type": "web_search",
                    # low context size = cheaper & faster, fine for product discovery
                    "search_context_size": "low",
                }
            ],
            tool_choice="auto",
            # IMPORTANT: do NOT set JSON mode / response_format here,
            # because that conflicts with web_search.
        )
    except BadRequestError as e:
        print("OpenAI BadRequestError:", e)
        raise

    # --- Debug how many web_search calls were made ---
    debug_web_search_usage(response)

    # Aggregate all assistant text into a single string
    reply_text = (response.output_text or "").strip()

    # ---- Parse JSON from assistant text robustly ----
    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        # Fallback: try to grab the first {...} block
        start = reply_text.find("{")
        end = reply_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = reply_text[start : end + 1]
            parsed = json.loads(json_str)
        else:
            print("⚠️ Model did not return valid JSON. Raw text:")
            print(reply_text)
            raise

    products = parsed.get("products", [])
    if not isinstance(products, list):
        raise RuntimeError("JSON does not contain a 'products' list")

    # Normalize fields and cap to max_results
    normalized: List[Dict[str, Any]] = []
    for p in products[:max_results]:
        normalized.append(
            {
                "name": (p.get("name") or "").strip(),
                "description": (p.get("description") or "").strip(),
                "price": p.get("price"),
                "imageUrl": p.get("imageUrl"),  # will be enriched later
                "sourceUrl": (p.get("sourceUrl") or "").strip(),
                "tone": p.get("tone"),
            }
        )

    return normalized


# ===================== Scraping for image URLs =====================


def _extract_image_url_from_html(html: str, base_url: str) -> Optional[str]:
    """
    Given HTML for a product page, try to extract a good product image URL.
    Priority:
      1. <meta property="og:image"> / twitter:image
      2. <link rel="image_src">
      3. A large-ish <img> that doesn't look like a logo / sprite
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) OpenGraph / Twitter image
    for prop in ["og:image", "twitter:image", "twitter:image:src"]:
        tag = soup.find("meta", property=prop) or soup.find(
            "meta", attrs={"name": prop}
        )
        if tag and tag.get("content"):
            from urllib.parse import urljoin

            return urljoin(base_url, tag["content"].strip())

    # 2) link rel="image_src"
    link_tag = soup.find("link", rel="image_src")
    if link_tag and link_tag.get("href"):
        from urllib.parse import urljoin

        return urljoin(base_url, link_tag["href"].strip())

    # 3) Fallback: pick a likely main <img>
    candidates = []
    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-srcset")
        )
        if not src:
            continue

        # If srcset-style, take the first URL
        src = str(src).split()[0]

        lower_src = src.lower()
        # Skip obvious non-product images
        if any(bad in lower_src for bad in ["sprite", "logo", "icon", "placeholder"]):
            continue

        # Try get size info (helps choose largest image)
        try:
            width = int(img.get("width") or 0)
            height = int(img.get("height") or 0)
        except ValueError:
            width = height = 0

        area = width * height
        if area == 0:
            # some sites don't set width/height; still consider them but at low priority
            area = 1

        candidates.append((area, src))

    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        best_src = candidates[0][1]
        from urllib.parse import urljoin

        return urljoin(base_url, best_src)

    return None


def _fetch_image_for_product(
    product: Dict[str, Any], timeout: int = 8
) -> Dict[str, Any]:
    """
    For a single product dict, try to fill imageUrl by scraping sourceUrl.
    If anything fails, we leave imageUrl as-is.
    """
    url = product.get("sourceUrl")
    if not url:
        return product

    # Don't re-scrape if we already have an imageUrl
    if product.get("imageUrl"):
        return product

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        img_url = _extract_image_url_from_html(resp.text, url)
        if img_url:
            product["imageUrl"] = img_url
    except Exception:
        # It's fine if we can't get an image; just skip
        pass

    return product


def enrich_products_with_images(
    products: List[Dict[str, Any]], max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Run image scraping in parallel for speed.
    Only touches imageUrl field; everything else stays the same.
    """
    if not products:
        return products

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        enriched = list(executor.map(_fetch_image_for_product, products))

    return enriched


# ===================== CLI entry =====================


def main() -> None:
    query = input("Enter fashion query: ").strip()
    products = search_fashion_with_web(query, max_results=5)

    # Scrape image URLs (fast & parallel)
    products = enrich_products_with_images(products, max_workers=5)

    print("\nStructured products:\n")
    for i, p in enumerate(products, start=1):
        print(f"Product {i}:")
        print(json.dumps(p, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
