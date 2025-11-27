"""
production_agent.py - PRODUCTION Fashion Bot with Full Observability
---------------------------------------------------------------------
COMPLETE VERSION - All features, full error handling, comprehensive logging
(Responses API only)
"""

from __future__ import annotations

import os
import json
import asyncio
import time
import hashlib
import traceback
import re
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, AsyncGenerator, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import requests
from openai import OpenAI
from dotenv import load_dotenv

from gpt_based import search_fashion_with_web

load_dotenv()

# =============================================================================
# Logging Setup - File + Console
# =============================================================================
class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"bot_{self.session_id}.log"
        self.perf_file = self.log_dir / f"perf_{self.session_id}.json"
        self.perf_data: List[Dict] = []

    def _write(self, level: str, msg: str, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        parts = [f"[{timestamp}]", f"[{level}]", msg]
        if kwargs:
            parts.append(f"| {json.dumps(kwargs, default=str)}")
        log_line = " ".join(parts)

        # Console (color coded)
        colors = {
            "ERROR": "\033[91m",
            "WARNING": "\033[93m",
            "SUCCESS": "\033[92m",
            "INFO": "\033[94m",
            "DEBUG": "\033[90m",
            "PERF": "\033[95m",
        }
        color = colors.get(level, "")
        reset = "\033[0m" if color else ""
        print(f"{color}{log_line}{reset}")

        # File logging
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception:
            # Never crash on logging failure
            pass

    def info(self, msg: str, **kwargs):
        self._write("INFO", msg, **kwargs)

    def success(self, msg: str, **kwargs):
        self._write("SUCCESS", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._write("WARNING", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._write("ERROR", msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        if Config.DEBUG:
            self._write("DEBUG", msg, **kwargs)

    def perf(self, operation: str, duration_ms: int, **kwargs):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_ms": duration_ms,
            **kwargs,
        }
        self.perf_data.append(entry)
        self._write("PERF", f"{operation}: {duration_ms}ms", **kwargs)

    def save_perf(self):
        try:
            with open(self.perf_file, "w", encoding="utf-8") as f:
                json.dump(self.perf_data, f, indent=2)
        except Exception:
            pass


logger = Logger()

# =============================================================================
# Configuration
# =============================================================================
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MAIN_MODEL = os.getenv("AGENT_MODEL", "gpt-5-mini")
    FAST_MODEL = os.getenv("NANO_MODEL", "gpt-5-nano")

    REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")

    MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "10"))
    MAX_LATENCY_MS = int(os.getenv("MAX_LATENCY_MS", "45000"))

    CATALOG_COLLECTION = os.getenv("CATALOG_COLLECTION", "new_qwen_embeddings")
    HNSW_EF = int(os.getenv("HNSW_EF", "256"))

    SIMPLE_SEARCH_LIMIT = 15
    DISCOVERY_QUERIES = int(os.getenv("DISCOVERY_QUERIES", "4"))
    PRODUCTS_PER_QUERY = 40
    RERANK_PER_QUERY = int(os.getenv("RERANK_PER_QUERY", "25"))
    RERANK_MIN_PER_QUERY = int(os.getenv("RERANK_MIN_PER_QUERY", "2"))
    FINAL_RERANK_TOP_K = 16
    DISPLAY_PRODUCTS_COUNT = int(os.getenv("DISPLAY_PRODUCTS_COUNT", "8"))
    MIN_PRODUCTS_TARGET = int(os.getenv("MIN_PRODUCTS_TARGET", "8"))
    CATALOG_MIN_RESULTS = int(os.getenv("CATALOG_MIN_RESULTS", "4"))  # below this, ask for web approval
    NUMERIC_RERANK_POOL = int(os.getenv("NUMERIC_RERANK_POOL", "40"))
    RERANK_POOL_BRAND_CAP = int(os.getenv("RERANK_POOL_BRAND_CAP", "2"))
    LLM_RERANK_INPUT_LIMIT = int(os.getenv("LLM_RERANK_INPUT_LIMIT", "20"))
    BRAND_CAP_PER_WINDOW = int(os.getenv("BRAND_CAP_PER_WINDOW", "6"))
    BRAND_CAP_WINDOW = int(os.getenv("BRAND_CAP_WINDOW", "32"))
    WEB_TOPUP_MIN_COUNT = int(os.getenv("WEB_TOPUP_MIN_COUNT", "2"))
    USE_LLM_RERANK = os.getenv("USE_LLM_RERANK", "0") == "1"
    USE_LLM_RERANK = False
    USE_WEB_TOPUP = os.getenv("USE_WEB_TOPUP", "0") == "1"
    MAX_PER_BRAND_DISPLAY = int(os.getenv("MAX_PER_BRAND_DISPLAY", "4"))
    CHECK_IMAGES = os.getenv("CHECK_IMAGES", "0") == "1"

    SEARCH_CACHE_TTL_HOURS = 24
    INTENT_CACHE_TTL_SECONDS = 1800

    # Trends: keep in memory and on disk for ~24 hours
    TRENDS_CACHE_TTL_SECONDS = int(os.getenv("TRENDS_CACHE_TTL_SECONDS", "86400"))
    WEATHER_CACHE_TTL_SECONDS = int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "600"))

    # External services
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # optional

    TAU_NO_RERANK = 0.86
    TAU_EARLY_EXIT = 0.90

    DEBUG = os.getenv("MUSEBOT_DEBUG", "1") == "1"


client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Trend cache on disk (JSON)
TREND_CACHE_FILE = Path(os.getenv("TREND_CACHE_FILE", "cache/fashion_trends.json"))

# Streaming metadata prefix
META_CHUNK_PREFIX = "__META__"

# =============================================================================
# TTL Cache
# =============================================================================
class TTLCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._store: Dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._store:
                ts, val = self._store[key]
                if time.time() - ts < self.ttl:
                    return val
                # expired
                del self._store[key]
        return None

    async def set(self, key: str, val: Any):
        async with self._lock:
            self._store[key] = (time.time(), val)
            # light GC
            if len(self._store) > 1000:
                cutoff = time.time() - self.ttl
                self._store = {
                    k: v for k, v in self._store.items() if v[0] > cutoff
                }



def cheap_ack(message: str) -> Optional[str]:
    """
    Returns a simple static ACK if the message is a greeting.
    Returns None if it's a task/query (so we skip ACK).
    """
    import random
    text = message.lower().strip()
    
    # Greetings -> fast ACK
    greetings = {"hi", "hey", "hello", "hola", "yo", "heya"}
    if text in greetings or any(text.startswith(g) for g in ["hi ", "hey ", "hello "]):
        return random.choice(["Hi! 🙂", "Hey there! 👋", "Hello! 😊", "Hey! ✨"])
        
    # Everything else -> No ACK (return None)
    return None


SEARCH_CACHE = TTLCache(Config.SEARCH_CACHE_TTL_HOURS * 3600)
INTENT_CACHE = TTLCache(Config.INTENT_CACHE_TTL_SECONDS)
TRENDS_CACHE = TTLCache(Config.TRENDS_CACHE_TTL_SECONDS)
WEATHER_CACHE = TTLCache(Config.WEATHER_CACHE_TTL_SECONDS)


def _cache_key(*parts) -> str:
    m = hashlib.md5()
    for p in parts:
        m.update(str(p).encode("utf-8"))
    return m.hexdigest()


# =============================================================================
# Services (Qdrant + Mem0 + Rerank)
# =============================================================================
class Services:
    mem = None
    qdr = None
    embed = None
    rerank = None

    _lock = asyncio.Lock()
    _loaded = False

    @classmethod
    async def ensure_loaded(cls):
        if cls._loaded:
            return
        async with cls._lock:
            if cls._loaded:
                return

            logger.info("🔄 Loading services...")
            t0 = time.perf_counter()
            try:
                try:
                    from services.mem0_qdrant import build_mem0_qdrant
                    from services.deepinfra import embed_catalog, rerank_qwen
                except ImportError:
                    from services.mem0_qdrant import build_mem0_qdrant
                    from services.deepinfra import embed_catalog, rerank_qwen

                def _build():
                    return build_mem0_qdrant()

                cls.mem, cls.qdr = await asyncio.to_thread(_build)
                cls.embed = embed_catalog
                cls.rerank = rerank_qwen
                cls._loaded = True

                ms = int((time.perf_counter() - t0) * 1000)
                logger.success("✅ Services loaded in %dms" % ms)
            except Exception as e:
                logger.error(f"❌ Service loading failed: {e}")
                logger.error(traceback.format_exc())
                raise


# =============================================================================
# Tavily Trends (cached, JSON on disk, LLM summarised + short)
# =============================================================================
async def _tavily_search_titles(query: str, topic: str = "general") -> List[Dict[str, Any]]:
    """
    Minimal Tavily wrapper. Returns up to 4 result dicts {title, url, snippet}.
    Fails gracefully and returns [] on error.
    """
    if not Config.TAVILY_API_KEY:
        logger.warning("⚠️ Tavily API key missing, skipping web trends", query=query)
        return []

    def _do_request():
        headers = {"Authorization": f"Bearer {Config.TAVILY_API_KEY}"}
        payload = {
            "query": query,
            "topic": topic,
            "search_depth": "basic",
            "max_results": 4,
            "include_answer": False,
            "include_images": False,
        }
        resp = requests.post(
            "https://api.tavily.com/search",
            headers=headers,
            json=payload,
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json()

    try:
        data = await asyncio.to_thread(_do_request)
        results = data.get("results") or []
        trimmed = []
        for r in results[:4]:
            trimmed.append(
                {
                    "title": (r.get("title") or "").strip(),
                    "snippet": (r.get("content") or "").strip(),
                    "url": r.get("url"),
                }
            )
        return trimmed
    except Exception as e:
        logger.warning("trend query failed", query=query, error=str(e))
        return []


async def get_fashion_trends_text() -> str:
    """
    Returns a compact trend summary string that can be injected as a system message.

    Caching:
    - In memory via TRENDS_CACHE for Config.TRENDS_CACHE_TTL_SECONDS (default 24h).
    - On disk in JSON at TREND_CACHE_FILE for 24h, so process restarts do not re-hit Tavily.

    NEW: Uses FAST_MODEL to compress Tavily results into a SHORT, structured summary:
        Western / global casual:
        - ...
        - ...

        Indian / festive:
        - ...
        - ...

        Use this only for light trend flavour, not strict truth.
    """
    cache_key = "fashion_trends_v2"

    # 1) In-memory cache
    cached_mem = await TRENDS_CACHE.get(cache_key)
    if cached_mem:
        logger.debug("💾 trend cache HIT (memory)")
        return cached_mem

    # 2) Disk cache
    try:
        if TREND_CACHE_FILE.exists():
            with open(TREND_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts_str = data.get("timestamp")
            text = data.get("text")
            if ts_str and text:
                ts = datetime.fromisoformat(ts_str)
                if datetime.utcnow() - ts < timedelta(hours=24):
                    logger.debug("💾 trend disk cache HIT", file=str(TREND_CACHE_FILE))
                    await TRENDS_CACHE.set(cache_key, text)
                    return text
    except Exception as e:
        logger.warning("trend disk cache read failed", error=str(e))

    # 3) Fetch fresh via Tavily and compress with LLM
    logger.info("🌐 Fetching fashion trends via Tavily")
    t0 = time.perf_counter()

    western_results = await _tavily_search_titles(
        "current fashion trends western casual and streetwear India 2025", topic="news"
    )
    ethnic_results = await _tavily_search_titles(
        "current fashion trends ethnic and traditional wear India 2025", topic="news"
    )

    # Prepare compact JSON for LLM
    raw_context = {
        "western": western_results,
        "ethnic": ethnic_results,
    }

    system_prompt = """
You are a fashion trend summariser for an India first stylist bot.

You will receive a small JSON structure with web results for:
- western casual / streetwear
- Indian ethnic / festive

Your job:
- Output a SHORT, structured, human readable summary that the stylist bot can lightly sprinkle into answers.
- Max ~100–120 words in total.

OUTPUT FORMAT (exact structure, no extra commentary):

Western / global casual:
- point 1
- point 2
- point 3

Indian / festive:
- point 1
- point 2
- point 3

Then ONE final line:
Use this only for light trend flavour, not strict truth.

Rules:
- Max 3 bullet points per section.
- Talk about silhouettes, fabrics, and categories like:
  - oversized tees, Korean trousers, co ord sets, straight fit pants
  - pastel kurtas, Nehru jackets, light embroidery, printed sets
- Do NOT mention article titles, URLs, publishers, dates, or celebrity names.
- No markdown backticks or JSON, only the plain text as shown above.
""".strip()

    try:
        resp = await asyncio.to_thread(
            client.responses.create,
            model=Config.FAST_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(raw_context, ensure_ascii=False)},
            ],
            max_output_tokens=260,
        )
        summary = (resp.output_text or "").strip()
        if not summary:
            raise ValueError("Empty LLM summary")
        text = summary
    except Exception as e:
        logger.warning("trend LLM summary failed, using fallback", error=str(e))
        text = (
            "Western / global casual:\n"
            "- Relaxed oversized tees and shirts with clean Korean style trousers\n"
            "- Straight and wide leg pants with minimal sneakers\n"
            "- Co ord sets, muted earthy tones and soft knits\n\n"
            "Indian / festive:\n"
            "- Pastel and earthy kurtas with subtle embroidery\n"
            "- Lightweight Nehru jackets and kurta co ord sets\n"
            "- Simple sherwanis and juttis with minimal details\n\n"
            "Use this only for light trend flavour, not strict truth."
        )

    ms = int((time.perf_counter() - t0) * 1000)
    logger.perf("tavily_trends", ms, queries=2)

    # Save to memory
    await TRENDS_CACHE.set(cache_key, text)

    # Save to disk
    try:
        TREND_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TREND_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "text": text,
                },
                f,
                indent=2,
            )
        logger.success("✅ trend disk cache saved", file=str(TREND_CACHE_FILE))
    except Exception as e:
        logger.warning("trend disk cache write failed", error=str(e))

    return text


# =============================================================================
# Weather via Open-Meteo (tool)
# =============================================================================
_WEATHER_CODE_MAP = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy with rime",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "heavy rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "heavy thunderstorm with hail",
}


async def t_get_weather(city: str, country: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current weather snapshot for a city using Open-Meteo.
    Only meant to help with outfit and packing suggestions.
    """
    cache_key = _cache_key("weather", city.lower(), (country or "").lower())
    cached = await WEATHER_CACHE.get(cache_key)
    if cached:
        logger.debug("💾 weather cache HIT", city=city, country=country)
        return cached

    t0 = time.perf_counter()
    logger.info("🌦️ get_weather", city=city, country=country)

    query_name = city if not country else f"{city}, {country}"

    def _geo_request():
        params = {
            "name": query_name,
            "count": 1,
            "language": "en",
            "format": "json",
        }
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params=params,
            timeout=6,
        )
        resp.raise_for_status()
        return resp.json()

    def _weather_request(lat: float, lon: float, timezone: str):
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": timezone or "auto",
        }
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=6,
        )
        resp.raise_for_status()
        return resp.json()

    try:
        geo = await asyncio.to_thread(_geo_request)
        results = geo.get("results") or []
        if not results:
            logger.warning("⚠️ No geocoding results", city=city, country=country)
            result = {
                "city": city,
                "country": country,
                "error": "location_not_found",
            }
            await WEATHER_CACHE.set(cache_key, result)
            return result

        g0 = results[0]
        lat = float(g0.get("latitude"))
        lon = float(g0.get("longitude"))
        resolved_city = g0.get("name") or city
        resolved_country = g0.get("country") or country
        timezone = g0.get("timezone") or "auto"

        weather_raw = await asyncio.to_thread(
            _weather_request, lat, lon, timezone
        )
        current = (weather_raw.get("current") or {}) if weather_raw else {}

        temp = current.get("temperature_2m")
        feels = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind_ms = current.get("wind_speed_10m")
        code = current.get("weather_code")
        desc = _WEATHER_CODE_MAP.get(int(code)) if code is not None else None

        wind_kmh = None
        if wind_ms is not None:
            try:
                wind_kmh = float(wind_ms) * 3.6
            except Exception:
                wind_kmh = None

        result = {
            "city": resolved_city,
            "country": resolved_country,
            "latitude": lat,
            "longitude": lon,
            "source": "open-meteo",
            "current": {
                "temperature_c": temp,
                "feels_like_c": feels,
                "humidity_percent": humidity,
                "wind_speed_kmh": wind_kmh,
                "weather_code": code,
                "summary": desc,
            },
        }

        ms = int((time.perf_counter() - t0) * 1000)
        logger.success("✅ get_weather", duration_ms=ms, city=resolved_city)
        await WEATHER_CACHE.set(cache_key, result)
        return result

    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.error("❌ get_weather failed", duration_ms=ms, error=str(e))
        logger.error(traceback.format_exc())
        result = {
            "city": city,
            "country": country,
            "error": str(e),
        }
        await WEATHER_CACHE.set(cache_key, result)
        return result


# =============================================================================
# Budget Tracker
# =============================================================================
@dataclass
class Budget:
    max_calls: int
    max_latency_ms: int
    calls_used: int = 0
    latency_ms: int = 0
    tool_log: List[Dict] = field(default_factory=list)
    # keep last search output for frontend
    last_search_result: Dict[str, Any] = field(default_factory=dict)
    last_products: List[Dict[str, Any]] = field(default_factory=list)
    aggregated_products: "OrderedDict[str, Dict[str, Any]]" = field(default_factory=OrderedDict)
    last_options: List[str] = field(default_factory=list)
    user_gender: Optional[str] = None  # 🆕 Add this


    def can_call(self, tool_name: str = "") -> bool:
        has_budget = (
            self.calls_used < self.max_calls
            and self.latency_ms < self.max_latency_ms
        )
        if not has_budget:
            logger.warning(
                f"❌ Budget check failed for {tool_name}",
                calls_used=self.calls_used,
                max_calls=self.max_calls,
                latency_ms=self.latency_ms,
                max_latency=self.max_latency_ms,
            )
        return has_budget

    def consume(self, tool_name: str, ms: int, success: bool = True):
        self.calls_used += 1
        self.latency_ms += ms
        self.tool_log.append(
            {
                "tool": tool_name,
                "duration_ms": ms,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }
        )
        logger.debug(
            "📊 Budget update",
            tool=tool_name,
            duration_ms=ms,
            total_calls=self.calls_used,
            total_latency=self.latency_ms,
        )

    def remaining_ms(self) -> int:
        return max(0, self.max_latency_ms - self.latency_ms)

    def get_summary(self):
        return {
            "calls_used": self.calls_used,
            "latency_ms": self.latency_ms,
            "tool_log": self.tool_log,
            "last_options": self.last_options,
        }

    def record_products(
        self,
        display_products: List[Dict[str, Any]],
        all_products: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Track product lists so the frontend can render the latest display set
        while still caching a deeper pool for follow-ups.
        """
        self.last_products = display_products or []
        source = all_products or display_products or []
        for product in source:
            key = self._product_key(product)
            if key not in self.aggregated_products:
                self.aggregated_products[key] = product

    def set_options(self, options: List[str]):
        """
        Track the most recent set of options/suggestions to surface in the UI.
        """
        cleaned = []
        for option in options:
            if option is None:
                continue
            text = str(option).strip()
            if not text:
                continue
            cleaned.append(text)
        if cleaned:
            self.last_options = cleaned

    def _product_key(self, product: Dict[str, Any]) -> str:
        pid = (
            product.get("product_id")
            or product.get("id")
            or product.get("url")
        )
        if pid:
            return str(pid)

        # Fallback to a stable hash when no obvious identifier exists
        raw = json.dumps(product, sort_keys=True, default=str)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_all_products(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        base = list(self.last_products) or list(self.aggregated_products.values())
        if limit is not None and base:
            return base[:limit]
        return base


# =============================================================================
# Profile tools (kept internal for future use)
# =============================================================================
async def t_profile_read(user_id: str) -> Dict[str, Any]:
    """
    Currently NOT exposed to the model but kept here for future use.
    """
    t0 = time.perf_counter()
    logger.info("🔍 profile_read", user_id=user_id)
    try:
        await Services.ensure_loaded()

        def _search():
            return Services.mem.search(
                "user profile: name, gender, preferences",
                user_id=user_id,
                limit=8,
            )

        results = await asyncio.wait_for(
            asyncio.to_thread(_search), timeout=3.0
        )

        profile: Dict[str, Any] = {
            "name": None,
            "gender": "unknown",
            "preferences": [],
        }

        for r in results.get("results", []):
            mem = r.get("text", "") or r.get("memory", "")
            ml = mem.lower()

            if "name is" in ml:
                import re

                m = re.search(r"name is (\w+)", mem, re.IGNORECASE)
                if m:
                    profile["name"] = m.group(1).capitalize()

            if any(w in ml for w in ["male", "man", "guy"]):
                profile["gender"] = "male"
            elif any(w in ml for w in ["female", "woman", "girl"]):
                profile["gender"] = "female"

            if any(w in ml for w in ["love", "prefer", "like"]):
                profile["preferences"].append(mem[:100])

        ms = int((time.perf_counter() - t0) * 1000)
        logger.success("✅ profile_read completed", duration_ms=ms, profile=profile)
        return profile
    except asyncio.TimeoutError:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("⏱️ profile_read timeout", duration_ms=ms)
        return {
            "name": None,
            "gender": "unknown",
            "preferences": [],
            "error": "timeout",
        }
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.error("❌ profile_read failed", duration_ms=ms, error=str(e))
        return {
            "name": None,
            "gender": "unknown",
            "preferences": [],
            "error": str(e),
        }


async def t_profile_write(
    user_id: str,
    name: Optional[str] = None,
    gender: Optional[str] = None,
    preference: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Also not exposed for now; kept for future use.
    """
    t0 = time.perf_counter()
    logger.info("💾 profile_write", user_id=user_id, name=name, gender=gender)
    saved: List[str] = []
    try:
        await Services.ensure_loaded()

        if name:
            Services.mem.add(
                messages=[{"role": "user", "content": f"My name is {name}"}],
                user_id=user_id,
                metadata={"type": "name"},
                infer=False,
            )
            saved.append("name")

        if gender and gender.lower() in ["male", "female", "other"]:
            Services.mem.add(
                messages=[
                    {"role": "user", "content": f"I identify as {gender}"}
                ],
                user_id=user_id,
                metadata={"type": "gender"},
                infer=False,
            )
            saved.append("gender")

        if preference:
            Services.mem.add(
                messages=[{"role": "user", "content": preference}],
                user_id=user_id,
                metadata={"type": "preference"},
                infer=True,
            )
            saved.append("preference")

        ms = int((time.perf_counter() - t0) * 1000)
        logger.success("✅ profile_write saved", duration_ms=ms, saved=saved)
        return {"status": "saved", "saved_items": saved}
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.error("❌ profile_write failed", duration_ms=ms, error=str(e))
        return {"status": "error", "saved_items": saved, "error": str(e)}


# =============================================================================
# Intent classification (fashion-only, no budget terms inside queries)
# =============================================================================

# ---------- HELPER: build system prompt with hard diversity rules ----------
INTENT_SYSTEM_PROMPT = """
You are a fashion search intent classifier for an India-first stylist bot.

You receive a short JSON like:
{"query": "...", "gender_context": "for women" | "for men" | "", "forced_type": "specific" | "discovery" | "pairing" | null, "product_context": "..." | null}

You must output ONLY a JSON object (no prose, no markdown, no backticks) in this format:

{
  "search_type": "specific" | "discovery" | "pairing",
  "queries": ["string", "string", "string"]
}

Rules for search_type:
- "specific": User clearly wants one concrete item (e.g. "red linen dress", "white shirt").
- "discovery": User wants to browse looks, outfits, or options for a vibe/occasion (e.g. "date", "office wear", "festive").
- "pairing": User wants items that go WITH something they already have (e.g. "what to wear with navy trousers").

If the input has "forced_type", you MUST set "search_type" to that exact value.

Rules for "queries":
- Always output exactly 3 strings.
- Each query max 6 words.
- They must be realistic shopping queries: "[fabric/fit/colour] + [item] + [short modifier]".
  Examples of shape only (do not copy words):
    "satin wrap midi dress"
    "linen wide leg trousers"
    "cotton kurta with palazzo"

Light diversity:
- For general or discovery queries, try to vary:
  - garment category (dress vs top+bottom vs jumpsuit/co-ord),
  - silhouette (mini/midi/maxi, fitted vs relaxed),
  - or vibe (casual, polished, festive).
- Do NOT output three near-identical phrases.

Gender context:
- If "gender_context" == "for women", use womenswear silhouettes.
- If "gender_context" == "for men", use menswear silhouettes.
- If empty or missing, stay neutral.

QUERY CONTEXT (REFINEMENTS - CRITICAL):
- If "last_query_context" is provided:
  - Check if the current query is a PURE REFINEMENT (price filter, color, fabric) or a NEW TOPIC.
  - PURE REFINEMENTS: "under 2000", "in blue", "cotton only", "size M"
  - NEW TOPICS: "party wear", "office clothes", "travel to Tokyo"
  
- If PURE REFINEMENT:
  - The user wants to FILTER the previous search, not start over.
  - Look at the EXISTING queries from last_query_context.
  - APPEND the filter to each existing query.
  - Example: Context="traveling to Dubai [queries: linen maxi dress, cotton kurta palazzo]", Query="under 2000"
    -> Output queries=["linen maxi dress under 2000", "cotton kurta palazzo under 2000", "breathable jumpsuit under 2000"]

- If "TRY AGAIN" / "RETRY" / "SOMETHING ELSE":
  - The user liked the INTENT but hated the specific results.
  - Keep the SAME search_type and general topic.
  - Generate 3 NEW, DIFFERENT queries for the same intent.
  - Do NOT repeat the queries from last_query_context.
  - Example: Context="traveling to Shimla [queries: wool sweater, puffer jacket]", Query="try again"
    -> Output queries=["thermal lined hoodie", "fleece jacket", "merino wool base layer"]
  
- If NEW TOPIC:
  - Ignore last_query_context and generate fresh queries for the new topic.

PRODUCT CONTEXT (CRITICAL):
- If "product_context" is provided (e.g. "User just saw: Blue linen shirt"), and the query is context-dependent (e.g. "pair this", "what goes with it", "shoes for this"):
  - Use the product context to generate specific pairing queries.
  - Example: Context="Blue linen shirt", Query="pair this" -> Output queries=["beige chinos", "white sneakers", "navy trousers"]
- If the query is NOT context-dependent (e.g. "red dress"), ignore the product context.

INDIA-FIRST LOGIC (VERY IMPORTANT):

1) DATE OUTFITS (MODERN INDIAN WOMEN)
- If the query mentions "date", "date night", "coffee date", "first date" and does NOT contain words like
  "ethnic", "traditional", "saree", "lehenga", "kurta":
  - Treat it as a MODERN DATE look.
  - For WOMEN:
    - At least 2 of the 3 queries MUST be WESTERN date pieces:
      - dresses (slip, wrap, fit and flare, bodycon, midi/mini),
      - jumpsuits / playsuits,
      - co-ord sets,
      - nice tops with skirts or trousers.
    - The 3rd can also be western. Only use ethnic (e.g. kurta set) if the user explicitly mentions ethnic/traditional.
  - DO NOT output saree, lehenga or heavy kurta sets for a normal "date" or "date night" query unless the text clearly asks for ethnic.

Examples of good womenswear date queries (shapes only):
- "black slip mini dress"
- "floral midi wrap dress"
- "satin co ord set"
- "chiffon blouse with skirt"

2) FESTIVE / WEDDING / ETHNIC QUERIES (INDIAN ETHNIC ONLY)
- If the query mentions any of:
  "festive", "wedding", "sangeet", "mehendi", "reception",
  "diwali", "navratri", "ethnic day", "traditional day", "puja", "eid":
  - Treat it as an ETHNIC occasion.
  - ALL 3 queries MUST be clearly Indian ethnic outfits, no western gowns or jumpsuits unless the user explicitly says "western festive" or similar.

  For WOMEN, prefer:
    - saree with blouse,
    - lehenga choli set,
    - anarkali suit,
    - sharara set,
    - kurta with palazzo or straight pants,
    - lightweight festive co-ord sets in Indian fabrics/prints.

  For MEN, prefer:
    - kurta pajama,
    - kurta with churidar or straight pants,
    - sherwani,
    - bandhgala,
    - Nehru jacket over kurta.

  You MUST NOT output:
    - western jumpsuits,
    - cocktail gowns,
    - generic "party dress" style queries
  for these festive / ethnic keywords.

3) NEUTRAL / OTHER OCCASIONS
- For office, casual, travel, party (without explicit "festive"/"ethnic" words):
  - Mix appropriate categories for that vibe (shirts, trousers, dresses, co-ords, jeans, etc.).
  - Still keep Indian context in mind (fabrics like cotton, linen, light layers for heat, etc.).

SPECIAL CASE: SPECIFIC ITEM QUERIES
- If the user clearly names one item type ("summer dress", "linen shirt"):
  - Keep all 3 queries in that item family, but vary colour/fabric/length.

Examples of output (shape only):

Input:
{"query": "date night outfit", "gender_context": "for women"}
Output:
{"search_type": "discovery",
 "queries": [
   "black satin slip dress",
   "floral wrap midi dress",
   "chiffon blouse with skirt"
 ]}

Input:
{"query": "festive wear for cousin wedding", "gender_context": "for women"}
Output:
{"search_type": "discovery",
 "queries": [
   "embroidered lehenga choli set",
   "silk saree with blouse",
   "anarkali suit with dupatta"
 ]}

REMINDERS:
- Obey "forced_type" if present.
- Return ONLY valid JSON.
- No extra text, no markdown, no comments.
""".strip()



# ---------- HELPER: extract raw text from Responses API result ----------
def _extract_text_from_response(response: Any) -> str:
    """
    Tries to robustly extract the model's text output from a Responses API object.
    Adjust this if your SDK's structure differs slightly.
    """
    # 1. New Responses-style: response.output -> list of blocks
    output = getattr(response, "output", None)
    if output:
        chunks: List[str] = []
        try:
            for block in output:
                btype = getattr(block, "type", None)

                # Typical reasoning output: type == "message", with .content list
                if btype == "message":
                    contents = getattr(block, "content", []) or []
                    for c in contents:
                        ctype = getattr(c, "type", None)

                        # For reasoning models: content.type == "output_text" with nested .output_text.text
                        if ctype == "output_text":
                            ot = getattr(c, "output_text", None)
                            if ot is not None:
                                txt = getattr(ot, "text", None)
                                if isinstance(txt, str):
                                    chunks.append(txt)
                        # Fallback: plain text
                        elif ctype == "text":
                            txt = getattr(c, "text", None)
                            if isinstance(txt, str):
                                chunks.append(txt)

                # Some variants may have top-level output_text blocks
                elif btype == "output_text":
                    txt = getattr(block, "text", None)
                    if isinstance(txt, str):
                        chunks.append(txt)

            if chunks:
                raw = "".join(chunks).strip()
                if raw:
                    return raw
        except Exception as e:
            logger.warning("Failed to parse response.output structure", error=str(e))

    # 2. Legacy or convenience attribute: response.output_text
    ot = getattr(response, "output_text", None)
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    # 3. Nothing we understand
    return ""


# ---------- HELPER: clean JSON text & parse ----------
def _clean_json_text(raw_text: str) -> str:
    """
    Strip markdown code fences and whitespace around the JSON.
    """
    raw_text = raw_text.strip()

    if "```" in raw_text:
        # Remove ```json ... ``` or ``` ... ```
        raw_text = re.sub(r"```json\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"```", "", raw_text)

    return raw_text.strip()


def _normalize_and_diversify_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    - Ensure search_type is one of the expected values (fallback to 'discovery').
    - Clean & dedupe queries, keep at most 3.
    - Log if diversity looks poor (but do not call the LLM again in this function).
    """
    valid_types = {"specific", "discovery", "pairing"}
    st = str(result.get("search_type", "")).lower()
    if st not in valid_types:
        logger.warning("Invalid or missing search_type from model; defaulting to 'discovery'", search_type=st)
        st = "discovery"

    raw_queries = result.get("queries", [])
    if not isinstance(raw_queries, list):
        logger.warning("Model returned non-list 'queries'; coercing to []", queries=raw_queries)
        raw_queries = []

    cleaned: List[str] = []
    seen_lower: set = set()

    for q in raw_queries:
        if not isinstance(q, str):
            continue
        q_clean = q.strip()
        if not q_clean:
            continue
        key = q_clean.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        cleaned.append(q_clean)

    if not cleaned:
        logger.warning("Model returned empty/invalid queries; setting fallback empty list.")
        cleaned = []

    # Truncate / enforce at most 3
    if len(cleaned) > 3:
        logger.debug("Truncating queries to 3 items", original_len=len(cleaned))
        cleaned = cleaned[:3]

    # Simple lexical overlap check just for logging (not strict enforcement)
    def overlap_score(a: str, b: str) -> int:
        toks_a = {t for t in re.findall(r"[a-zA-Z]+", a.lower()) if len(t) > 3}
        toks_b = {t for t in re.findall(r"[a-zA-Z]+", b.lower()) if len(t) > 3}
        return len(toks_a & toks_b)

    if len(cleaned) >= 2:
        max_overlap = 0
        for i in range(len(cleaned)):
            for j in range(i + 1, len(cleaned)):
                max_overlap = max(max_overlap, overlap_score(cleaned[i], cleaned[j]))
        if max_overlap > 2:
            logger.warning(
                "Queries appear lexically similar (possible low diversity from model).",
                queries=cleaned,
                max_overlap=max_overlap,
            )

    result["search_type"] = st
    result["queries"] = cleaned
    return result


async def t_classify_intent(
    query: str,
    user_gender: Optional[str] = None,
    forced_type: Optional[str] = None,
    product_context: Optional[str] = None,
    last_query_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify the fashion search intent. Uses FAST_MODEL (Responses API) with
    very explicit JSON instructions in the system prompt.
    """
    cache_key = _cache_key("intent", query.lower(), forced_type or "", user_gender or "", product_context or "", last_query_context or "")
    cached = await INTENT_CACHE.get(cache_key)
    if cached:
        logger.debug("💾 intent cache HIT")
        return cached

    t0 = time.perf_counter()
    logger.info("🧠 classify_intent", query=query[:80], forced_type=forced_type)

    # Map user_gender to simple terms for queries
    gender_term = ""
    if user_gender:
        gender_map = {
            "menswear": "for men",
            "men": "for men",
            "male": "for men",
            "womenswear": "for women",
            "women": "for women",
            "female": "for women",
            "neutral": "",
            "unisex": "",
        }
        gender_term = gender_map.get(user_gender.lower(), "")

    user_content_dict: Dict[str, Any] = {"query": query}
    if gender_term:
        user_content_dict["gender_context"] = gender_term
    if forced_type:
        user_content_dict["forced_type"] = forced_type
    if product_context:
        user_content_dict["product_context"] = product_context
    if last_query_context:
        user_content_dict["last_query_context"] = last_query_context

    user_content = json.dumps(user_content_dict)

    models_to_try = [Config.FAST_MODEL]
    if Config.MAIN_MODEL not in models_to_try:
        models_to_try.append(Config.MAIN_MODEL)

    for mdl in models_to_try:
        # logger.info("Calling model", model=mdl)

        try:
            # Use asyncio.to_thread to avoid blocking
            response = await asyncio.to_thread(
                client.responses.create,
                model=mdl,
                input=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                reasoning={"effort": "low"},
                max_output_tokens=1500,
            )

            logger.debug("Raw response object", response=str(response)[:500])

            raw_text = _extract_text_from_response(response)
            raw_text = _clean_json_text(raw_text)

            logger.debug("Extracted raw text", raw_text=raw_text[:500])

            if not raw_text:
                logger.warning("Empty response text from LLM", model=mdl)
                continue

            try:
                result_json = json.loads(raw_text)
            except json.JSONDecodeError as e:
                logger.error("JSON parsing failed", error=str(e), raw_text=raw_text)
                continue # Try next model if JSON is invalid

            normalized = _normalize_and_diversify_result(result_json)
            
            # Cache and return
            await INTENT_CACHE.set(cache_key, normalized)
            
            ms = int((time.perf_counter() - t0) * 1000)
            logger.success(
                "✅ classify_intent",
                duration_ms=ms,
                result=normalized
            )
            return normalized

        except Exception as e:
            logger.error("API call failed", model=mdl, error=str(e))
            # traceback.print_exc() # Use logger instead

    # Fallback if all models fail
    logger.error("❌ classify_intent failed after trying all models")
    return {
        "search_type": forced_type or "discovery",
        "queries": [query] # Minimal fallback
    }

# =============================================================================
# Catalog search helpers
# =============================================================================
def _product_identity(product: Dict[str, Any]) -> Optional[str]:
    return (
        product.get("product_id")
        or product.get("id")
        or product.get("url")
    )


def _extract_price_value(payload: Dict[str, Any]) -> Optional[float]:
    """
    Extract a single numeric price (INR) from Qdrant payload.

    Priority:
    1) attributes.price  (scraped numeric price)
    2) price.current     (from the 'price' object)
    """
    try:
        attrs = payload.get("attributes") or {}
        attr_price = attrs.get("price")
        if isinstance(attr_price, (int, float)):
            return float(attr_price)

        price_block = payload.get("price") or {}
        current = price_block.get("current")
        if isinstance(current, (int, float)):
            return float(current)
    except Exception:
        pass

    return None


def _interleave_results(results_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not results_lists:
        return []

    max_len = max(len(lst) for lst in results_lists)
    seen: Set[str] = set()
    combined: List[Dict[str, Any]] = []

    for i in range(max_len):
        for lst in results_lists:
            if i >= len(lst):
                continue
            product = lst[i]
            pid = _product_identity(product)
            if pid and pid in seen:
                continue
            if pid:
                seen.add(pid)
            combined.append(product)
    return combined


def _dedupe_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for product in products:
        pid = _product_identity(product)
        if not pid:
            pid = f"anon::{hashlib.md5(json.dumps(product, sort_keys=True).encode('utf-8')).hexdigest()}"
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append(product)
    return deduped


def _brand_key(product: Dict[str, Any]) -> str:
    brand = (product.get("brand") or "Unknown").strip().lower()
    return brand or "unknown"


def _brand_histogram(products: List[Dict[str, Any]], limit: int = 50) -> Dict[str, int]:
    hist = defaultdict(int)
    for p in products[:limit]:
        hist[_brand_key(p)] += 1
    return dict(hist)


def _brand_cap(products: List[Dict[str, Any]], max_per_brand: int) -> List[Dict[str, Any]]:
    if max_per_brand <= 0 or len(products) <= 1:
        return products
    seen = defaultdict(int)
    capped: List[Dict[str, Any]] = []
    for p in products:
        b = _brand_key(p)
        if seen[b] >= max_per_brand:
            continue
        seen[b] += 1
        capped.append(p)
    return capped


def _is_image_ok(url: Optional[str], timeout: float = 2.0) -> bool:
    if not url:
        return False
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if 200 <= resp.status_code < 400:
            return True
        # Some CDNs may block HEAD; fall back to GET
        resp = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
        return 200 <= resp.status_code < 400
    except Exception:
        return False


async def _pick_displayable_products(
    products: List[Dict[str, Any]], desired: int
) -> List[Dict[str, Any]]:
    """Return up to `desired` products preferring those with valid image URLs.
    Falls back to original ordering if not enough valid images are found."""
    if desired <= 0 or not products:
        return []

    if not Config.CHECK_IMAGES:
        return products[:desired]

    max_checks = min(len(products), desired + 4)
    subset = products[:max_checks]
    sem = asyncio.Semaphore(8)

    async def _probe(product: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        url = product.get("image_url")
        async with sem:
            ok = await asyncio.to_thread(_is_image_ok, url)
        return product, ok

    results = await asyncio.gather(*[_probe(p) for p in subset])

    chosen: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for product, ok in results:
        if ok and len(chosen) < desired:
            chosen.append(product)
        else:
            skipped.append(product)

    if len(chosen) < desired:
        remaining = desired - len(chosen)
        chosen.extend(skipped[:remaining])
        if remaining:
            logger.warning(
                "Insufficient valid images, filled from skipped",
                needed=desired,
                with_images=len(chosen) - remaining,
            )
    return chosen


def _rebalance_brand_pool(
    products: List[Dict[str, Any]],
    cap_per_brand: int,
    window: int,
) -> List[Dict[str, Any]]:
    """
    Soft-cap how many times a single brand can dominate the top of the pool.
    Keeps order otherwise, pushing overflow to the tail so rerankers still see variety.
    """
    if len(products) <= 1 or cap_per_brand <= 0 or window <= 0:
        return products

    capped: List[Dict[str, Any]] = []
    overflow: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)

    for product in products:
        brand = _brand_key(product)
        if len(capped) < window and counts[brand] >= cap_per_brand:
            overflow.append(product)
            continue

        if len(capped) < window:
            counts[brand] += 1
            capped.append(product)
        else:
            overflow.append(product)

    return capped + overflow


async def _numeric_rerank_products(user_query: str, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(products) <= 1:
        return products

    pool_limit = min(Config.NUMERIC_RERANK_POOL, len(products))
    pool = products[:pool_limit]
    remainder = products[pool_limit:]

    texts = [
        f"{p.get('title') or ''} {p.get('category_path') or p.get('category') or p.get('category_leaf') or ''}".strip()
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

        # Ensure each query keeps a minimum presence before we append the rest
        if Config.RERANK_MIN_PER_QUERY > 0:
            counts: Dict[Optional[str], int] = defaultdict(int)
            for p in ordered:
                counts[p.get("from_query")] += 1

            by_query: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
            for p in pool:
                by_query[p.get("from_query")].append(p)

            for q, items in by_query.items():
                while counts[q] < Config.RERANK_MIN_PER_QUERY and items:
                    candidate = items.pop(0)
                    pid = _product_identity(candidate)
                    if pid and pid in seen:
                        continue
                    if pid:
                        seen.add(pid)
                    ordered.append(candidate)
                    counts[q] += 1

        for product in pool:
            pid = _product_identity(product)
            if pid and pid in seen:
                continue
            if pid:
                seen.add(pid)
            ordered.append(product)

        ordered.extend(remainder)
        rerank_ms = int((time.perf_counter() - rerank_t0) * 1000)
        logger.perf(
            "rerank_vector",
            rerank_ms,
            pool=len(pool),
            total=len(products),
        )
        return ordered
    except Exception as e:
        logger.warning("Vector rerank failed", error=str(e))
        return products


async def _llm_rerank_products(
    user_query: str,
    products: List[Dict[str, Any]],
    min_results: int,
) -> List[Dict[str, Any]]:
    if len(products) <= 1:
        return products

    candidates = products[: Config.LLM_RERANK_INPUT_LIMIT]
    payload_products = []
    for product in candidates:
        payload_products.append(
            {
                "id": str(_product_identity(product)),
                "title": product.get("title"),
                "brand": product.get("attributes", {}).get("brand") or product.get("brand"),
                "category_leaf": product.get("category") or product.get("category_leaf"),
                "score": round(float(product.get("score", 0.0)), 4),
                "source": product.get("from_query") or product.get("source") or "catalog",
                "price": product.get("price_inr") or product.get("price") or "Unknown",
            }
        )

    system_prompt = """
You are a ranking model for a fashion shopping assistant.

CRITICAL DIVERSITY REQUIREMENT:
- You MUST prioritize brand diversity in the top results
- In the top 8 results, try to include at least 3-4 different brands
- Do NOT rank all products from the same brand consecutively

Ranking rules:
1. Match user request (relevance)
2. Ensure brand diversity (mix different brands)
3. Drop only clearly irrelevant items
4. Return at least `min_results` items if available
5. Preserve IDs exactly as provided

Output ONLY this JSON:
{"ordered_ids": ["id1", "id2", ...]}

Example of GOOD diversity (top 8):
- Rare Rabbit shirt
- Nonasties shirt  ← Different brand
- Rare Rabbit trouser
- Cultstore shirt  ← Different brand
- Rare Rabbit kurta
- Fabindia shirt  ← Different brand

Example of BAD diversity (top 8):
- Rare Rabbit shirt
- Rare Rabbit shirt  ← Same brand repeatedly!
- Rare Rabbit trouser
- Rare Rabbit kurta
- Rare Rabbit shirt
""".strip()

    llm_t0 = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            client.responses.create,
            model=Config.FAST_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_query": user_query,
                            "min_results": min_results,
                            "products": payload_products,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_output_tokens=400,
        )
        raw_text = (response.output_text or "").strip()
        data = {}
        if raw_text:
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw_text[start : end + 1])
        
        ordered_ids = data.get("ordered_ids") or data.get("products") or []
        id_map = {str(_product_identity(p)): p for p in candidates}
        ordered: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for pid in ordered_ids:
            pid = str(pid)
            product = id_map.get(pid)
            if not product:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            ordered.append(product)

        for pid, product in id_map.items():
            if pid in seen:
                continue
            ordered.append(product)

        # 🆕 Log brand distribution
        brand_counts = {}
        for p in ordered[:8]:
            brand = p.get("brand", "Unknown")
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        logger.info(f"📊 Top 8 brand distribution: {brand_counts}")

        llm_ms = int((time.perf_counter() - llm_t0) * 1000)
        logger.perf("rerank_llm", llm_ms, in_count=len(candidates), out_count=len(ordered))
        
        return ordered
    except Exception as e:
        logger.warning("LLM rerank failed", error=str(e))
        return products


async def _fetch_web_topup_products(
    user_query: str,
    existing_keys: Set[str],
    needed: int,
) -> List[Dict[str, Any]]:
    if needed <= 0:
        return []

    request_count = max(needed, Config.WEB_TOPUP_MIN_COUNT)
    web_t0 = time.perf_counter()
    try:
        web_results = await asyncio.to_thread(
            search_fashion_with_web,
            user_query,
            request_count,
        )
        web_ms = int((time.perf_counter() - web_t0) * 1000)
    except Exception as e:
        logger.error("web_search_topup_failed", error=str(e))
        return []

    mapped: List[Dict[str, Any]] = []
    for idx, item in enumerate(web_results):
        url = (item.get("sourceUrl") or "").strip()
        name = (item.get("name") or "").strip()
        if not url and not name:
            continue
        pid_seed = url or f"{name}-{idx}"
        pid = f"web::{hashlib.md5(pid_seed.encode('utf-8')).hexdigest()}"
        if pid in existing_keys or (url and url in existing_keys):
            continue

        product = {
            "id": pid,
            "product_id": pid,
            "title": name or "Web product",
            "brand": item.get("tone") or "Web",
            "category": item.get("description"),
            "image_url": item.get("imageUrl"),
            "url": url or item.get("sourceUrl"),
            "price": item.get("price"),
            "price_inr": item.get("price"),
            "score": 0.0,
            "from_query": "web_search",
            "source": "web",
        }
        mapped.append(product)
        if pid:
            existing_keys.add(pid)
        if url:
            existing_keys.add(url)

        if len(mapped) >= needed:
            break

    return mapped


# =============================================================================
# Catalog search
# =============================================================================
async def t_search_fashion_products(
    query: str,
    user_gender: Optional[str] = None,
    category_filter: Optional[str] = None,
    search_type: str = "auto",
    allow_web_search: bool = False,
    budget: Optional[Budget] = None,
    user_message: Optional[str] = None,
    min_price_inr: Optional[float] = None,
    max_price_inr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    - Filters by user_gender when provided (Menswear -> men, Womenswear -> women).
    - For discovery / pairing: uses the 3 short fashion-only queries returned by t_classify_intent.
    - Combines products from all queries, dedupes by product id, then reranks (if needed).
    - Web search top-up is disabled unless allow_web_search=True (only set after explicit user consent).
    """

    if budget and user_gender:
        budget.user_gender = user_gender
    
    # If gender wasn't provided but we have it from before, use it
    if budget and not user_gender and budget.user_gender:
        user_gender = budget.user_gender
        logger.info(f"♻️ Reusing saved gender: {user_gender}")

    web_allowed = bool(Config.USE_WEB_TOPUP or allow_web_search)

    raw_user_message = (user_message or "").strip()
    raw_user_lower = raw_user_message.lower()
    vibe_chips = {"casual", "work", "date", "travel", "festive"}

    # If the user only provided a high-level vibe chip, pause search and ask for details
    if raw_user_lower in vibe_chips:
        logger.info("🛑 Pure vibe chip received, deferring search", vibe=raw_user_lower)

        clarification_messages = {
            "travel": (
                "User only said the vibe 'Travel'. Ask where they are going, when (month or dates), how many days, and whether it's for work, sightseeing, or chill before searching.",
                [
                    "Which city or place are you travelling to?",
                    "When are you going (dates or month)?",
                    "Is it for work, sightseeing, or chill and how many days?",
                ],
            ),
            "work": (
                "User picked 'Work' without details. Ask about office vibe or dress code (formal vs smart casual), whether they need shirts, trousers, or a full look, and any colour constraints before searching.",
                [
                    "What is your office vibe or dress code (formal, smart casual, startup)?",
                    "Do you need a full look or just shirts/trousers?",
                    "Any colours or fits you avoid?",
                ],
            ),
            "date": (
                "User picked 'Date' without details. Ask what kind of date (coffee, dinner, outdoors), time of day, and how dressed up they want to be before searching.",
                [
                    "What kind of date is it (coffee, dinner, outdoors)?",
                    "When is it happening (time of day or day of week)?",
                    "Do you want something relaxed, smart casual, or dressier?",
                ],
            ),
            "festive": (
                "User picked 'Festive' without details. Ask which occasion (puja, sangeet, party), timing, and whether they want traditional or fusion before searching.",
                [
                    "What is the occasion (puja, sangeet, party)?",
                    "When is it happening?",
                    "Do you prefer traditional or a fusion vibe?",
                ],
            ),
            "casual": (
                "User picked 'Casual' without details. Ask where they're wearing it, preferred vibe (minimal, sporty, street), and any colour or fit preferences before searching.",
                [
                    "Where will you wear this (out with friends, errands, home)?",
                    "Do you prefer minimal, sporty, or street vibes?",
                    "Any colours or fits you avoid?",
                ],
            ),
        }

        clar_message, clar_questions = clarification_messages.get(
            raw_user_lower,
            (
                "User picked a vibe chip without context. Ask for the specific occasion, timing, formality, and items they need before searching.",
                [],
            ),
        )

        result_payload = {
            "products": [],
            "total_found": 0,
            "catalog_count": 0,
            "catalog_min_threshold": Config.CATALOG_MIN_RESULTS,
            "needs_web_search": False,
            "used_web_search": False,
            "web_allowed": web_allowed,
            "web_topup_count": 0,
            "search_type": "discovery",
            "queries_used": [],
            "original_query": raw_user_message or query,
            "best_score": 0.0,
            "total_candidates": 0,
            "needs_clarification": True,
            "clarification_type": "travel" if raw_user_lower == "travel" else raw_user_lower,
            "clarification_message": clar_message,
            "clarification_questions": clar_questions,
        }

        return result_payload


    # Enforce that the base query comes from the user's latest message if provided
    if user_message and user_message != query and raw_user_lower not in vibe_chips:
        logger.debug("🔄 Overriding model query with user_message", model_query=query, user_message=user_message)
        query = user_message

    # Add gender context to the base query when available
    gender_phrase = None
    if user_gender:
        gmap = {"menswear": "for men", "womenswear": "for women", "neutral": "unisex"}
        gender_phrase = gmap.get(user_gender.lower())

    ql = query.strip().lower()
    if gender_phrase and gender_phrase not in ql:
        query = f"{query.strip()} {gender_phrase}".strip()
        ql = query.lower()

    cache_key = _cache_key("search", query.lower(), search_type, user_gender, web_allowed, min_price_inr, max_price_inr)
    cached = await SEARCH_CACHE.get(cache_key)
    if cached:
        logger.success("💾 search cache HIT")
        return cached

    t0 = time.perf_counter()
    logger.info("🔍 search_fashion_products", query=query, search_type=search_type)

    try:
        await Services.ensure_loaded()

        # Always classify intent and allow a forced search_type hint when provided
        ql = query.lower()

        # 🆕 Extract product context from budget if available
        product_context = None
        if budget and budget.last_products:
            # Create a short summary of the last 3 products
            titles = [p.get("title", "item") for p in budget.last_products[:3]]
            product_context = f"User just saw: {', '.join(titles)}"
            logger.info(f"📋 Using product context: {product_context}")

        # 🆕 Extract last query context from budget (use original query + queries_used)
        last_query_context = None
        if budget and budget.last_search_result:
            original_q = budget.last_search_result.get("original_query")
            prev_queries = budget.last_search_result.get("queries_used") or []
            if original_q and prev_queries:
                # Format: "original intent [queries: q1, q2, q3]"
                queries_str = ", ".join(prev_queries[:3])
                last_query_context = f"{original_q} [queries: {queries_str}]"
                logger.info(f"📋 Using last query context: {last_query_context}")

        normalized_type = (search_type or "auto").lower()
        forced_type = None
        if normalized_type in ("discovery", "pairing", "specific"):
            forced_type = normalized_type
        elif normalized_type not in ("auto",):
            logger.warning("Invalid search_type from model; defaulting to auto classification", search_type=normalized_type)
            normalized_type = "auto"

        intent = await t_classify_intent(
            query, 
            user_gender, 
            forced_type=forced_type, 
            product_context=product_context,
            last_query_context=last_query_context
        )

        search_type = forced_type or intent.get("search_type") or "specific"
        queries = intent.get("queries") or [query]
        if not queries:
            queries = [query]

        # Extra specialisation for "trending" like queries
        if "trend" in ql or "trending" in ql:
            search_type = "discovery"
            # Rely on classifier for queries, do not hardcode "India" ones here.

        logger.debug("Search queries", type=search_type, queries=queries)

        if user_gender:
            logger.info(f"🎯 Final gender for search: {user_gender}")
        else:
            logger.warning("⚠️ No gender provided or saved - searching without gender filter")

        # Embedding
        embed_t0 = time.perf_counter()
        vectors = await Services.embed(queries)
        embed_ms = int((time.perf_counter() - embed_t0) * 1000)
        logger.perf("embed", embed_ms, num_queries=len(queries))

        # Qdrant search
        from qdrant_client.http import models as rest

        # Map user_gender to catalog gender values
        gender_filter_value = None
        if user_gender:
            gender_map = {
                "menswear": "men",
                "men": "men",
                "male": "men",
                "womenswear": "women", 
                "women": "women",
                "female": "women",
                "neutral": None,
                "unisex": "unisex",
            }
            gender_filter_value = gender_map.get(user_gender.lower())

        query_filter = None
        if gender_filter_value:
            # Include unisex for both men and women
            if gender_filter_value == "men":
                gender_condition = rest.FieldCondition(
                    key="attributes.gender",
                    match=rest.MatchAny(any=["men", "unisex"])
                )
            elif gender_filter_value == "women":
                gender_condition = rest.FieldCondition(
                    key="attributes.gender",
                    match=rest.MatchAny(any=["women", "unisex"])
                )
            else:  
                # neutral or unhandled values → only match explicit 'unisex'
                gender_condition = rest.FieldCondition(
                    key="attributes.gender",
                    match=rest.MatchValue(value="unisex")
                )

            query_filter = rest.Filter(must=[gender_condition])

        async def _search_one(q: str, vec: List[float]):
            def _do():
                return Services.qdr.query_points(
                    collection_name=Config.CATALOG_COLLECTION,
                    query=vec,
                    limit=(
                        Config.PRODUCTS_PER_QUERY
                        if len(queries) > 1
                        else Config.SIMPLE_SEARCH_LIMIT
                    ),
                    with_payload=True,
                    search_params=rest.SearchParams(hnsw_ef=Config.HNSW_EF),
                    query_filter=query_filter,
                )

            return await asyncio.to_thread(_do)

        search_t0 = time.perf_counter()
        results = await asyncio.gather(
            *[_search_one(q, v) for q, v in zip(queries, vectors)],
            return_exceptions=True,
        )
        search_ms = int((time.perf_counter() - search_t0) * 1000)
        logger.perf("qdrant_search", search_ms, num_queries=len(queries))

        # Aggregate products + candidate preparation
        results_lists: List[List[Dict[str, Any]]] = []
        best_score = 0.0
        per_query = (
            min(Config.RERANK_PER_QUERY, Config.PRODUCTS_PER_QUERY)
            if len(queries) > 1
            else Config.SIMPLE_SEARCH_LIMIT
        )

        for i, (q, result) in enumerate(zip(queries, results)):
            if isinstance(result, Exception):
                logger.warning(f"Query {i} failed", error=str(result))
                continue

            q_products: List[Dict[str, Any]] = []
            for point in (result.points or [])[:per_query]:
                payload = point.payload or {}
                pid = (
                    payload.get('product_id')
                    or payload.get('id')
                    or getattr(point, 'id', None)
                )
                if not pid:
                    continue
                pid = str(pid)

                score = float(point.score)
                best_score = max(best_score, score)

                price_val = _extract_price_value(payload)

                card = {
                    'id': pid,
                    'product_id': pid,
                    'title': payload.get('title'),
                    'brand': payload.get('brand') or (payload.get("attributes") or {}).get("brand"),
                    'category': payload.get('category_leaf'),
                    'category_leaf': payload.get('category_leaf'),
                    'category_path': payload.get('category_path'),
                    'image_url': payload.get('primary_image') or payload.get('image_url'),
                    'url': payload.get('url'),
                    # canonical numeric price in INR
                    'price': price_val,
                    'price_inr': price_val,
                    'score': score,
                    'from_query': q,
                    'source_tags': payload.get('source_tags') or [],
                }
                q_products.append(card)
            results_lists.append(q_products)

            # Debug sample of raw Qdrant hits per query
            try:
                sample = [
                    {
                        "brand": p.get("brand"),
                        "title": p.get("title"),
                        "category_path": p.get("category_path"),
                        "score": p.get("score"),
                    }
                    for p in q_products[:5]
                ]
                logger.debug("qdrant_raw_sample", query=q, count=len(q_products), top5=sample)
            except Exception:
                pass

        interleaved = _interleave_results(results_lists)
        candidates = _dedupe_products(interleaved)
        if len(queries) == 1:
            candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        total_candidates = len(candidates)

        if candidates and Config.DEBUG:
            logger.debug(
                "Brand mix (raw candidates)",
                top_counts=_brand_histogram(candidates, 50),
            )

        logger.debug(
            'Aggregated products',
            count=total_candidates,
            best_score=best_score,
        )

        # Apply price filter in Python if requested
        if min_price_inr is not None or max_price_inr is not None:
            before = len(candidates)
            filtered: List[Dict[str, Any]] = []
            for p in candidates:
                price_val = p.get("price_inr") or p.get("price")
                if price_val is None:
                    continue
                if min_price_inr is not None and price_val < min_price_inr:
                    continue
                if max_price_inr is not None and price_val > max_price_inr:
                    continue
                filtered.append(p)

            candidates = filtered
            total_candidates = len(candidates)
            logger.info(
                "Price filter applied",
                before=before,
                after=total_candidates,
                min_price_inr=min_price_inr,
                max_price_inr=max_price_inr,
            )

        base_products = candidates
        if base_products:
            base_products = await _numeric_rerank_products(query, base_products)
            base_products = _dedupe_products(base_products)
            # Debug sample after numeric rerank
            try:
                sample = [
                    {
                        "brand": p.get("brand"),
                        "title": p.get("title"),
                        "source": p.get("from_query"),
                        "category_path": p.get("category_path"),
                        "score": p.get("score"),
                    }
                    for p in base_products[:5]
                ]
                logger.debug("rerank_sample", count=len(base_products), top5=sample)
            except Exception:
                pass

        final_products = base_products
        if final_products and Config.USE_LLM_RERANK:
            llm_ranked = await _llm_rerank_products(
                query, final_products, Config.MIN_PRODUCTS_TARGET
            )
            if llm_ranked:
                final_products = llm_ranked

        catalog_count = len(final_products)
        needs_web_topup = catalog_count < Config.CATALOG_MIN_RESULTS
        web_products: List[Dict[str, Any]] = []
        used_web_search = False

        if needs_web_topup and web_allowed:
            needed = max(
                Config.MIN_PRODUCTS_TARGET - catalog_count,
                Config.CATALOG_MIN_RESULTS - catalog_count,
                0,
            )
            existing_keys = {
                key
                for key in (_product_identity(p) for p in final_products)
                if key
            }
            web_products = await _fetch_web_topup_products(
                query, existing_keys, needed
            )
            if web_products:
                used_web_search = True
                logger.info(
                    'Using web search fallback',
                    added=len(web_products),
                    needed=needed,
                )
                base_with_web = _dedupe_products((base_products or []) + web_products)
                reranked = await _llm_rerank_products(
                    query, base_with_web, Config.MIN_PRODUCTS_TARGET
                )
                final_products = reranked or base_with_web
            else:
                logger.warning(
                    'Web search fallback returned nothing',
                    requested=needed,
                )
        elif needs_web_topup and not web_allowed:
            logger.info(
                'Skipping web search fallback (not approved)',
                needed=Config.CATALOG_MIN_RESULTS - catalog_count,
                available=catalog_count,
            )

        ranked_products: List[Dict[str, Any]] = []
        for idx, product in enumerate(final_products, 1):
            ranked = dict(product)
            ranked["rank"] = idx
            ranked_products.append(ranked)
        final_products = ranked_products

        storage_limit = max(Config.FINAL_RERANK_TOP_K, Config.DISPLAY_PRODUCTS_COUNT)
        stored_products = final_products[:storage_limit]
        display_products = await _pick_displayable_products(
            stored_products, Config.DISPLAY_PRODUCTS_COUNT
        )
        logger.debug(
            "Display products order",
            titles=[p.get("title") for p in display_products],
            sources=[p.get("from_query") for p in display_products],
        )

        result_payload = {
            "products": display_products,
            "total_found": len(final_products),
            "catalog_count": catalog_count,
            "catalog_min_threshold": Config.CATALOG_MIN_RESULTS,
            "needs_web_search": needs_web_topup,
            "used_web_search": used_web_search,
            "web_allowed": web_allowed,
            "web_topup_count": len(web_products),
            "search_type": search_type,
            "queries_used": queries,
            "original_query": query,
            "best_score": best_score,
            "total_candidates": total_candidates,
            "min_price_inr": min_price_inr,
            "max_price_inr": max_price_inr,
            "_all_products": stored_products,
        }

        await SEARCH_CACHE.set(cache_key, result_payload)

        total_ms = int((time.perf_counter() - t0) * 1000)
        logger.success(
            "✅ search completed",
            duration_ms=total_ms,
            displayed=len(display_products),
            stored=len(stored_products),
            score=best_score,
        )
        return result_payload

    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.error("❌ search failed", duration_ms=ms, error=str(e))
        logger.error(traceback.format_exc())
        return {
            "products": [],
            "total_found": 0,
            "catalog_count": 0,
            "catalog_min_threshold": Config.CATALOG_MIN_RESULTS,
            "needs_web_search": True,
            "used_web_search": False,
            "web_allowed": web_allowed,
            "web_topup_count": 0,
            "search_type": search_type,
            "queries_used": [query],
            "best_score": 0.0,
            "error": str(e),
        }


# =============================================================================
# Post-search suggestions (chips) – Under ₹5000, colours, pairing
# =============================================================================
async def _generate_search_suggestions(
    query: str, context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal helper to generate 3-4 *refinement suggestions* to show AFTER products.
    """
    ql = query.lower()
    suggestions: List[Dict[str, Any]] = []

    # 1) Budget chip (common default)
    suggestions.append(
        {
            "label": "Under ₹5000",
            "kind": "price_filter",
            "payload": {"max_price_inr": 5000},
        }
    )

    # 2) Colour refinement
    suggestions.append(
        {
            "label": "Show different colours",
            "kind": "color_refine",
            "payload": {},
        }
    )

    # 3) Pairing chip – only if the base looks like a single item
    if any(
        kw in ql
        for kw in [
            "shirt",
            "t shirt",
            "tee",
            "trouser",
            "pants",
            "jeans",
            "kurta",
            "co ord",
            "hoodie",
            "sweatshirt",
            "jacket",
            "dress",
            "saree",
            "sari",
        ]
    ):
        suggestions.append(
            {
                "label": "Pair this with something",
                "kind": "pairing",
                "payload": {},
            }
        )

    # Fallback if nothing matched (should be rare)
    if not suggestions:
        suggestions = [
            {
                "label": "Under ₹5000",
                "kind": "price_filter",
                "payload": {"max_price_inr": 5000},
            },
            {
                "label": "Different colours",
                "kind": "color_refine",
                "payload": {},
            },
            {
                "label": "More similar options",
                "kind": "more_like_this",
                "payload": {},
            },
        ]

    return {
        "base_query": query,
        "context": context,
        "suggestions": suggestions,
    }


# =============================================================================
# Tone-polishing helper
# =============================================================================
async def t_tone_reply(
    prompt: str, products: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Tone-polishing helper. Uses FAST_MODEL via Responses API.
    """
    t0 = time.perf_counter()
    logger.info("✨ tone_reply")

    try:
        response = await asyncio.to_thread(
            client.responses.create,
            model=Config.FAST_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite into a short, warm MuseBot reply for a fashion chat.\n"
                        "- Max 2 short sentences before any bullets.\n"
                        "- Use 2 or 3 emojis total, make it feel like a stylish Indian friend.\n"
                        "- For product bullets, keep them aesthetic, for example:\n"
                        "  \"A comfy blue cotton regular fit shirt from Rare Rabbit, clean office match with navy trousers ✨\"\n"
                        "- Do NOT mention price, size, or color lists unless the user explicitly asked about budget, size, or color.\n"
                        "- Do NOT use long punctuation dashes like — or – between clauses, use commas or emojis instead.\n"
                        "- Never use the rainbow emoji.\n"
                        "- Keep all factual details and product identities unchanged."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"prompt": prompt, "products": (products or [])[:3]}
                    ),
                },
            ],
            max_output_tokens=250,
        )

        text = (response.output_text or "").strip()
        ms = int((time.perf_counter() - t0) * 1000)
        logger.success("✅ tone_reply", duration_ms=ms)
        return {"text": text}
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.error("❌ tone_reply failed", duration_ms=ms, error=str(e))
        return {"text": prompt}


# =============================================================================
# Show Options (Chips)
# =============================================================================
async def _generate_quick_options_local(
    context: str,
    hint: Optional[str],
    logger: Logger,
) -> List[str]:
    """
    Generate short clickable options using the OpenAI FAST model directly.
    """
    if not context:
        return []

    system_prompt = (
        "You are a UI helper for a fashion chatbot.\n"
        "Given the bot's last message, return 3-5 short clickable options as JSON array of strings.\n"
        "Rules:\n"
        "1) ONLY output a JSON list of strings like [\"Menswear\", \"Womenswear\", \"Neutral\"].\n"
        "2) Keep options very short (1-4 words).\n"
        "3) If no clear options, return [].\n"
        "4) If hint='product_refinement', generate ACTIONS like 'Under ₹2000', 'Show matching footwear', 'Different colors'. Do NOT just repeat attributes like 'Cotton' or 'Blue'.\n"
        "5) If hint='question', generate the specific answer choices implied by the question.\n"
        "6) CRITICAL: If the user just answered a question (e.g. 'Menswear'), DO NOT generate options asking the same thing again. If the conversation is flowing naturally without a clear need for chips, return [].\n"
    )

    user_lines = [f"Bot message: {context}"]
    if hint:
        user_lines.append(f"Hint: {hint}")
    user_content = "\n".join(user_lines)

    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            # remove starting fence
            t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rstrip("` \n\r\t")
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0].rstrip()
        return t.strip()

    try:
        resp = await asyncio.to_thread(
            client.responses.create,
            model=Config.FAST_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            reasoning = {"effort":"low"},
            max_output_tokens=500,
        )
        raw = (resp.output_text or "").strip()
        if not raw:
            logger.warning("Quick options empty response")
            return []
        cleaned = _strip_fences(raw)
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if isinstance(x, (str, int, float))][:5]
        if isinstance(parsed, dict):
            for key in ("options", "choices", "chips", "suggestions"):
                if key in parsed and isinstance(parsed[key], list):
                    return [
                        str(x) for x in parsed[key] if isinstance(x, (str, int, float))
                    ][:5]
        logger.warning("Quick options JSON not a list/dict", raw=raw[:200])
        return []
    except json.JSONDecodeError as e:
        logger.warning("Quick options JSON decode failed", error=str(e), raw=raw[:200])
        return []
    except Exception as e:
        logger.error("Quick options generation failed", error=str(e))
        return []

async def t_show_options(
    context: str = "",
    hint: str = None,
    budget: Optional["Budget"] = None
) -> Dict[str, Any]:
    """
    Generate and display clickable UI chips/buttons to the user.
    
    Args:
        context: Current conversation context (what the bot just said or showed)
        hint: Optional hint for what kind of options to generate:
            - "product_refinement": After showing products (price, colors, pairing)
            - "question": Answering a question (Menswear/Womenswear/Neutral)
            - None: Auto-detect from context
        budget: Budget tracker to store options
    
    Returns:
        Dict with "options" key containing list of suggestion strings
    """
    logger.info("🔘 show_options", context_len=len(context), hint=hint)
    
    try:
        options = await _generate_quick_options_local(
            context=context,
            hint=hint,
            logger=logger,
        )
        
        if budget:
            budget.last_options = options
        
        logger.info("✅ Options generated", count=len(options), options=options)
        return {"options": options}
    
    except Exception as e:
        logger.error("show_options failed", error=str(e))
        return {"options": []}


async def t_search_catalog_metadata(
    query: Optional[str] = None,
    field: str = "brand",
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search for available metadata values (like brands) in the catalog.
    Useful for answering "What brands do you have?" or checking if a specific brand exists.
    """
    logger.info("🔍 search_catalog_metadata", query=query, field=field)
    try:
        await Services.ensure_loaded()
        
        from qdrant_client.http import models as rest
        
        unique_values = set()
        
        if query:
            # Search for specific value using a match filter
            should_filter = [
                rest.FieldCondition(
                    key=f"attributes.{field}",
                    match=rest.MatchText(text=query)
                ),
                rest.FieldCondition(
                    key=field,
                    match=rest.MatchText(text=query)
                )
            ]
            
            filter_query = rest.Filter(should=should_filter)
            
            # Check existence
            results = await asyncio.to_thread(
                Services.qdr.scroll,
                collection_name=Config.CATALOG_COLLECTION,
                scroll_filter=filter_query,
                limit=limit,
                with_payload=True
            )
            points = results[0]
        else:
            # No query, just get a sample to list available options
            results = await asyncio.to_thread(
                Services.qdr.scroll,
                collection_name=Config.CATALOG_COLLECTION,
                limit=50, # Fetch more to find unique values
                with_payload=True
            )
            points = results[0]
            
        # Extract values
        found_items = []
        for point in points:
            payload = point.payload or {}
            val = payload.get("attributes", {}).get(field) or payload.get(field)
            if val:
                norm = str(val).strip()
                # Simple dedupe (case-insensitive check)
                if norm and norm.lower() not in [x.lower() for x in unique_values]:
                    unique_values.add(norm)
                    found_items.append(norm)
                    
        return {
            "field": field,
            "query": query,
            "found": sorted(list(unique_values))[:limit],
            "count": len(unique_values)
        }

    except Exception as e:
        logger.error("❌ search_catalog_metadata failed", error=str(e))
        return {"error": str(e), "found": []}


TOOLS_MAP = {
    "search_fashion_products": t_search_fashion_products,
    "tone_reply": t_tone_reply,
    "get_weather": t_get_weather,
    # "generate_search_suggestions": t_generate_search_suggestions,
    "show_options": t_show_options,
    "search_catalog_metadata": t_search_catalog_metadata,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
    "name": "search_fashion_products",
    "description": (
        "Smart fashion search backed ONLY by the internal catalog (Qdrant). "
        "Use this for any request that involves buying, showing, or recommending items. "
        "Provide ONE tight fashion query (material + category + vibe) and the tool will auto-expand discovery/pairing cases into short searches, dedupe, rerank (vector + Qwen reranker), and return catalog results only. "
        "If catalog results are fewer than 4, ask the user for consent before setting allow_web_search=true and retrying. "
        "Pass the user's latest message verbatim in user_message; do not rewrite it. "
        "Do NOT call this repeatedly in the same reply unless the user clearly asks for a brand-new category or filter. "
        "BAD Query: 'masculine wardrobe staples', 'party wear', 'office outfit'. "
        "GOOD Query: 'White Linen Shirt', 'Navy Blue Chinos', 'Black Leather Boots', 'Beige Cotton Trousers'. "
            "The search engine only understands: Material, Color, Fit, Category, Pattern."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "user_message": {
                    "type": "string",
                    "description": "The user's latest message verbatim. Use this as the base query; do not rewrite.",
                },
                "user_gender": {"type": "string"},
                "search_type": {
                    "type": "string",
                    "enum": ["auto", "specific", "discovery", "pairing"],
                },
                "allow_web_search": {
                    "type": "boolean",
                    "description": "Set to true ONLY after the user explicitly approves web search when catalog results are sparse (<4).",
                },
                "min_price_inr": {
                    "type": "number",
                    "description": "Minimum price in INR (inclusive). Optional.",
                },
                "max_price_inr": {
                    "type": "number",
                    "description": "Maximum price in INR (inclusive). Optional.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "tone_reply",
        "description": "Polish response tone (optional).",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "products": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather snapshot for a city using Open-Meteo. Use ONLY when the user asks about weather or packing for a specific city and near term.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["city"],
        },
    },
    {
        "type": "function",
        "name": "show_options",
        "description": (
            "Generate and display 3-5 clickable suggestion chips to help the user respond. "
            "Call this when:\n"
            "1. After showing products: Use hint='product_refinement' to suggest options like 'Under 3k', 'Different colors', 'What goes well', 'Similar items'\n"
            "2. When asking a question: Use hint='question' to provide answer choices like 'Menswear', 'Womenswear', 'Neutral'\n"
            "3. Auto-detect: Leave hint=None to let the model decide based on context\n"
            "IMPORTANT: Call this frequently to improve UX - after product displays, when asking questions, or offering refinements."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "What you just said or showed to the user - this helps generate relevant options"
                },
                "hint": {
                    "type": "string",
                    "enum": ["product_refinement", "question"],
                    "description": "Optional hint: 'product_refinement' after products, 'question' when asking, or omit for auto-detect"
                },
            },
            "required": ["context"],
        },
    },
    {
        "type": "function",
        "name": "search_catalog_metadata",
        "description": "Search for available brands or metadata in the catalog. Use this when the user asks 'What brands do you have?' or 'Do you have X?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific value to check for (e.g. 'Fabindia'). Leave empty to list random available ones."
                },
                "field": {
                    "type": "string",
                    "default": "brand",
                    "description": "Field to search in (default: 'brand')."
                },
                "limit": {
                    "type": "integer",
                    "default": 10
                }
            },
        },
    },
]

SYSTEM_PROMPT = """
You are MuseBot 🎨, the chat stylist for MUSE, an India first fashion discovery platform.
MUSE helps people discover Indian and global brands online, you are the friendly stylist on top.

ASSUMPTIONS:
- Assume the user is in India by default unless they clearly say otherwise.
- You MUST determine the user's gender preference (Menswear, Womenswear, or Neutral) BEFORE searching.
- EXCEPTION: Do NOT ask this in your very first "Hello" greeting. In the first greeting, just introduce yourself and ask for their name.
- When the user asks for products (e.g. "I want shirts"), IF you don't know the gender yet, ask for it THEN.
- Do NOT search until you have this preference.

CRITICAL GENDER RULE:
- Once you know the user's gender preference (Menswear/Womenswear/Neutral), you MUST pass it as `user_gender` to EVERY call to search_fashion_products.
- Example: If user chose "Menswear", then ALL searches must include user_gender="Menswear"
- This applies to work wear, pairing, traditional wear, casual wear, EVERYTHING.
- Only skip user_gender if the user explicitly wants opposite gender items (e.g., "show me women's dresses for my sister").

OVERALL UX VIBE:
- Replies must be short and product forward.
  - Maximum 1 or 2 short sentences before the product bullets.
  - Use a bullet list for products, then ONE short follow up question.
  - Use 2 or 3 emojis in most answers (at least 1). Playful but not cringe.
- The ACK might have already greeted; in the main reply do NOT greet again unless it's the very first message. Skip "Hi/Hey/Hello" and jump straight to the fit or next step.
- Less lecture, more looks.
- Never use long punctuation dashes like — or –. Use commas, full stops, or emojis instead.
- Never use the rainbow emoji.

1) Brand and “What is MUSE”:
- If the user asks “What is Muse or MUSE or what do you do”:
  - In 1 or 2 sentences:
    - Say that MUSE is an India based fashion discovery platform curating Indian D2C fashion brands.
    - Say that MuseBot is the chat stylist that plans outfits and surfaces real catalog products.
  - Example:
    - “I am MuseBot, the chat stylist for MUSE, an India first fashion discovery platform. I help you plan outfits and find real pieces you can actually buy 🙂✨”

2) Domain lock:
- You only talk about fashion, style, grooming, clothing, footwear, and accessories.
- If the user asks non fashion things (coding, elephants, matcha, politics, etc):
  - Give ONE playful line, for example:
    - “I am only licensed to style outfits, not politics, but we can still make your look iconic”
  - Then nudge back with a simple fashion question:
    - “What are you dressing for next, work, date, travel, or festive”

3) India first styling rules:
- For “traditional day”, “ethnic day”, “festive”, “Diwali”, “puja”, “wedding”, “sangeet”, “mehendi”:
  - Prefer:
    - Cotton or linen kurtas, churidar or straight pants, Nehru or Modi jackets, bandhgalas, juttis or loafers.
    - You can still add 1 or 2 smart Western looks like chinos with a shirt, but ethnic should be visible.

- For normal “office”, “meeting”, “interview”:
  - Talk in terms of modern Indian office wear: shirts, chinos, minimal sneakers or loafers, sometimes blazers or suits.
- For “travel to <city> in <month> or next week”:
  - Set `search_type="discovery"` (or leave it blank/auto) so the tool automatically generates 3 concise queries and reranks them.
  - Use a neutral but relevant query phrase like “smart casual shirts India” or “knit polos for office India”, then pick colours in the answer.
- When building the search query string:
  - If “catalog_count” is 1–3: show those pieces, say the catalog is thin, and ask “Want me to search the web for more?”.
  - If “catalog_count” is 0: say “No good matches in the catalog”, then ask if they want web search or a refined query.
- When you ask, offer quick-reply chips in text: ["Yes, search the web", "No, stay in catalog"].
- Only if the user explicitly says yes (or clicks yes) should you call search_fashion_products again with allow_web_search=true to top up results.
- If the user declines, stay in catalog mode and offer a refinement question instead of using the web.

4) Weather & Travel:
- If the user mentions travelling to a specific city or asks about packing for a location:
  - Call `get_weather(city=...)` to get real-time context.
  - Use the weather info (temp, rain) to justify your outfit recommendations.
  - Example: "Since it's 12°C and rainy in London, I've picked waterproof layers..."

5) Brand & Metadata Questions:
- If the user asks "What brands do you have?" or "Do you have Fabindia?":
  - Call `search_catalog_metadata(query='Fabindia', field='brand')` to check.
  - If they ask generally, call `search_catalog_metadata(field='brand')` to get a sample list.
  - Do NOT hallucinate brands. Only list what the tool returns.

6) Trends:
- You will receive a separate system message with cached fashion trend notes for western and ethnic wear.
- Use this only as light seasoning, not strict truth.
- For any question like “what is trending” or “what are the latest trends”:
  - First, read the cached trend system message.
  - Give exactly ONE short sentence summarising what is trending using that context.
  - Then, if helpful, call “search_fashion_products” with short discovery queries (for example oversized shirts, Korean trousers, co ord sets).
- For example, if the user searches items that are trending, like “Korean trousers” or “oversized shirts”:
  - You can say:
    - “Excellent choice, Korean style trousers are very much in trend right now, relaxed straight fits with clean lines look super sharp 👌”
  - Then show catalog products and pairing ideas.
- Do not overdo trend talk, 1 simple line is enough.

7) Asking for user info:
- Name:
  - Within your first one or two helpful replies in a new conversation, after giving some value, you must ask once:
    - “By the way, what should I call you, you can skip if you like”
  - Ask at most once per conversation.
- Presentation or gender vibe:
  - You MUST ask for gender preference at the start if unknown.
  - Once known, stick to it for the session unless changed.
  - If the UI has buttons for gender, you can say:
    - “You can tap an option or just tell me”
- After the user picks Menswear/Womenswear/Neutral, your next clarification should be the occasion / vibe (for example Work, Date, Travel, Festive, Casual). Prefer those options over specific product categories so discovery stays broad.
- If the user only gives a vibe chip (Work, Date, Travel, Festive, Casual) with no details, do NOT call search_fashion_products yet. Ask 2-3 pointed follow-ups first (Travel: city + timing + trip type, Work: office vibe or dress code and whether they need shirts/trousers/full look, Date: venue/time/dressiness, Festive or Casual: exact occasion, timing, how dressed up they want). Search only after they answer.
- If search_fashion_products returns with "needs_clarification": true, do NOT assume there are products. Read the "clarification_message" and "clarification_questions", ask 1-3 of them in your own words, and do not call search_fashion_products again until the user answers.
- The question about name or gender should be separate from other questions and not spammy.
- End each message with just one simple next step question.

8) Smart Options (show_options tool):
- Call "show_options" to provide clickable chips ONLY when there are clear, actionable choices.
- When to call:
  - AFTER calling search_fashion_products: Use hint="product_refinement"
    - Example context: "I just showed you blue shirts"
    - Generates: "Under 3k", "Different colors", "What goes well", "Similar items"
  - WHEN asking for styling preferences with clear options: Use hint="question"
    - Example context: "Do you prefer Menswear, Womenswear, or Neutral?"
    - Generates: "Menswear", "Womenswear", "Neutral"
  - For other cases: Leave hint=None for auto-detection
- When NOT to call:
  - Name questions ("What should I call you, you can skip if you like") - NO OPTIONS NEEDED
  - Open-ended questions without clear choices
  - Simple greetings or acknowledgments
  - Follow-up clarifications
- The context field should contain what you just said/showed
- Call show_options at the END of your response, after your main message
- Do NOT call show_options repeatedly or excessively - once per response maximum
- IMPORTANT: Only call when the user would benefit from clickable suggestions, not every message

8) Pairing and diversity:
- For “what should I wear with X” or “what goes with navy trousers”:
  - Offer variety:
    - Different colour families like light blue, white, pastel pink, subtle prints, stripes.
    - Different product types like shirts, polos, knitwear, overshirts, jackets if the catalog allows.
    - Aim for at least 2 different brands if available.
  - Do not give many bullets of almost identical blue or white shirts.
  - Mention why each piece works in a short phrase, like contrast, balance, or vibe.

9) Style of writing:
- Tone: stylish Indian friend or wingman, not a corporate assistant.
- Structure:
  1) One or two short sentences of context with 1 to 3 emojis.
  2) Bullet list of products in the stylist voice (max 8). Each bullet must be super tight: “Brand + key adjective + garment + short vibe clause” – aim for 10 words or fewer, e.g. “Rare Rabbit linen shirt, soft pastel lift for Monday meetings 🙂”. Do not repeat “men’s” everywhere, skip long descriptors, and keep it punchy.
  3) One short question to move forward (size, budget, vibe, or occasion).
- Example of good bullets without long dashes:
  - “A soft off white mandarin collar kurta from Fabindia, perfect for office traditional day without feeling overdressed 🎉”
  - “A navy knit polo from Rare Rabbit, easy upgrade from a tee that still feels relaxed for Fridays 🙂”

10) Use of the tone_reply tool:
- If your natural answer is correct but wordy or stiff:
  - You may call “tone_reply” with your draft and a few products.
  - Then send only the polished reply.

11) Post-search suggestions:
- After you call “search_fashion_products” and show some products, you may suggest refinement options in text.
- Use its output to display 2 or 3 refinement options like:
  - “Under ₹5000”
  - “Show different colours”
  - “Pair this with something” (only if relevant)
- Do NOT automatically apply these filters.
  - Instead, show the suggestions as short chips in text and ask the user which one they want.
  - When the user picks one, THEN call “search_fashion_products” again with a refined query or filters based on their choice.

12) Interactive Options:
- Whenever you ask a question with clear choices (e.g. "Masculine or Feminine?", "Work or Party?", "Under 2k or 5k?"):
  - Just list them in the text or ask clearly.
  - Example: "Do you prefer Masculine, Feminine, or Neutral?"

IMPORTANT:
- Never respond with an empty message.
- Each reply must have at least one or two sentences in total, or one sentence plus bullets plus a question.
- Do not use the long dash character — anywhere in your reply. Prefer commas, full stops, and emojis instead.
- Never use the rainbow emoji.

""".strip()


# =============================================================================
# Function-calling helpers (Responses API)
# =============================================================================
async def _run_single_tool_call(tool_call, budget: Budget) -> Dict[str, Any]:
    """
    Execute a single function tool_call from the Responses API and return
    a function_call_output block suitable for the next Responses request.
    """
    name = getattr(tool_call, "name", None)
    call_id = getattr(tool_call, "call_id", None)

    if not budget.can_call(name or ""):
        logger.warning("❌ Skipping tool, budget exhausted", tool=name)
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps({"error": "Budget exhausted"}),
        }

    raw_args = getattr(tool_call, "arguments", {}) or {}
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}

    fn = TOOLS_MAP.get(name)
    if not fn:
        logger.error("❌ Unknown tool", name=name)
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps({"error": f"Unknown tool: {name}"}),
        }

    t0 = time.perf_counter()
    try:
        logger.info("🔧 Executing tool", tool=name, args=args)
        
        # 🆕 Inject budget for search_fashion_products
        if name == "search_fashion_products":
            args["budget"] = budget
        
        result = await fn(**args)

        # Capture last product search for frontend
        if name == "search_fashion_products" and isinstance(result, dict):
            safe_result = dict(result)
            all_products = safe_result.pop("_all_products", None)
            display_products = safe_result.get("products") or []
            budget.last_search_result = safe_result
            budget.record_products(display_products, all_products)
            result = safe_result

        # Capture options from show_options
        if name == "show_options" and isinstance(result, dict):
            budget.set_options(result.get("options", []) or [])

        # Capture refinement suggestions as options
        if name == "generate_search_suggestions" and isinstance(result, dict):
            suggestions = result.get("suggestions") or []
            if isinstance(suggestions, list):
                labels = []
                for suggestion in suggestions:
                    if isinstance(suggestion, dict):
                        label = suggestion.get("label") or suggestion.get("text")
                    else:
                        label = suggestion
                    if label:
                        labels.append(label)
                budget.set_options(labels)

        ms = int((time.perf_counter() - t0) * 1000)
        budget.consume(name or "unknown", ms, success=True)
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result, ensure_ascii=False),
        }
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        budget.consume(name or "unknown", ms, success=False)
        logger.error("❌ Tool execution failed", tool=name, duration_ms=ms, error=str(e))
        logger.error(traceback.format_exc())
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps({"error": str(e)}),
        }


# =============================================================================
# Conversation Logic (Responses API)
# =============================================================================
async def run_conversation(
    user_id: str, 
    message: str, 
    thread_id: str, 
    history: List[Dict[str, str]] = [],
    token_callback: Optional[callable] = None
) -> Dict[str, Any]:
    logger.info(
        "🎯 NEW CONVERSATION",
        user_id=user_id,
        thread_id=thread_id,
        message=message[:200],
    )

    try:
        await Services.ensure_loaded()
    except Exception as e:
        logger.error("❌ Services failed to load", error=str(e))
        # Return structured dict even on startup failure
        return {
            "text": "Sorry, I am having trouble starting up. Please try again in a moment 🔧",
            "products": [],
            "search_result": {},
        }

    # Fetch cached trend context (non blocking from user point of view)
    try:
        trends_text = await get_fashion_trends_text()
    except Exception as e:
        logger.warning("⚠️ Trend context failed", error=str(e))
        trends_text = ""

    budget = Budget(
        max_calls=Config.MAX_TOOL_CALLS,
        max_latency_ms=Config.MAX_LATENCY_MS,
    )

    conversation: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "system",
            "content": f"Context: UserID={user_id} | ThreadID={thread_id}",
        },
        {
            "role": "system",
            "content": (
                "An ACK might have been sent. In this main reply, do NOT greet again or repeat their name unless necessary. "
                "Start directly with the outfit/help and follow the product-forward rules."
            ),
        },
    ]

    if trends_text:
        conversation.append(
            {
                "role": "system",
                "content": f"Cached fashion trend context for India:\n{trends_text}",
            }
        )

    # Inject history (memory)
    if history:
        # Sanitize history to ensure only role and content are passed
        clean_history = [{"role": m["role"], "content": m["content"]} for m in history if "role" in m and "content" in m]
        conversation.extend(clean_history)

    conversation.append({"role": "user", "content": message})

    conv_t0 = time.perf_counter()

    try:
        response: Any = None
        for iteration in range(8):
            llm_t0 = time.perf_counter()
            logger.info("🤖 Calling LLM", model=Config.MAIN_MODEL, iteration=iteration + 1)

            # Reverting to non-streaming call to ensure reliability with the custom 'responses' endpoint
            # We will simulate streaming output for the frontend UX
            # Using asyncio.to_thread to prevent blocking the event loop
            response = await asyncio.to_thread(
                client.responses.create,
                model=Config.MAIN_MODEL,
                input=conversation,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                reasoning={"effort": Config.REASONING_EFFORT},
                max_output_tokens=800,
                max_tool_calls=Config.MAX_TOOL_CALLS,
                parallel_tool_calls=True,
            )

            llm_ms = int((time.perf_counter() - llm_t0) * 1000)
            op_name = "llm_initial" if iteration == 0 else f"llm_iteration_{iteration}"
            logger.perf(op_name, llm_ms, model=Config.MAIN_MODEL)

            # Separate blocks for logging and decision making
            reasoning_blocks = [
                block for block in response.output if block.type == "reasoning"
            ]
            function_calls = [
                block for block in response.output if block.type == "function_call"
            ]
            message_blocks = [
                block for block in response.output if block.type == "message"
            ]
            final_text = (response.output_text or "").strip()

            logger.debug(
                "🔄 Iteration summary",
                iteration=iteration + 1,
                num_reasoning=len(reasoning_blocks),
                num_function_calls=len(function_calls),
                num_messages=len(message_blocks),
            )

            # Add all assistant output blocks IN ORDER to maintain reasoning->function_call pairing
            # OpenAI requires: reasoning MUST be immediately followed by function_call or message
            for block in response.output:
                if block.type in ("reasoning", "message"):
                    conversation.append(block.to_dict())
                elif block.type == "function_call":
                    # Add function_call and then execute it
                    conversation.append(block.to_dict())

            # Execute function calls and add their outputs
            if function_calls:
                logger.info("🔧 Executing %d tools" % len(function_calls))
                tool_outputs = await asyncio.gather(
                    *[_run_single_tool_call(fc, budget) for fc in function_calls]
                )
                for out_block in tool_outputs:
                    conversation.append(out_block)

            # If tools were used, loop again to let the model read them
            if function_calls:
                continue

            if not final_text:
                logger.warning("⚠️ Empty LLM reply, falling back to FAST_MODEL")
                try:
                    fb = await asyncio.to_thread(
                        client.responses.create,
                        model=Config.FAST_MODEL,
                        input=[
                            {
                                "role": "system",
                                "content": (
                                    SYSTEM_PROMPT
                                    + "\n\nYou previously returned an empty reply. "
                                      "Now answer the user directly in 1 or 2 short sentences with 1 or 2 emojis. "
                                      "Do NOT call tools, just respond."
                                ),
                            },
                            {"role": "user", "content": message},
                        ],
                        max_output_tokens=200,
                    )
                    final_text = (fb.output_text or "").strip()
                except Exception as e:
                    logger.error("❌ Fallback LLM failed", error=str(e))
                    final_text = (
                        "Sorry, I glitched and could not finish that answer. "
                        "Could you try again once 😅"
                    )

            # Simulate streaming for UX
            if token_callback and final_text:
                chunk_size = 8
                for i in range(0, len(final_text), chunk_size):
                    await token_callback(final_text[i : i + chunk_size])
                    await asyncio.sleep(0.005)

            total_ms = int((time.perf_counter() - conv_t0) * 1000)
            budget_summary = budget.get_summary()

            logger.success(
                "✅ Conversation complete",
                duration_ms=total_ms,
                iterations=iteration + 1,
                **budget_summary,
            )
            logger.perf("conversation_total", total_ms, **budget_summary)
            logger.save_perf()

            await _ensure_fallback_options(budget, message)

            return {
                "text": final_text,
                "products": budget.get_all_products(Config.DISPLAY_PRODUCTS_COUNT),
                "search_result": budget.last_search_result,
                "options": budget.last_options,
            }

        logger.warning("⚠️ Max iterations (8) reached")
        await _ensure_fallback_options(budget, message)
        return {
            "text": "I got a bit carried away there 😅 Could you say it a bit simpler",
            "products": budget.get_all_products(Config.DISPLAY_PRODUCTS_COUNT),
            "search_result": budget.last_search_result,
            "options": budget.last_options,
        }

    except Exception as e:
        total_ms = int((time.perf_counter() - conv_t0) * 1000)
        logger.error("❌ Conversation failed", duration_ms=total_ms, error=str(e))
        await _ensure_fallback_options(budget, message)
        logger.error(traceback.format_exc())
        return {
            "text": "Oops, something went wrong on my side. Try once more 😅",
            "products": budget.get_all_products(Config.DISPLAY_PRODUCTS_COUNT),
            "search_result": budget.last_search_result,
            "options": budget.last_options,
        }


# =============================================================================
# Streaming helpers
# =============================================================================
async def _ensure_fallback_options(budget: Budget, user_message: str):
    """
    Guarantee the UI receives some actionable options even if the model forgets
    to call show_options / generate_search_suggestions.
    """
    if budget.last_options:
        return

    search_result = budget.last_search_result or {}
    products = search_result.get("products") or budget.last_products
    if not products:
        return

    queries = search_result.get("queries_used") or []
    base_query = queries[0] if queries else user_message

    context_parts = [
        f"User message: {user_message}",
    ]
    if search_result.get("search_type"):
        context_parts.append(f"Search type: {search_result['search_type']}")
    if queries:
        context_parts.append(f"Queries used: {', '.join(queries[:3])}")
    context = " | ".join(context_parts)

    try:
        logger.info("Auto generating fallback options", query=base_query)
        response = await _generate_search_suggestions(query=base_query, context=context)
    except Exception as e:
        logger.warning("Fallback options generation failed", error=str(e))
        return

    suggestions = response.get("suggestions") or []
    labels: List[str] = []
    for item in suggestions:
        label = None
        if isinstance(item, dict):
            label = item.get("label") or item.get("text")
        else:
            label = str(item)
        if label:
            labels.append(label)

    if labels:
        budget.set_options(labels)


def _make_meta_chunk(kind: str, payload: Any) -> Optional[str]:
    """
    Wrap metadata payloads so the frontend can reliably parse non-text events.
    """
    try:
        encoded = json.dumps({"type": kind, "data": payload}, ensure_ascii=False, default=str)
        return f"{META_CHUNK_PREFIX}{encoded}"
    except Exception as e:
        logger.error("�?�,? Failed to encode stream metadata", kind=kind, error=str(e))
        return None


# =============================================================================
# Streaming wrapper
# =============================================================================
async def run_conversation_stream(
    user_id: str, message: str, thread_id: str, history: List[Dict[str, str]] = []
) -> AsyncGenerator[str, None]:
    logger.info("📡 STREAMING conversation", user_id=user_id, message=message[:50])

    ack_t0 = time.perf_counter()
    
    # Cheap ACK logic
    ack = cheap_ack(message)
    if ack:
        ack_ms = int((time.perf_counter() - ack_t0) * 1000)
        logger.success("✅ ACK generated (local)", duration_ms=ack_ms, ack=ack)
        yield f"__ACK__{ack}"
    else:
        logger.info("⏩ Skipping ACK (not a greeting)")
        
    # Old LLM ACK logic removed
    # try:
    #     logger.info("⚡ Generating ACK")
    #     ack_resp = await asyncio.to_thread(...)

    await asyncio.sleep(0.1)

    logger.info("🤖 Starting full response")
    
    # We use a queue to bridge the callback to the generator
    queue = asyncio.Queue()
    
    async def _callback(token: str):
        await queue.put(token)
        
    # Start conversation in background
    task = asyncio.create_task(
        run_conversation(user_id, message, thread_id, history=history, token_callback=_callback)
    )
    
    # Yield tokens as they come
    while not task.done():
        try:
            # Wait for token or task completion
            # We use a small timeout to check task status frequently
            token = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield token
        except asyncio.TimeoutError:
            continue
            
    # Flush any remaining tokens
    while not queue.empty():
        yield await queue.get()
        
    # Get final result to ensure we catch any exceptions
    final = await task
    
    # Yield metadata chunks so the UI can render products/options
    if isinstance(final, dict):
        products = final.get("products") or []
        if products:
            chunk = _make_meta_chunk("products", products)
            if chunk:
                logger.debug("Streaming products metadata", count=len(products))
                yield chunk

        options = final.get("options") or []
        if options:
            chunk = _make_meta_chunk("options", options)
            if chunk:
                logger.debug("Streaming options metadata", count=len(options))
                yield chunk
    # We don't yield final['text'] here because it would duplicate what was streamed.


# =============================================================================
# Gradio Integration
# =============================================================================
async def gradio_handler(
    message: str, history: List, user_id: str = "demo"
) -> AsyncGenerator[str, None]:
    thread_id = f"webui-{user_id}"
    logger.info("🌐 Gradio request", user_id=user_id, thread_id=thread_id)

    full_response = ""
    async for chunk in run_conversation_stream(user_id, message, thread_id, history=history):
        if full_response:
            yield chunk
        else:
            full_response = chunk
            yield chunk
            await asyncio.sleep(0.3)


def create_gradio_interface():
    import gradio as gr

    demo = gr.ChatInterface(
        fn=gradio_handler,
        type="messages",
        title="🎨 MuseBot – MUSE India Fashion Stylist",
        description="Chat with MuseBot, the stylist for MUSE, an India first fashion discovery platform. Ask for outfits, styling help, or product ideas.",
        examples=[
            "What should I wear for a date",
            "I have traditional day in office, help",
            "I am travelling to Delhi next month, what to pack",
            "Need some casual shirts under ₹2000",
        ],
    )
    return demo


# =============================================================================
# CLI Testing
# =============================================================================
async def test_cli():
    print("🎨 MuseBot Production Agent - CLI Test")
    print("=" * 60)

    # Warm up services
    await Services.ensure_loaded()

    test_queries = [
        "Hey, I am looking for a blue cotton shirt",
        "What goes well with navy trousers for office",
        "What is trending right now",
        "I have traditional day in office, help",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}/{len(test_queries)}: {query}")
        print("-" * 60)
        try:
            response = await run_conversation(
                "test_user", query, f"test_thread_{i}"
            )
            print(f"🤖 Response:\n{response}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
            logger.error("Test %d failed" % i, error=str(e))

    print("\n" + "=" * 60)
    print(f"📊 Logs saved to: {logger.log_file}")
    print(f"📈 Performance data: {logger.perf_file}")


async def test_streaming():
    print("\n🌊 Testing Streaming ACK")
    print("=" * 60)
    query = "I need a blue shirt for office"
    print(f"Query: {query}\n")
    async for chunk in run_conversation_stream(
        "test_user", query, "test_stream"
    ):
        print(f"📨 Chunk: {chunk}\n")
        await asyncio.sleep(0.5)


# =============================================================================
# Main Entry Points
# =============================================================================
async def main():
    await test_cli()
    await test_streaming()


def launch_gradio(share: bool = False, port: int = 7860):
    logger.info("🚀 Launching Gradio", port=port, share=share)
    # Pre warm services in background so first user hit is not cold
    try:
        asyncio.get_event_loop().create_task(Services.ensure_loaded())
    except RuntimeError:
        # If no loop, just ignore; it will warm on first call
        pass

    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        launch_gradio(share=False, port=7860)
    else:
        asyncio.run(main())
