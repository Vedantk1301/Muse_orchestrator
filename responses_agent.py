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
from collections import OrderedDict
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

    CATALOG_COLLECTION = os.getenv("CATALOG_COLLECTION", "fashion_qwen4b_text")
    HNSW_EF = int(os.getenv("HNSW_EF", "500"))

    SIMPLE_SEARCH_LIMIT = 15
    DISCOVERY_QUERIES = 3
    PRODUCTS_PER_QUERY = 40
    FINAL_RERANK_TOP_K = 16
    DISPLAY_PRODUCTS_COUNT = int(os.getenv("DISPLAY_PRODUCTS_COUNT", "8"))
    MIN_PRODUCTS_TARGET = int(os.getenv("MIN_PRODUCTS_TARGET", "8"))
    NUMERIC_RERANK_POOL = int(os.getenv("NUMERIC_RERANK_POOL", "30"))
    LLM_RERANK_INPUT_LIMIT = int(os.getenv("LLM_RERANK_INPUT_LIMIT", "20"))
    WEB_TOPUP_MIN_COUNT = int(os.getenv("WEB_TOPUP_MIN_COUNT", "2"))

    SEARCH_CACHE_TTL_HOURS = 24
    INTENT_CACHE_TTL_SECONDS = 1800

    # Trend + Weather cache
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

    def record_products(self, products: List[Dict[str, Any]]):
        """
        Track products returned across multiple tool calls so the frontend
        can render everything the assistant referenced.
        """
        if not products:
            return

        self.last_products = products
        for product in products:
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
        aggregated = list(self.aggregated_products.values())
        if not aggregated:
            aggregated = list(self.last_products)

        if limit is not None and aggregated:
            return aggregated[:limit]
        return aggregated


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
async def t_classify_intent(
    query: str,
    user_gender: Optional[str] = None,
    forced_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify the fashion search intent. Uses FAST_MODEL (Responses API) and caches
    results. Uses plain JSON output for robustness.
    - Always uses gender neutral language in queries.
    - For discovery or pairing, always returns 3 short SKU like queries.
    - Queries MUST be fashion / catalog friendly only (no 'under 5000', no generic chit chat).
    """
    # Use gender if provided
    # user_gender = None

    cache_key = _cache_key("intent", query.lower(), forced_type or "")
    cached = await INTENT_CACHE.get(cache_key)
    if cached:
        logger.debug("💾 intent cache HIT")
        return cached

    t0 = time.perf_counter()
    logger.info("🧠 classify_intent", query=query[:80], forced_type=forced_type)

    system_prompt = """
You are an intent classifier for a fashion search bot.

You must return ONLY a JSON object with this exact schema (no extra text, no explanations):

{
  "search_type": "specific" | "discovery" | "pairing",
  "queries": ["string", "string", ...]
}

Rules for ALL queries:
- They must be SIMPLE, DIRECT fashion product search strings.
- Examples: "Cotton Linen Shorts", "Navy Blue Trousers", "Beige Oversized T Shirt", "Floral Midi Dress".
- Only talk about clothing, footwear, and accessories.
- All queries must be gender neutral. Do NOT use words like "men", "women", "male", "female".
  - You MAY use "unisex" if you want a neutral signal.
- Do NOT include any price or budget words in queries.
- Do NOT include generic prompts like "what is trending" or "good outfits".
- Do NOT include non fashion topics like weather, politics, coding, AI, etc.

FORBIDDEN WORDS (Do NOT use these unless explicitly part of the product name):
- "India"
- "Trend", "Trending", "Viral"
- "Winterwear", "Summerwear"
- "Best", "Top", "Cheap", "Online"

Types:
- "specific":
  - User is clearly looking for ONE focused thing.
  - Return exactly 1 tight query in `queries`.
- "discovery":
  - User is browsing, says things like "ideas", "options", "trending", "what's in", "show me some".
  - Return exactly 3 short, concrete queries that each describe one product family.
  - Make them diverse but simple.
- "pairing":
  - User wants items that go with another item.
  - Return exactly 3 short, complementary queries.

Remember:
- `queries` MUST be short and simple, not long paragraphs.
- NO "India" suffix.
- NO "Trending" prefix.
""".strip()

    try:
        response = await asyncio.to_thread(
            client.responses.create,
            model=Config.FAST_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps({"query": query}),
                },
            ],
            max_output_tokens=200,
        )

        raw_text = (response.output_text or "").strip()
        parsed: Optional[Dict[str, Any]] = None

        if raw_text:
            try:
                parsed = json.loads(raw_text)
            except Exception as e:
                logger.error(
                    "❌ classify_intent JSON parse failed",
                    raw=raw_text,
                    error=str(e),
                )

        if not parsed:
            # Safe default
            result = {"search_type": "specific", "queries": [query]}
            await INTENT_CACHE.set(cache_key, result)
            return result

        search_type = parsed.get("search_type", "specific")
        if forced_type in ("specific", "discovery", "pairing"):
            search_type = forced_type

        queries = parsed.get("queries", [query])

        # Normalize queries and enforce counts
        if search_type == "specific":
            queries = queries[:1]
        else:
            # Discovery or pairing: enforce 3 queries
            if len(queries) < 3:
                base = query
                expanded = []
                for q in queries:
                    if q and q not in expanded:
                        expanded.append(q)
                while len(expanded) < 3:
                    expanded.append(base)
                queries = expanded[:3]
            else:
                queries = queries[:3]

        result = {"search_type": search_type, "queries": queries}
        await INTENT_CACHE.set(cache_key, result)

        ms = int((time.perf_counter() - t0) * 1000)
        logger.success(
            "✅ classify_intent",
            duration_ms=ms,
            search_type=search_type,
            num_queries=len(queries),
            parsed=result,
        )
        return result
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.error("❌ classify_intent failed", duration_ms=ms, error=str(e))
        # Safe default
        result = {"search_type": "specific", "queries": [query]}
        await INTENT_CACHE.set(cache_key, result)
        return result


# =============================================================================
# Catalog search helpers
# =============================================================================
def _product_identity(product: Dict[str, Any]) -> Optional[str]:
    return (
        product.get("product_id")
        or product.get("id")
        or product.get("url")
    )


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


async def _numeric_rerank_products(user_query: str, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(products) <= 1:
        return products

    pool_limit = min(Config.NUMERIC_RERANK_POOL, len(products))
    pool = products[:pool_limit]
    remainder = products[pool_limit:]

    texts = [
        f"{p.get('title') or ''} {p.get('brand') or ''} {p.get('category') or ''}".strip()
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
                "brand": product.get("brand"),
                "category_leaf": product.get("category") or product.get("category_leaf"),
                "score": round(float(product.get("score", 0.0)), 4),
                "source": product.get("from_query") or product.get("source") or "catalog",
            }
        )

    system_prompt = """
You are a ranking model for a fashion shopping assistant.
Reorder the candidate products so that the top items best match the user request.
- Drop only the products that are clearly irrelevant.
- If at least `min_results` items are relevant, return at least that many.
- Preserve the IDs exactly as provided.
- Respond ONLY with a JSON object:
  {"ordered_ids": ["id1", "id2", ...]}
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

        llm_ms = int((time.perf_counter() - llm_t0) * 1000)
        logger.perf(
            "rerank_llm",
            llm_ms,
            in_count=len(candidates),
            out_count=len(ordered),
        )
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
        logger.perf(
            "web_search_topup",
            web_ms,
            requested=request_count,
            received=len(web_results),
        )
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
) -> Dict[str, Any]:
    """
    - Always treats searches as gender neutral internally.
    - For discovery / pairing: expands to 3 short fashion-only queries and runs them in PARALLEL.
    - Combines products from all queries, dedupes by product id, then reranks (if needed).
    """
    # Use gender if provided
    # user_gender = None

    cache_key = _cache_key("search", query.lower(), search_type)
    cached = await SEARCH_CACHE.get(cache_key)
    if cached:
        logger.success("💾 search cache HIT")
        return cached

    t0 = time.perf_counter()
    logger.info("🔍 search_fashion_products", query=query, search_type=search_type)

    try:
        await Services.ensure_loaded()

        # Decide when to call classifier:
        # - auto / discovery / pairing => classifier with optional forced type
        # - specific => just one query
        ql = query.lower()
        if search_type in ("auto", "discovery", "pairing"):
            forced_type = None if search_type == "auto" else search_type
            intent = await t_classify_intent(query, user_gender, forced_type=forced_type)
            search_type = intent["search_type"]
            queries = intent["queries"]
        else:
            search_type = "specific"
            queries = [query]

        # Extra specialisation for "trending" like queries
        if "trend" in ql or "trending" in ql:
            search_type = "discovery"
            # Rely on classifier for queries, do not hardcode "India" ones here.

        logger.debug("📋 Search queries", type=search_type, queries=queries)

        # Embedding
        embed_t0 = time.perf_counter()
        vectors = await Services.embed(queries)
        embed_ms = int((time.perf_counter() - embed_t0) * 1000)
        logger.perf("embed", embed_ms, num_queries=len(queries))

        # Qdrant search
        from qdrant_client.http import models as rest

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
                    query_filter=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="category_path",
                                match=rest.MatchText(text=category_filter),
                            )
                        ]
                    ) if category_filter else None,
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
            min(Config.FINAL_RERANK_TOP_K * 2, Config.PRODUCTS_PER_QUERY)
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
                commerce = payload.get('commerce') or {}
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

                card = {
                    'id': pid,
                    'product_id': pid,
                    'title': payload.get('title'),
                    'brand': payload.get('brand'),
                    'category': payload.get('category_leaf'),
                    'category_leaf': payload.get('category_leaf'),
                    'category_path': payload.get('category_path'),
                    'image_url': payload.get('primary_image')
                    or payload.get('image_url'),
                    'url': payload.get('url'),
                    'price': commerce.get('price'),
                    'price_inr': commerce.get('price_inr'),
                    'score': score,
                    'from_query': q,
                    'source_tags': payload.get('source_tags') or [],
                }
                q_products.append(card)
            results_lists.append(q_products)

        interleaved = _interleave_results(results_lists)
        candidates = _dedupe_products(interleaved)
        total_candidates = len(candidates)

        if len(queries) == 1:
            candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        logger.debug(
            'Aggregated products',
            count=total_candidates,
            best_score=best_score,
        )

        base_products = candidates
        if base_products:
            base_products = await _numeric_rerank_products(query, base_products)
            base_products = _dedupe_products(base_products)

        final_products = base_products
        if final_products:
            llm_ranked = await _llm_rerank_products(
                query, final_products, Config.MIN_PRODUCTS_TARGET
            )
            if llm_ranked:
                final_products = llm_ranked

        web_products: List[Dict[str, Any]] = []
        if len(final_products) < Config.MIN_PRODUCTS_TARGET:
            needed = Config.MIN_PRODUCTS_TARGET - len(final_products)
            existing_keys = {
                key
                for key in (_product_identity(p) for p in final_products)
                if key
            }
            web_products = await _fetch_web_topup_products(
                query, existing_keys, needed
            )
            if web_products:
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

        storage_limit = max(Config.FINAL_RERANK_TOP_K, Config.DISPLAY_PRODUCTS_COUNT)
        stored_products = final_products[:storage_limit]
        display_products = stored_products[: Config.DISPLAY_PRODUCTS_COUNT]

        result_payload = {
            "products": display_products,
            "all_products": stored_products,
            "total_found": len(final_products),
            "search_type": search_type,
            "queries_used": queries,
            "best_score": best_score,
            "total_candidates": total_candidates,
            "web_topup_count": len(web_products),
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
            "search_type": search_type,
            "queries_used": [query],
            "best_score": 0.0,
            "error": str(e),
        }


# =============================================================================
# Post-search suggestions (chips) – Under ₹5000, colours, pairing
# =============================================================================
async def t_generate_search_suggestions(
    query: str, context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate 3-4 *refinement suggestions* to show AFTER products.

    These are NOT new search queries fired automatically.
    They are UI chips the user can click, like:
      - "Under ₹5000"
      - "Show different colours"
      - "Pair this with something"

    The assistant should:
      - show these suggestions after listing products
      - ask the user which one they prefer
      - only then call search_fashion_products again with a refined query / filters.
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
async def t_show_options(options: List[str]) -> Dict[str, Any]:
    """
    Display clickable UI chips/buttons to the user.
    """
    logger.info("🔘 show_options", options=options)
    return {"options": options}


TOOLS_MAP = {
    "search_fashion_products": t_search_fashion_products,
    "tone_reply": t_tone_reply,
    "get_weather": t_get_weather,
    "generate_search_suggestions": t_generate_search_suggestions,
    "show_options": t_show_options,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "name": "search_fashion_products",
        "description": (
            "Smart fashion search backed ONLY by the internal catalog (Qdrant). "
            "Use this for any request that involves buying, showing, or recommending items. "
            "Provide ONE tight fashion query (material + category + vibe) and the tool will auto-expand discovery/pairing cases into up to 3 short searches, dedupe, rerank (vector + LLM), and guarantee at least 8 strong products with web-search fallback when needed. "
            "Do NOT call this repeatedly in the same reply unless the user clearly asks for a brand-new category or filter. "
            "BAD Query: 'masculine wardrobe staples', 'party wear', 'office outfit'. "
            "GOOD Query: 'White Linen Shirt', 'Navy Blue Chinos', 'Black Leather Boots', 'Beige Cotton Trousers'. "
            "The search engine only understands: Material, Color, Fit, Category, Pattern."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "user_gender": {"type": "string"},
                "category_filter": {
                    "type": "string",
                    "description": "Optional category keyword to filter by (e.g. 'ethnic', 'traditional').",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["auto", "specific", "discovery", "pairing"],
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
        "name": "generate_search_suggestions",
        "description": (
            "Generate 3–4 post-search refinement suggestions like 'Under ₹5000', "
            "'Show different colours', or 'Pair this with something'. "
            "Call this ONLY after you have already shown some products from search_fashion_products."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "show_options",
        "description": (
            "Show clickable UI buttons/chips to the user. "
            "Use this whenever you ask a multiple-choice question (e.g. Gender, Budget, Occasion) "
            "or when you want to offer quick replies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of short button labels (max 4-5 items)",
                },
            },
            "required": ["options"],
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
- When the user asks for products (e.g. "I want shirts"), IF you don't know the gender yet, ask for it THEN (and use `show_options`).
- Do NOT search until you have this preference.

OVERALL UX VIBE:
- Replies must be short and product forward.
  - Maximum 1 or 2 short sentences before the product bullets.
  - Use a bullet list for products, then ONE short follow up question.
  - Use 2 or 3 emojis in most answers (at least 1). Playful but not cringe.
- Less lecture, more looks.
- Never use long punctuation dashes like — or –. Use commas, full stops, or emojis instead.
- Never use the rainbow emoji.

1) Brand and “What is MUSE”:
- If the user asks “What is Muse or MUSE or what do you do”:
  - In 1 or 2 sentences:
    - Say that MUSE is an India based fashion discovery platform curating Indian and global brands.
    - Say that MuseBot is the chat stylist that plans outfits and surfaces real catalog products.
  - Example:
    - “I am MuseBot, the chat stylist for MUSE, an India first fashion discovery platform. I help you plan outfits and find real pieces you can actually buy 🙂✨”

2) Domain lock:
- You only talk about fashion, style, grooming, clothing, footwear, and accessories.
- If the user asks non fashion things (coding, elephants, matcha, politics, etc):
  - Give ONE playful line, for example:
    - “I am only licensed to style outfits, not politics, but we can still make your look iconic 🐘✨”
  - Then nudge back with a simple fashion question:
    - “What are you dressing for next, work, date, travel, or festive”

3) India first styling rules:
- For “traditional day”, “ethnic day”, “festive”, “Diwali”, “puja”, “wedding”, “sangeet”, “mehendi”:
  - Prefer:
    - Cotton or linen kurtas, churidar or straight pants, Nehru or Modi jackets, bandhgalas, juttis or loafers.
    - You can still add 1 or 2 smart Western looks like chinos with a shirt, but ethnic should be visible.
    - CRITICAL: When searching for these, you MUST set `category_filter='ethnic'` in `search_fashion_products`.
- For normal “office”, “meeting”, “interview”:
  - Talk in terms of modern Indian office wear: shirts, chinos, minimal sneakers or loafers, sometimes blazers or suits.
- For “travel to <city> in <month> or next week”:
  - Mention the climate in one short line.
  - If month or season is unclear, ask ONE small clarifying question:
    - “Which month are you going, May vs December is very different”

4) Product grounding:
- The function “search_fashion_products” is your only source of real products, brands, and prices. Call it ONCE per user ask; it already fans out discovery/pairing queries, reranks (vector + LLM), and guarantees at least 8 relevant products with a web-search fallback when the catalog is thin.
- Never invent a brand or product that is not in the tool output.
- When the products list is non empty:
  - Treat them as valid matches and speak from that list.
  - Show up to 8 bullets by default (the UI only displays the top 8). For broader discovery or if the user explicitly asks for more, you may reference additional cached products.
  - Bullet style must be aesthetic, for example:
    - “A crisp light blue cotton shirt from Rare Rabbit, sharp contrast with navy trousers for office days 🙂”
    - “A relaxed navy shacket from Cultstore, throw on over a tee and chinos for casual dates 😌”
  - Do not list sizes, prices, or color arrays unless the user explicitly asks about budget, size, or color.
- Only when products is empty:
  - Say “I do not have good matches for that in the catalog right now”.
  - Then give high level outfit advice without fabricating specific SKUs.

5) Tool usage:
- For any outfit, clothing, shoes, or shopping question:
  - Make ONE call to “search_fashion_products” early, even if information is incomplete. Let the tool handle its own multi-query fan-out and reranking; do not spam multiple calls in the same reply unless the user clearly asks for a totally different category later.
- For pairing or discovery questions (e.g., “what goes with navy trousers”):
  - Set `search_type="discovery"` (or leave it blank/auto) so the tool automatically generates 3 concise queries and reranks them.
  - Use a neutral but relevant query phrase like “smart casual shirts India” or “knit polos for office India”, then pick colours in the answer.
- When building the search query string:
  - Keep it gender neutral (no “men/women”), “unisex” is OK.
  - Do NOT include price or budget words like “under 2000”, “cheap”, “budget”; handle those via follow ups.
- When the user refines with size, budget, colour, or vibe:
  - You may call “search_fashion_products” again with that new filter, otherwise reuse the cached products.
- Weather:
  - If the user asks “what is the weather in <city> today or this week” or asks what to pack based on weather, you may call “get_weather”.
  - Use the weather only as a short helper line to adjust fabrics and layers.
  - Do not mention Open Meteo or any API by name in your reply.

6) Trends and Korean trousers:
- You will receive a separate system message with cached fashion trend notes for western and ethnic wear.
- Use this only as light seasoning, not strict truth.
- For any question like “what is trending” or “what are the latest trends”:
  - First, read the cached trend system message.
  - Give exactly ONE short sentence summarising what is trending using that context.
  - Then, if helpful, call “search_fashion_products” with short discovery queries (for example oversized shirts, Korean trousers, co ord sets).
- For example, if the user searches for “Korean trousers” or “oversized shirts”:
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
- The question about name or gender should be separate from other questions and not spammy.
- End each message with just one simple next step question.

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
  2) Bullet list of products in the stylist voice.
  3) One short question to move forward (size, budget, vibe, or occasion).
- Example of good bullets without long dashes:
  - “A soft off white mandarin collar kurta from Fabindia, perfect for office traditional day without feeling overdressed 🎉”
  - “A navy knit polo from Rare Rabbit, easy upgrade from a tee that still feels relaxed for Fridays 🙂”

10) Use of the tone_reply tool:
- If your natural answer is correct but wordy or stiff:
  - You may call “tone_reply” with your draft and a few products.
  - Then send only the polished reply.

11) Post-search suggestions:
- After you call “search_fashion_products” and show some products, you may call “generate_search_suggestions” once.
- Use its output to display 2 or 3 refinement options like:
  - “Under ₹5000”
  - “Show different colours”
  - “Pair this with something” (only if relevant)
- Do NOT automatically apply these filters.
  - Instead, show the suggestions as short chips in text and ask the user which one they want.
  - When the user picks one, THEN call “search_fashion_products” again with a refined query or filters based on their choice.

12) Interactive Options:
- Whenever you ask a question with clear choices (e.g. "Masculine or Feminine?", "Work or Party?", "Under 2k or 5k?"):
  - Call the "show_options" tool with the list of choices.
  - This will show clickable buttons to the user.
  - Example: show_options(options=["Masculine", "Feminine", "Neutral"])

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
        result = await fn(**args)

        # Capture last product search for frontend
        if name == "search_fashion_products" and isinstance(result, dict):
            budget.last_search_result = result
            source_products = result.get("all_products") or result.get("products") or []
            budget.record_products(source_products)

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

            reasoning_blocks = [
                block.to_dict() for block in response.output if block.type == "reasoning"
            ]
            function_calls = [
                block for block in response.output if block.type == "function_call"
            ]
            message_blocks = [
                block.to_dict() for block in response.output if block.type == "message"
            ]
            final_text = (response.output_text or "").strip()

            logger.debug(
                "🔄 Iteration summary",
                iteration=iteration + 1,
                num_reasoning=len(reasoning_blocks),
                num_function_calls=len(function_calls),
                num_messages=len(message_blocks),
            )

            if reasoning_blocks:
                conversation.extend(reasoning_blocks)

            if function_calls:
                logger.info("🔧 Executing %d tools" % len(function_calls))
                tool_outputs = await asyncio.gather(
                    *[_run_single_tool_call(fc, budget) for fc in function_calls]
                )
                for fc, out_block in zip(function_calls, tool_outputs):
                    conversation.append(fc.to_dict())
                    conversation.append(out_block)

            if message_blocks:
                conversation.extend(message_blocks)

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
        response = await t_generate_search_suggestions(query=base_query, context=context)
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
    try:
        logger.info("⚡ Generating ACK")
        ack_resp = await asyncio.to_thread(
            client.responses.create,
            model=Config.FAST_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are MuseBot, a helpful and stylish fashion assistant. \n"
                        "Your goal: Acknowledge the user's message naturally and briefly. \n"
                        "Rules:\n"
                        "1. If the user says 'Hello', 'Hi', 'Hey': Reply with 'Hey there!', 'Hi!', 'Hello!'. \n"
                        "2. If the user gives a task/preference: Say 'Got it', 'Sure', 'On it', 'Understood'. \n"
                        "3. Max 5 words. An emoji or two to be used. \n"
                        "4. NEVER say 'I will help you' or 'Let us fix your fits'. Be concise.\n"
                        "5. Avoid overusing slang like 'Bet' or 'Say less' unless it fits perfectly. Keep it polite but modern."
                    ),
                },
                *[{"role": m["role"], "content": m["content"]} for m in history[-4:] if "role" in m and "content" in m], # Context for ACK
                {"role": "user", "content": message},
            ],
            reasoning={"effort": "low"},
            max_output_tokens=500,
        )
        ack = (ack_resp.output_text or "").strip()
        
        # Debug empty response
        if not ack:
            logger.warning("⚠️ ACK response was empty", raw_resp=str(ack_resp))
            import random
            fallbacks = ["Bet!", "On it!", "Say less.", "Gotcha.", "Cooking that up...", "Yo, checking!"]
            ack = random.choice(fallbacks)
            
        ack_ms = int((time.perf_counter() - ack_t0) * 1000)
        logger.success("✅ ACK generated", duration_ms=ack_ms, ack=ack)
        yield ack
    except Exception as e:
        import random
        fallbacks = ["Bet!", "On it!", "Say less.", "Gotcha.", "Cooking that up...", "Yo, checking!"]
        fallback_ack = random.choice(fallbacks)
        logger.warning("⚠️ ACK generation failed", error=str(e))
        yield fallback_ack

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
    async for chunk in run_conversation_stream(user_id, message, thread_id):
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
