# data/news.py
import html
import json
import math
import os
import re
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests

from utils.logger import get_logger

log = get_logger(__name__)

_NEWS_CACHE_TTL: int = 300          # 5 minutes
_NEWS_BUFFER_SIZE: int = 200        # Rolling buffer max items
_DEDUP_PREFIX_LEN: int = 40         # Title prefix length for dedup
_FETCH_TIMEOUT: int = 5             # HTTP timeout for news fetchers
_SENTIMENT_NEUTRAL_BAND: float = 0.2  # |score| below this is neutral
_MAX_RETRIES: int = 2               # Retry attempts for failed requests
_RETRY_DELAY_BASE: float = 0.5      # Base delay between retries (seconds)
_RATE_LIMIT_DELAY: float = 0.3      # Delay between API calls to avoid rate limiting

# Positive keywords (Chinese financial, Unicode-escaped to avoid encoding drift)
POSITIVE_WORDS: dict[str, float] = {
    "\u6da8\u505c": 2.0,
    "\u5927\u6da8": 1.8,
    "\u66b4\u6da8": 1.8,
    "\u521b\u65b0\u9ad8": 1.5,
    "\u7a81\u7834": 1.3,
    "\u5229\u597d": 1.5,
    "\u91cd\u5927\u5229\u597d": 2.0,
    "\u8d85\u9884\u671f": 1.5,
    "\u4e1a\u7ee9\u5927\u589e": 1.8,
    "\u51c0\u5229\u6da6\u589e\u957f": 1.3,
    "\u8425\u6536\u589e\u957f": 1.2,
    "\u76c8\u5229": 1.0,
    "\u626d\u4e8f": 1.5,
    "\u56de\u8d2d": 1.2,
    "\u589e\u6301": 1.3,
    "\u673a\u6784\u4e70\u5165": 1.3,
    "\u5317\u5411\u8d44\u91d1\u6d41\u5165": 1.2,
    "\u4e0a\u6da8": 0.8,
    "\u53cd\u5f39": 0.7,
    "\u964d\u606f": 1.0,
    "\u652f\u6301": 0.6,
    "\u4e2d\u6807": 1.0,
    "\u5408\u4f5c": 0.6,
    "\u6280\u672f\u7a81\u7834": 1.0,
    "\u8ba2\u5355\u589e\u957f": 1.0,
    "\u653f\u7b56\u652f\u6301": 1.0,
    "\u65b0\u80fd\u6e90": 0.6,
}

NEGATIVE_WORDS: dict[str, float] = {
    "\u8dcc\u505c": -2.0,
    "\u5927\u8dcc": -1.8,
    "\u66b4\u8dcc": -1.8,
    "\u5d29\u76d8": -2.0,
    "\u5229\u7a7a": -1.5,
    "\u91cd\u5927\u5229\u7a7a": -2.0,
    "\u7206\u96f7": -2.0,
    "\u8fdd\u89c4": -1.5,
    "\u5904\u7f5a": -1.5,
    "\u9000\u5e02": -2.0,
    "\u4e8f\u635f": -1.3,
    "\u4e1a\u7ee9\u4e0b\u6ed1": -1.5,
    "\u51c0\u5229\u6da6\u4e0b\u964d": -1.3,
    "\u51cf\u6301": -1.3,
    "\u8d28\u62bc": -0.8,
    "\u7206\u4ed3": -1.8,
    "\u8fdd\u7ea6": -1.5,
    "\u4e0b\u8dcc": -0.8,
    "\u56de\u8c03": -0.5,
    "\u52a0\u606f": -0.8,
    "\u76d1\u7ba1": -0.6,
    "\u9650\u5236": -0.6,
    "\u5236\u88c1": -1.0,
    "\u8d38\u6613\u6218": -1.0,
    "\u505c\u4ea7": -1.0,
    "\u8bc9\u8bbc": -0.7,
    "\u8c03\u67e5": -0.7,
    "\u98ce\u9669": -0.5,
    "\u8b66\u544a": -0.6,
    "\u6ce1\u6cab": -0.8,
    "\u901a\u80c0": -0.5,
    "\u5317\u5411\u8d44\u91d1\u6d41\u51fa": -1.0,
}

# Max times a single keyword is counted (prevents spam amplification)
_MAX_KEYWORD_COUNT: int = 3

_NEGATION_PREFIX_HINTS: tuple[str, ...] = (
    "\u4e0d",
    "\u6ca1",
    "\u65e0",
    "\u672a",
    "\u5e76\u975e",
    "\u4e0d\u662f",
    "\u5426\u8ba4",
    "not",
    "no ",
    "without",
    "lack ",
)

_NEGATION_SUFFIX_HINTS: tuple[str, ...] = (
    "\u4e0d\u53ca\u9884\u671f",
    "\u4e0d\u53ca",
    "\u4e0d\u4f73",
    "\u4e0d\u632f",
    "miss",
    "weaker than expected",
)

_UNCERTAINTY_HINTS: tuple[str, ...] = (
    "\u4f20\u95fb",
    "\u6216\u5c06",
    "\u6216\u8bb8",
    "\u53ef\u80fd",
    "\u62df",
    "\u9884\u8ba1",
    "\u9884\u671f",
    "rumor",
    "maybe",
    "might",
    "could",
)

_AMPLIFIER_HINTS: tuple[str, ...] = (
    "\u91cd\u5927",
    "\u5927\u5e45",
    "\u663e\u8457",
    "\u5f3a\u52b2",
    "\u660e\u663e",
    "\u975e\u5e38",
    "strong",
    "sharply",
    "significant",
)

_DIMINISHER_HINTS: tuple[str, ...] = (
    "\u5c0f\u5e45",
    "\u7565",
    "\u8f7b\u5fae",
    "\u6682\u65f6",
    "\u6709\u9650",
    "slightly",
    "mild",
    "limited",
)

_CONTRAST_HINT_RE = re.compile(
    r"(?:\u4f46\u662f|\u4f46|\u7136\u800c|\u4e0d\u8fc7|\u53ef\u662f|but|however|yet)",
    flags=re.IGNORECASE,
)


def _contains_any_token(text: str, tokens: tuple[str, ...]) -> bool:
    """Return True if any token exists in text."""
    if not text:
        return False
    return any(tok in text for tok in tokens)


def _iter_keyword_positions(
    text: str,
    keyword: str,
    max_count: int,
) -> list[int]:
    """Find up to max_count non-overlapping occurrences of keyword in text."""
    out: list[int] = []
    if not text or not keyword or max_count <= 0:
        return out

    start = 0
    while len(out) < max_count:
        idx = text.find(keyword, start)
        if idx < 0:
            break
        out.append(int(idx))
        start = idx + len(keyword)
    return out


def _contrast_boost_for_position(
    text: str,
    position: int,
) -> float:
    """Boost keywords that appear after contrast markers (e.g., 'but')."""
    if not text:
        return 1.0
    contrast_count = sum(
        1 for m in _CONTRAST_HINT_RE.finditer(text) if int(m.start()) < int(position)
    )
    return float(min(1.35, 1.0 + (0.14 * contrast_count)))


def _context_adjusted_weight(
    text: str,
    start: int,
    end: int,
    base_weight: float,
) -> float:
    """Adjust keyword weight with local context heuristics.

    This keeps the scorer lightweight while improving handling of:
    - negation near a keyword (e.g., "not bullish", "\u4e0d\u662f\u5229\u597d")
    - uncertainty/rumor phrasing
    - emphasis/dampening words
    """
    left = text[max(0, int(start) - 16) : int(start)]
    left_tail = left[-8:]
    right = text[int(end) : min(len(text), int(end) + 16)]
    near = f"{left} {right}"

    weight = float(base_weight)
    negated = (
        _contains_any_token(left_tail, _NEGATION_PREFIX_HINTS)
        or _contains_any_token(right, _NEGATION_SUFFIX_HINTS)
    )
    if negated:
        weight *= -0.85

    if _contains_any_token(near, _UNCERTAINTY_HINTS):
        weight *= 0.75

    if _contains_any_token(near, _AMPLIFIER_HINTS):
        weight *= 1.18

    if _contains_any_token(near, _DIMINISHER_HINTS):
        weight *= 0.82

    return float(weight)


def _safe_age_seconds_from_now(
    publish_time: datetime | None,
) -> float | None:
    """Age in seconds vs now for naive/aware datetimes without type errors."""
    if not isinstance(publish_time, datetime):
        return None
    try:
        if publish_time.tzinfo is not None:
            now_dt = datetime.now(tz=publish_time.tzinfo)
        else:
            now_dt = datetime.now()
        return float((now_dt - publish_time).total_seconds())
    except Exception:
        return None


def _safe_age_hours_from_now(
    publish_time: datetime | None,
) -> float | None:
    age_s = _safe_age_seconds_from_now(publish_time)
    if age_s is None:
        return None
    return float(age_s / 3600.0)


def analyze_sentiment(text: str) -> tuple[float, str]:
    """Weighted keyword sentiment scoring.

    - Counts each keyword up to _MAX_KEYWORD_COUNT times
    - Normalizes by total absolute weight contribution
    - Applies tanh to produce smooth [-1, 1] output
    - Returns (score, label) where label in {positive, negative, neutral}
    """
    if not text:
        return 0.0, "neutral"

    normalized_text = str(text).lower()

    raw_score = 0.0
    abs_contrib = 0.0

    for word, weight in POSITIVE_WORDS.items():
        for pos in _iter_keyword_positions(
            normalized_text,
            word,
            _MAX_KEYWORD_COUNT,
        ):
            adj_weight = _context_adjusted_weight(
                normalized_text,
                start=pos,
                end=pos + len(word),
                base_weight=float(weight),
            )
            adj_weight *= _contrast_boost_for_position(normalized_text, pos)
            raw_score += float(adj_weight)
            abs_contrib += abs(float(adj_weight))

    for word, weight in NEGATIVE_WORDS.items():
        for pos in _iter_keyword_positions(
            normalized_text,
            word,
            _MAX_KEYWORD_COUNT,
        ):
            adj_weight = _context_adjusted_weight(
                normalized_text,
                start=pos,
                end=pos + len(word),
                base_weight=float(weight),
            )
            adj_weight *= _contrast_boost_for_position(normalized_text, pos)
            raw_score += float(adj_weight)
            abs_contrib += abs(float(adj_weight))

    if abs_contrib < 1e-9:
        return 0.0, "neutral"

    base = raw_score / abs_contrib
    normalized = math.tanh(1.25 * base)

    if normalized >= _SENTIMENT_NEUTRAL_BAND:
        label = "positive"
    elif normalized <= -_SENTIMENT_NEUTRAL_BAND:
        label = "negative"
    else:
        label = "neutral"

    return round(float(normalized), 3), label


@dataclass
class NewsItem:
    """Single news article with auto-computed sentiment."""

    title: str
    content: str = ""
    source: str = ""
    url: str = ""
    publish_time: datetime = field(default_factory=datetime.now)
    stock_codes: list[str] = field(default_factory=list)
    category: str = ""  # policy, earnings, market, industry, company
    sentiment_score: float = 0.0  # -1.0 .. +1.0
    sentiment_label: str = "neutral"  # positive, negative, neutral
    importance: float = 0.5  # 0.0 .. 1.0
    keywords: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Only auto-compute sentiment if not already set and text exists
        if self.sentiment_score == 0.0 and (self.title or self.content):
            combined = (self.title or "") + " " + (self.content or "")
            self.sentiment_score, self.sentiment_label = analyze_sentiment(
                combined.strip()
            )

    def age_minutes(self) -> float:
        age_s = _safe_age_seconds_from_now(self.publish_time)
        if age_s is None:
            return 0.0
        return max(0.0, float(age_s) / 60.0)

    def is_relevant_to(self, stock_code: str) -> bool:
        return stock_code in self.stock_codes

    def to_dict(self) -> dict:
        ts = self.publish_time if isinstance(self.publish_time, datetime) else None
        return {
            "title": self.title,
            "source": self.source,
            "time": ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "",
            "sentiment": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "importance": self.importance,
            "category": self.category,
            "codes": self.stock_codes,
        }


class _BaseNewsFetcher:
    """Shared session setup for news fetchers."""

    _JSONP_WRAPPER_RE = re.compile(
        r"^[\w\.\$]+\((?P<body>[\s\S]*)\)\s*;?\s*$",
        flags=re.IGNORECASE,
    )
    _ASSIGNMENT_WRAPPER_RE = re.compile(
        r"^[\w\.\$]+\s*=\s*(?P<body>[\s\S]*)\s*;?\s*$",
        flags=re.IGNORECASE,
    )

    @staticmethod
    def _allow_insecure_tls() -> bool:
        raw = str(os.environ.get("TRADING_NEWS_ALLOW_INSECURE_TLS", "0"))
        return raw.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _resolve_tls_verify() -> bool | str:
        """Resolve a usable CA bundle path for requests.

        Some environments can have a stale certifi path; in that case we first
        try system/default bundles. If no valid bundle path can be found,
        remain secure-by-default (verify=True) unless explicitly overridden.
        """
        candidates: list[str] = []
        try:
            import certifi

            p = str(certifi.where() or "").strip()
            if p:
                candidates.append(p)
        except Exception:
            pass

        try:
            from requests.utils import DEFAULT_CA_BUNDLE_PATH

            p = str(DEFAULT_CA_BUNDLE_PATH or "").strip()
            if p:
                candidates.append(p)
        except Exception:
            pass

        try:
            dflt = ssl.get_default_verify_paths()
            for p in (dflt.cafile, dflt.capath):
                s = str(p or "").strip()
                if s:
                    candidates.append(s)
        except Exception:
            pass

        seen: set[str] = set()
        for raw in candidates:
            if raw in seen:
                continue
            seen.add(raw)
            try:
                if Path(raw).exists():
                    return raw
            except OSError:
                continue

        if _BaseNewsFetcher._allow_insecure_tls():
            return False
        # Keep TLS verification enabled by default.
        return True

    def __init__(self, referer: str = ""):
        self._session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            ),
        }
        if referer:
            headers["Referer"] = referer
        self._session.headers.update(headers)
        verify_cfg = self._resolve_tls_verify()
        self._session.verify = verify_cfg
        if verify_cfg is False:
            log.warning(
                "News TLS verification disabled by env "
                "(TRADING_NEWS_ALLOW_INSECURE_TLS=1)"
            )
        
        # Rate limiting state
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """FIX Bug 6: Add rate limiting to avoid API throttling.
        Enforces minimum delay between consecutive requests.
        """
        import time
        
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get_with_retry(
        self,
        url: str,
        params: dict = None,
        timeout: int = None,
        retries: int = None,
    ) -> requests.Response:
        """FIX Bug 4: Add retry logic with exponential backoff for failed requests.
        
        Args:
            url: URL to fetch
            params: Query parameters
            timeout: Request timeout (uses _FETCH_TIMEOUT if not specified)
            retries: Number of retry attempts (uses _MAX_RETRIES if not specified)
        
        Returns:
            requests.Response object
        
        Raises:
            requests.RequestException: If all retries fail
        """
        import time
        
        timeout = timeout or _FETCH_TIMEOUT
        retries = retries if retries is not None else _MAX_RETRIES
        
        last_exception: Exception | None = None
        
        for attempt in range(retries + 1):
            try:
                self._rate_limit()
                response = self._session.get(
                    url,
                    params=params,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                log.debug(f"Request timeout (attempt {attempt + 1}/{retries + 1}): {url}")
                
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                log.debug(f"Connection error (attempt {attempt + 1}/{retries + 1}): {url}")
                
            except requests.exceptions.HTTPError as e:
                last_exception = e
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    log.debug(f"HTTP client error (no retry): {url} - {e}")
                    raise
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                log.debug(f"Request failed (attempt {attempt + 1}/{retries + 1}): {url}")
            
            # Exponential backoff before retry
            if attempt < retries:
                delay = _RETRY_DELAY_BASE * (2 ** attempt)
                log.debug(f"Retrying in {delay:.2f}s...")
                time.sleep(delay)
        
        # All retries exhausted
        raise last_exception or requests.exceptions.RequestException("Unknown error")

    @staticmethod
    def _clean_text(value: object) -> str:
        """Strip tags/entities and normalize whitespace."""
        text = str(value or "")
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        """Best-effort datetime parser supporting epoch and common formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value

        raw = str(value).strip()
        if not raw:
            return None

        if raw.isdigit():
            try:
                iv = int(raw)
                if iv > 2_000_000_000_000:
                    return datetime.fromtimestamp(iv / 1000.0)
                if iv > 0:
                    return datetime.fromtimestamp(iv)
            except (ValueError, OSError):
                pass

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(raw[:19], fmt)
            except ValueError:
                continue
        return None

    @classmethod
    def _decode_json_payload(cls, payload: object) -> object | None:
        """Parse direct JSON, JSONP, and assignment wrappers."""
        text = str(payload or "").strip()
        if not text:
            return None

        candidates: list[str] = [text]

        m_jsonp = cls._JSONP_WRAPPER_RE.match(text)
        if m_jsonp:
            body = str(m_jsonp.group("body") or "").strip()
            if body:
                candidates.append(body)

        m_assign = cls._ASSIGNMENT_WRAPPER_RE.match(text)
        if m_assign:
            body = str(m_assign.group("body") or "").strip()
            if body:
                candidates.append(body)

        for left_char, right_char in (("{", "}"), ("[", "]")):
            left = text.find(left_char)
            right = text.rfind(right_char)
            if left >= 0 and right > left:
                candidates.append(text[left : right + 1])

        seen: set[str] = set()
        for candidate in candidates:
            chunk = str(candidate or "").strip().lstrip("\ufeff")
            if not chunk or chunk in seen:
                continue
            seen.add(chunk)
            try:
                return json.loads(chunk)
            except Exception:
                continue
        return None

    def _response_to_json(self, response: object) -> object | None:
        """Best-effort JSON decode for provider responses."""
        try:
            return response.json()  # type: ignore[attr-defined]
        except Exception:
            pass
        raw_text = str(getattr(response, "text", "") or "")
        return self._decode_json_payload(raw_text)

    def _extract_titles_from_html(
        self,
        html_text: str,
        *,
        max_items: int,
    ) -> list[str]:
        """Extract title candidates from mixed HTML/script payloads."""
        text = str(html_text or "")
        if not text:
            return []

        patterns = (
            r"<h[1-4][^>]*>\s*<a[^>]*>(.*?)</a>\s*</h[1-4]>",
            r"<a[^>]*class=[\"'][^\"']*(?:title|news|headline)[^\"']*[\"'][^>]*>(.*?)</a>",
            r"\"title\"\s*:\s*\"([^\"]{4,240})\"",
            r"'title'\s*:\s*'([^']{4,240})'",
        )
        titles: list[str] = []
        seen: set[str] = set()
        for pat in patterns:
            for raw in re.findall(pat, text, flags=re.IGNORECASE | re.DOTALL):
                title = self._clean_text(raw)
                if len(title) < 4:
                    continue
                key = title.lower()
                if key in seen:
                    continue
                seen.add(key)
                titles.append(title)
                if len(titles) >= int(max_items):
                    return titles
        return titles


class SinaNewsFetcher(_BaseNewsFetcher):
    """Fetch news from Sina Finance (works on China IP)."""

    def __init__(self) -> None:
        super().__init__(referer="https://finance.sina.com.cn/")

    def fetch_market_news(self, count: int = 20) -> list[NewsItem]:
        """Fetch general market news."""
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            params = {
                "pageid": "153",
                "lid": "2516",
                "k": "",
                "num": str(min(count, 50)),
                "page": "1",
            }
            # FIX Bug 4: Use retry helper
            r = self._get_with_retry(url, params=params)
            data = self._response_to_json(r)
            if not isinstance(data, dict):
                return []

            items: list[NewsItem] = []
            for article in data.get("result", {}).get("data", []):
                if not isinstance(article, dict):
                    continue

                title = self._clean_text(article.get("title", ""))
                if not title:
                    continue

                pub_time = datetime.now()
                parsed_time = self._parse_datetime(article.get("ctime"))
                if parsed_time is not None:
                    pub_time = parsed_time

                items.append(NewsItem(
                    title=title,
                    content=(
                        self._clean_text(article.get("summary", ""))
                        or self._clean_text(article.get("intro", ""))
                    ),
                    source="sina",
                    url=str(article.get("url", "") or ""),
                    publish_time=pub_time,
                    category="market",
                ))

            log.debug(f"Sina: fetched {len(items)} market news")
            return items

        except Exception as exc:
            log.warning(f"Sina market news failed: {exc}")
            return []

    def fetch_stock_news(
        self, stock_code: str, count: int = 10
    ) -> list[NewsItem]:
        """Fetch news for a specific stock via Sina API.
        
        FIX Bug 1: Use proper Sina finance API endpoint instead of 
        search.sina.com.cn which returns HTML not JSON.
        """
        try:
            code6 = str(stock_code).zfill(6)
            # FIX: Use Sina's actual finance API for stock news
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            
            # Map stock code to Sina's market ID
            # 600xxx, 601xxx, 688xxx = SSE (sh), 000xxx, 002xxx, 300xxx = SZSE (sz)
            if code6.startswith(("600", "601", "603", "605", "688")):
                market_id = "2516"  # SSE stocks
                symbol_param = f"sh{code6}"
            elif code6.startswith(("000", "001", "002", "003", "300", "301")):
                market_id = "2520"  # SZSE stocks
                symbol_param = f"sz{code6}"
            else:
                # BSE or unknown - use general market news
                market_id = "2516"
                symbol_param = code6
            
            params = {
                "pageid": "153",
                "lid": market_id,
                "k": symbol_param,  # Stock symbol filter
                "num": str(min(count, 50)),
                "page": "1",
            }
            
            # FIX Bug 4: Use retry helper
            r = self._get_with_retry(url, params=params)
            data = self._response_to_json(r)
            
            items: list[NewsItem] = []
            seen_titles: set[str] = set()
            
            if not isinstance(data, dict):
                # Fallback: try HTML extraction as last resort
                pass
            else:
                # Sina API may return 'data' or 'list' depending on endpoint
                result = data.get("result", {})
                articles = result.get("data") or result.get("list", [])
                for article in articles:
                    if not isinstance(article, dict):
                        continue
                    
                    title = self._clean_text(article.get("title", ""))
                    if not title:
                        continue
                    
                    key = title.lower()
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    
                    pub_time = datetime.now()
                    parsed_time = self._parse_datetime(article.get("ctime"))
                    if parsed_time is not None:
                        pub_time = parsed_time
                    
                    items.append(NewsItem(
                        title=title,
                        content=(
                            self._clean_text(article.get("summary", ""))
                            or self._clean_text(article.get("intro", ""))
                        ),
                        source="sina",
                        url=str(article.get("url", "") or ""),
                        publish_time=pub_time,
                        stock_codes=[code6],
                        category="company",
                    ))
                    
                    if len(items) >= int(count):
                        return items[:count]
            
            # Fallback: If no news found, try general market news filtered by keyword
            if len(items) < int(count):
                market_news = self.fetch_market_news(count=30)
                for news_item in market_news:
                    if len(items) >= int(count):
                        break
                    # Check if title/content mentions the stock code
                    if (code6 in news_item.title or 
                        code6 in news_item.content):
                        news_item.stock_codes = [code6]
                        items.append(news_item)
            
            return items[:count]

        except Exception as exc:
            log.debug(f"Sina stock news failed for {stock_code}: {exc}")
            return []


class EastmoneyNewsFetcher(_BaseNewsFetcher):
    """Fetch news from Eastmoney (works on China IP only)."""

    def __init__(self) -> None:
        super().__init__(referer="https://www.eastmoney.com/")

    def fetch_stock_news(
        self, stock_code: str, count: int = 10
    ) -> list[NewsItem]:
        """Fetch stock-specific news and announcements from Eastmoney.
        
        FIX Bug 2: Use proper JSONP handling with multiple callback patterns
        and improved error recovery.
        """
        try:
            code6 = str(stock_code).zfill(6)
            url = "https://search-api-web.eastmoney.com/search/jsonp"
            
            # FIX: Try multiple callback names for better compatibility
            callbacks = ["jQuery", "callback", "eastmoney_callback"]
            
            items: list[NewsItem] = []
            seen_titles: set[str] = set()
            
            for cb_name in callbacks:
                if items:  # Already got results
                    break
                    
                params = {
                    "cb": cb_name,
                    "param": json.dumps({
                        "uid": "",
                        "keyword": code6,
                        "type": ["cmsArticleWebOld", "cmsArticleWeb"],
                        "client": "web",
                        "clientType": "web",
                        "clientVersion": "curr",
                        "param": {
                            "cmsArticleWebOld": {
                                "searchScope": "default",
                                "sort": "default",
                                "pageIndex": 1,
                                "pageSize": count,
                            },
                            "cmsArticleWeb": {
                                "searchScope": "default",
                                "sort": "default",
                                "pageIndex": 1,
                                "pageSize": count,
                            },
                        },
                    }),
                }

                # FIX Bug 4: Use retry helper
                r = self._get_with_retry(url, params=params)
                data = self._response_to_json(r)
                
                if not isinstance(data, dict):
                    continue
                
                # Try multiple response paths
                articles: list[dict] = []
                candidate_paths = [
                    lambda d: d.get("result", {}).get("cmsArticleWebOld", {}).get("list", []),
                    lambda d: d.get("result", {}).get("cmsArticleWeb", {}).get("list", []),
                    lambda d: d.get("result", {}).get("news", {}).get("list", []),
                    lambda d: d.get("data", {}).get("list", []),
                    lambda d: d.get("list", []),
                ]
                
                for path_fn in candidate_paths:
                    try:
                        result = path_fn(data)
                        if isinstance(result, list) and result:
                            articles = [row for row in result if isinstance(row, dict)]
                            if articles:
                                break
                    except Exception:
                        continue
                
                if not articles:
                    continue
                
                for article in articles:
                    title = self._clean_text(
                        article.get("title")
                        or article.get("name")
                        or article.get("headline")
                        or ""
                    )
                    if not title:
                        continue
                    
                    key = title.lower()
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)

                    pub_time = self._parse_datetime(
                        article.get("date")
                        or article.get("publish_time")
                        or article.get("showTime")
                        or article.get("ctime")
                        or article.get("showtime")
                    ) or datetime.now()

                    content = self._clean_text(
                        article.get("content", "")
                        or article.get("summary", "")
                        or article.get("mediaName", "")
                        or ""
                    )

                    items.append(NewsItem(
                        title=title,
                        content=content[:500],
                        source="eastmoney",
                        url=str(article.get("url") or article.get("link") or ""),
                        publish_time=pub_time,
                        stock_codes=[code6],
                        category="company",
                    ))
                    
                    if len(items) >= int(count):
                        break
            
            # Fallback: HTML title extraction if all JSONP attempts failed
            if len(items) < int(count):
                fallback_titles = self._extract_titles_from_html(
                    str(getattr(r, "text", "") or ""),
                    max_items=max(6, int(count) * 3),
                )
                seen = {str(it.title).strip().lower() for it in items}
                for title in fallback_titles:
                    key = title.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(NewsItem(
                        title=title,
                        source="eastmoney",
                        stock_codes=[code6],
                        category="company",
                    ))
                    if len(items) >= int(count):
                        break

            log.debug(f"Eastmoney: fetched {len(items)} news for {code6}")
            return items[:count]

        except Exception as exc:
            log.debug(f"Eastmoney news failed for {stock_code}: {exc}")
            return []

    def fetch_policy_news(self, count: int = 15) -> list[NewsItem]:
        """Fetch policy and regulatory news."""
        try:
            url = "https://np-listapi.eastmoney.com/comm/web/getNewsByColumns"
            params = {
                "columns": "CSRC,PBOC,MOF",
                "pageSize": str(min(count, 30)),
                "pageIndex": "1",
            }
            # FIX Bug 4: Use retry helper
            r = self._get_with_retry(url, params=params)
            data = self._response_to_json(r)
            if not isinstance(data, dict):
                return []

            items: list[NewsItem] = []
            for article in data.get("data", {}).get("list", []):
                if not isinstance(article, dict):
                    continue
                title = article.get("title", "").strip()
                if not title:
                    continue
                items.append(NewsItem(
                    title=self._clean_text(title),
                    source="eastmoney_policy",
                    publish_time=(
                        self._parse_datetime(
                            article.get("showtime")
                            or article.get("date")
                            or article.get("ctime")
                        ) or datetime.now()
                    ),
                    category="policy",
                    importance=0.8,
                ))

            return items

        except Exception as exc:
            log.debug(f"Eastmoney policy news failed: {exc}")
            return []


class TencentNewsFetcher(_BaseNewsFetcher):
    """Fetch news from Tencent Finance (works from ANY IP)."""

    def __init__(self) -> None:
        super().__init__()

    def fetch_market_news(self, count: int = 20) -> list[NewsItem]:
        """Fetch general financial news via Tencent."""
        try:
            url = "https://r.inews.qq.com/getSimpleNews"
            params = {
                "ids": "finance_hot",
                "num": str(min(count, 30)),
            }
            # FIX Bug 4: Use retry helper
            r = self._get_with_retry(url, params=params)
            data = self._response_to_json(r)
            if not isinstance(data, dict):
                return []

            items: list[NewsItem] = []
            for article in data.get("newslist", []):
                if not isinstance(article, dict):
                    continue
                title = self._clean_text(article.get("title", ""))
                if not title:
                    continue

                pub_time = datetime.now()
                parsed_time = self._parse_datetime(
                    article.get("timestamp")
                    or article.get("ctime")
                    or article.get("publish_time")
                )
                if parsed_time is not None:
                    pub_time = parsed_time

                items.append(NewsItem(
                    title=title,
                    content=self._clean_text(article.get("abstract", "") or ""),
                    source="tencent",
                    url=str(article.get("url", "") or ""),
                    publish_time=pub_time,
                    category="market",
                ))

            log.debug(f"Tencent: fetched {len(items)} market news")
            return items

        except Exception as exc:
            log.warning(f"Tencent market news failed: {exc}")
            return []


_POLICY_KEYWORDS: tuple[str, ...] = (
    "\u592e\u884c",
    "\u8bc1\u76d1\u4f1a",
    "\u8d22\u653f\u90e8",
    "\u56fd\u52a1\u9662",
    "\u653f\u7b56",
    "\u76d1\u7ba1",
    "\u6539\u9769",
    "\u6cd5\u89c4",
)


def _make_dedup_key(item: NewsItem) -> tuple[str, str]:
    """Create a stable dedup key from title prefix and publish_time.

    FIX Bug 1: Original code referenced item.published_at which does not
    exist on NewsItem. The correct attribute is item.publish_time.
    """
    title_part = str(item.title or "").strip()[:_DEDUP_PREFIX_LEN]
    if isinstance(item.publish_time, datetime):
        time_part = item.publish_time.strftime("%Y-%m-%d %H:%M")
    else:
        time_part = ""
    return (title_part, time_part)


# Lazy import for NewsAggregator to avoid circular dependency
def __getattr__(name: str):
    """Lazy import for NewsAggregator."""
    if name == "NewsAggregator":
        from data.news_aggregator import NewsAggregator as _NewsAggregator
        return _NewsAggregator
    if name == "get_news_aggregator":
        from data.news_aggregator import get_news_aggregator as _get_news_aggregator
        return _get_news_aggregator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



