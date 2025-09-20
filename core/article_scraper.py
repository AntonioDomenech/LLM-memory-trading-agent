
import re, html, time
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
import requests
from typing import Dict, Tuple, Any, Optional, Union
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

BLOCK_TAGS = {"script","style","noscript","svg","canvas","footer","nav","aside","form","iframe","amp-auto-ads"}
PRIORITY_IDS = {"article","main","content","story","post","entry","read"}
PRIORITY_CLASSES = {"article","content","story","post","entry","article-body","post-content","main-content"}

class _TextCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._stack = []
        self._bufs = []
        self._in_block = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self._stack.append(tag)
        if tag in BLOCK_TAGS:
            self._in_block = True

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._stack:
            self._stack.pop()
        if tag in BLOCK_TAGS:
            self._in_block = False
        if tag in ("p","br","div","li","h1","h2","h3","h4"):
            self._bufs.append("\n")

    def handle_data(self, data):
        if self._in_block:
            return
        if not data or not data.strip():
            return
        self._bufs.append(data)

    def text(self):
        txt = "".join(self._bufs)
        # Collapse whitespace
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        delay = float(value)
        if delay >= 0:
            return delay
    except (TypeError, ValueError):
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delay = (dt - now).total_seconds()
        return delay if delay > 0 else None
    except Exception:
        return None


def fetch_fulltext(url: str, timeout: int = 12, ua: str = None, max_len: int = 20000,
                   *, return_meta: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """Best-effort article text extraction with stdlib only.
       Returns plain text by default for backward compatibility.
       When return_meta=True, returns (plain text, metadata) where metadata contains
       status / retry info so callers can distinguish failures.
    """
    meta: Dict[str, Any] = {
        "status": None,
        "error": None,
        "retryable": False,
        "attempts": 0,
        "retry_after": None,
    }
    if not url:
        meta.update({"error": "no_url", "retryable": False})
        return ("", meta) if return_meta else ""
    headers = {"User-Agent": ua or DEFAULT_UA, "Accept":"text/html,application/xhtml+xml"}
    backoff = 1.0
    last_exc: Optional[BaseException] = None
    response = None
    for attempt in range(3):
        meta["attempts"] = attempt + 1
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        except requests.RequestException as exc:
            last_exc = exc
            meta.update({"status": None, "error": f"request:{type(exc).__name__}", "retryable": True})
        except Exception as exc:
            last_exc = exc
            meta.update({"status": None, "error": f"exception:{type(exc).__name__}", "retryable": False})
            break
        else:
            status = getattr(response, "status_code", None)
            meta["status"] = status
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            if retry_after is not None:
                meta["retry_after"] = retry_after
            if status != 200:
                meta.update({
                    "error": f"http:{status}",
                    "retryable": bool(status in RETRYABLE_STATUS)
                })
                if meta.get("retryable") and attempt < 2:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                break
            break
        if attempt < 2:
            time.sleep(backoff)
            backoff *= 2
    else:
        response = None

    if response is None:
        if last_exc and meta.get("error") is None:
            meta.update({"error": f"exception:{type(last_exc).__name__}", "retryable": True})
        return ("", meta) if return_meta else ""

    status = response.status_code
    if status != 200:
        return ("", meta) if return_meta else ""

    ctype = (response.headers.get("Content-Type","").split(";")[0] or "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        meta.update({"error": "unsupported_content_type", "retryable": False})
        return ("", meta) if return_meta else ""

    html_str = response.text
    # Heuristic pre-trim: drop nav/header/footer blocks crudely
    html_str = re.sub(r"<(nav|footer|aside|script|style|noscript)[\\s\\S]*?</\\1>", " ", html_str, flags=re.I)
    # Simple parse
    p = _TextCollector()
    try:
        p.feed(html_str)
    except Exception:
        # Some pages break the parser; still try to emit what we have
        pass
    txt = p.text()
    if not txt:
        meta.update({"error": "empty_text", "retryable": False})
        return ("", meta) if return_meta else ""
    txt = txt.strip()
    if len(txt) > max_len:
        txt = txt[:max_len] + " ..."
    meta.update({"error": None, "retryable": False})
    return (txt, meta) if return_meta else txt
