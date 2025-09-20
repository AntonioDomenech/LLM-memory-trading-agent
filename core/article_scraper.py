
import re, html, time
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
import requests
from typing import Dict, Tuple

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

def fetch_fulltext(url: str, timeout: int = 12, ua: str = None, max_len: int = 20000) -> Tuple[str, Dict[str, str]]:
    """Best-effort article text extraction with stdlib only.
       Returns ``(plain_text, diagnostics)`` where ``plain_text`` is empty when extraction fails.
    """
    if not url:
        return "", {"phase": "pre", "error": "empty_url"}

    headers = {"User-Agent": ua or DEFAULT_UA, "Accept": "text/html,application/xhtml+xml"}
    start_ts = time.time()

    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    except Exception as e:
        elapsed = int((time.time() - start_ts) * 1000)
        return "", {"phase": "request", "error": type(e).__name__, "message": str(e), "elapsed_ms": str(elapsed)}

    elapsed = int((time.time() - start_ts) * 1000)
    diagnostics: Dict[str, str] = {
        "phase": "request",
        "status": str(r.status_code),
        "elapsed_ms": str(elapsed),
    }

    ctype = (r.headers.get("Content-Type", "").split(";")[0] or "").lower()
    diagnostics["content_type"] = ctype

    if r.status_code >= 400:
        diagnostics.update({"phase": "http_error", "error": f"status_{r.status_code}"})
        return "", diagnostics

    if "text/html" not in ctype and "application/xhtml" not in ctype:
        diagnostics.update({"phase": "content_type", "error": "non_html"})
        return "", diagnostics

    html_str = r.text
    # Heuristic pre-trim: drop nav/header/footer blocks crudely
    html_str = re.sub(r"<(nav|footer|aside|script|style|noscript)[\\s\\S]*?</\\1>", " ", html_str, flags=re.I)

    # Simple parse
    p = _TextCollector()
    parse_error = None
    try:
        p.feed(html_str)
    except Exception as e:
        # Some pages break the parser; still try to emit what we have
        parse_error = e

    txt = (p.text() or "").strip()
    if not txt:
        diagnostics.update({"phase": "extract", "error": "empty_text"})
        if parse_error:
            diagnostics["parser_error"] = f"{type(parse_error).__name__}: {parse_error}"
        return "", diagnostics

    if len(txt) > max_len:
        txt = txt[:max_len] + " ..."
        diagnostics["truncated"] = "1"

    diagnostics.update({"phase": "success", "length": str(len(txt))})
    if parse_error:
        diagnostics["parser_warning"] = f"{type(parse_error).__name__}: {parse_error}"

    return txt, diagnostics
