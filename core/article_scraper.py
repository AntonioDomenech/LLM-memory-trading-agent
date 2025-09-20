
import re, html, time
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
import requests

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

def fetch_fulltext(url: str, timeout: int = 12, ua: str = None, max_len: int = 20000) -> str:
    """Best-effort article text extraction with stdlib only.
       Returns plain text, or "" if not parseable.
    """
    if not url:
        return ""
    headers = {"User-Agent": ua or DEFAULT_UA, "Accept":"text/html,application/xhtml+xml"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    except Exception:
        return ""
    ctype = (r.headers.get("Content-Type","").split(";")[0] or "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        return ""
    html_str = r.text
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
        return ""
    txt = txt.strip()
    if len(txt) > max_len:
        txt = txt[:max_len] + " ..."
    return txt
