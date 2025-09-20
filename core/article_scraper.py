
import re, html, time
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
import requests
from readability import Document

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


class _CandidateRegionFinder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._stack = []
        self._captures = []  # list of tuples (depth, buffer list)
        self._regions = []
        self._body_buf = None
        self._body_region = ""

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        id_val = attr_map.get("id", "").lower()
        class_val = attr_map.get("class", "").lower()

        should_capture = tag in {"article", "main"}
        if tag in {"section", "div"}:
            if any(key in id_val for key in PRIORITY_IDS) or any(key in class_val for key in PRIORITY_CLASSES):
                should_capture = True

        start_txt = self.get_starttag_text() or f"<{tag}>"
        self._stack.append(tag)

        if tag == "body" and self._body_buf is None:
            self._body_buf = []

        for _, buf in self._captures:
            buf.append(start_txt)
        if self._body_buf is not None:
            self._body_buf.append(start_txt)

        if should_capture:
            buf = [start_txt]
            self._captures.append((len(self._stack), buf))

    def handle_endtag(self, tag):
        tag = tag.lower()
        closing = f"</{tag}>"
        for _, buf in self._captures:
            buf.append(closing)
        if self._body_buf is not None:
            self._body_buf.append(closing)

        for idx in range(len(self._captures) - 1, -1, -1):
            depth, buf = self._captures[idx]
            if depth == len(self._stack):
                self._regions.append("".join(buf))
                self._captures.pop(idx)

        if self._stack:
            self._stack.pop()

        if tag == "body" and self._body_buf is not None:
            self._body_region = "".join(self._body_buf)
            self._body_buf = None

    def handle_data(self, data):
        if not data:
            return
        for _, buf in self._captures:
            buf.append(data)
        if self._body_buf is not None:
            self._body_buf.append(data)

    def handle_startendtag(self, tag, attrs):
        tag = tag.lower()
        start_txt = self.get_starttag_text() or f"<{tag}/>"
        for _, buf in self._captures:
            buf.append(start_txt)
        if self._body_buf is not None:
            self._body_buf.append(start_txt)

    @property
    def regions(self):
        return self._regions

    @property
    def body_region(self):
        return self._body_region


def _iter_candidate_regions(html_str: str):
    finder = _CandidateRegionFinder()
    try:
        finder.feed(html_str)
        finder.close()
    except Exception:
        return []

    seen = set()
    for frag in finder.regions:
        frag = (frag or "").strip()
        if not frag:
            continue
        if frag in seen:
            continue
        seen.add(frag)
        yield frag

    body = (finder.body_region or "").strip()
    if body and body not in seen:
        yield body


def _collect_text(fragment: str) -> str:
    if not fragment:
        return ""
    collector = _TextCollector()
    try:
        collector.feed(fragment)
        collector.close()
    except Exception:
        pass
    return collector.text().strip()


def _readability_fallback(html_str: str) -> str:
    try:
        doc = Document(html_str)
    except Exception:
        return ""
    summary_html = ""
    try:
        summary_html = doc.summary(html_partial=True)
    except TypeError:
        try:
            summary_html = doc.summary()
        except Exception:
            summary_html = ""
    except Exception:
        summary_html = ""
    text = _collect_text(summary_html)
    if text:
        try:
            title = (doc.title() or "").strip()
        except Exception:
            title = ""
        if title and title not in text:
            text = f"{title}\n\n{text}" if text else title
    return text

def fetch_fulltext(url: str, timeout: int = 12, ua: str = None, max_len: int = 20000) -> str:
    """Best-effort article text extraction with readability fallback.
       Returns plain text, or "" if not parseable."""
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
    best_txt = ""
    for region_html in _iter_candidate_regions(html_str):
        candidate_txt = _collect_text(region_html)
        if not candidate_txt:
            continue
        if len(candidate_txt) >= 400 or candidate_txt.count("\n") >= 3:
            best_txt = candidate_txt
            break
        if len(candidate_txt) > len(best_txt):
            best_txt = candidate_txt

    if not best_txt:
        best_txt = _readability_fallback(html_str)

    if not best_txt:
        best_txt = _collect_text(html_str)

    if not best_txt:
        return ""

    if len(best_txt) > max_len:
        best_txt = best_txt[:max_len] + " ..."
    return best_txt.strip()
