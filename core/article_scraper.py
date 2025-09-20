import re, html, time
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
import requests
from typing import Dict, Tuple, Any, Optional, Union
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from readability import Document

DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

BLOCK_TAGS = {"script", "style", "noscript", "svg", "canvas", "footer", "nav", "aside", "form", "iframe", "amp-auto-ads"}
PRIORITY_IDS = {"article", "main", "content", "story", "post", "entry", "read"}
PRIORITY_CLASSES = {"article", "content", "story", "post", "entry", "article-body", "post-content", "main-content"}


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
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4"):
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
    """Best-effort article text extraction with readability fallback and diagnostics."""

    meta: Dict[str, Any] = {
        "status": None,
        "error": None,
        "retryable": False,
        "attempts": 0,
        "retry_after": None,
        "phase": "pre",
    }

    def _result(text: str) -> Union[str, Tuple[str, Dict[str, Any]]]:
        if return_meta:
            return text, meta
        return text

    if not url:
        meta.update({"error": "no_url", "retryable": False, "phase": "pre"})
        return _result("")

    headers = {"User-Agent": ua or DEFAULT_UA, "Accept": "text/html,application/xhtml+xml"}
    backoff = 1.0
    last_exc: Optional[BaseException] = None
    response = None

    for attempt in range(3):
        meta["attempts"] = attempt + 1
        start_ts = time.time()
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        except requests.RequestException as exc:
            elapsed = int((time.time() - start_ts) * 1000)
            last_exc = exc
            meta.update({
                "status": None,
                "error": f"request:{type(exc).__name__}",
                "retryable": True,
                "phase": "request",
                "elapsed_ms": elapsed,
                "message": str(exc) or None,
            })
        except Exception as exc:
            elapsed = int((time.time() - start_ts) * 1000)
            last_exc = exc
            meta.update({
                "status": None,
                "error": f"exception:{type(exc).__name__}",
                "retryable": False,
                "phase": "exception",
                "elapsed_ms": elapsed,
                "message": str(exc) or None,
            })
            break
        else:
            elapsed = int((time.time() - start_ts) * 1000)
            status = getattr(response, "status_code", None)
            meta.update({
                "status": status,
                "phase": "request",
                "elapsed_ms": elapsed,
            })
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            if retry_after is not None:
                meta["retry_after"] = retry_after
            if status != 200:
                meta.update({
                    "error": f"http:{status}",
                    "retryable": bool(status in RETRYABLE_STATUS),
                    "phase": "http_error",
                })
                if meta["retryable"] and attempt < 2:
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
            meta.update({
                "error": f"exception:{type(last_exc).__name__}",
                "retryable": True,
                "phase": "exception",
            })
        return _result("")

    status = response.status_code
    if status != 200:
        return _result("")

    ctype = (response.headers.get("Content-Type", "").split(";")[0] or "").lower()
    meta["content_type"] = ctype
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        meta.update({"error": "unsupported_content_type", "retryable": False, "phase": "content_type"})
        return _result("")

    html_str = response.text
    # Heuristic pre-trim: drop nav/header/footer blocks crudely
    html_str = re.sub(r"<(nav|footer|aside|script|style|noscript)[\\s\\S]*?</\\1>", " ", html_str, flags=re.I)

    best_txt = ""
    for region_html in _iter_candidate_regions(html_str):
        candidate_txt = _collect_text(region_html)
        if not candidate_txt:
            continue
        if len(candidate_txt) >= 400 or candidate_txt.count("\n") >= 3:
            best_txt = candidate_txt
            meta["extraction"] = "candidate_region"
            break
        if len(candidate_txt) > len(best_txt):
            best_txt = candidate_txt

    if not best_txt:
        readability_txt = _readability_fallback(html_str)
        if readability_txt:
            best_txt = readability_txt
            meta["extraction"] = "readability"

    parse_error: Optional[BaseException] = None
    if not best_txt:
        parser = _TextCollector()
        try:
            parser.feed(html_str)
            parser.close()
        except Exception as exc:
            parse_error = exc
        best_txt = (parser.text() or "").strip()
        if parse_error:
            meta["parser_error"] = f"{type(parse_error).__name__}: {parse_error}"
        if best_txt:
            meta["extraction"] = "fallback_parser"

    if not best_txt:
        meta.update({"error": "empty_text", "retryable": False, "phase": "extract"})
        return _result("")

    best_txt = best_txt.strip()
    if len(best_txt) > max_len:
        best_txt = best_txt[:max_len] + " ..."
        meta["truncated"] = True

    meta.update({
        "error": None,
        "retryable": False,
        "phase": "success",
        "length": len(best_txt),
    })
    if parse_error and "parser_error" not in meta:
        meta["parser_warning"] = f"{type(parse_error).__name__}: {parse_error}"

    return _result(best_txt)
