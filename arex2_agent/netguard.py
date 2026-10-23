# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Fetch a web page for the agent without letting the model, or a page it reads, reach the machine's own network.

The model chooses the URLs, and a page it reads can name more. This app runs on a LAN box beside services that must stay unreachable
from it (the model server itself, ollama, a metadata address, the operator's other apps). The rules:

* only http and https, no login in the URL, only the ports 80, 443, 8080 and 8443;
* the name is resolved here, every address it resolves to must be a public one, and the connection goes to the address that was
  checked (the name is not looked up a second time), so a DNS answer that changes between the check and the connection gains nothing;
* each redirect is checked again from the start, with the same rules;
* no cookies and no credentials are sent or kept; the body is read to a size cap and the whole fetch has a time limit;
* only text-like content types are read (HTML becomes plain text, scripts and styles dropped).
Standard library only. A page's text is untrusted: the caller must never treat it as an instruction or put it into the page as HTML.
"""
from __future__ import annotations

import codecs
import http.client
import ipaddress
import re
import socket
import ssl
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

ALLOWED_PORTS = (80, 443, 8080, 8443)
USER_AGENT = "tt-arex2-agent/0.1 (research demo; read-only)"
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml+xml", "application/atom+xml", "application/rss+xml")
SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "footer", "aside"}
BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote",
              "pre", "dd", "dt", "dl", "main", "header", "form", "figure", "figcaption", "hr"}


class FetchError(Exception):
    """The fetch failed; the message is plain text the model and the person can read."""


class Blocked(FetchError):
    """The address or URL is not one the agent may open."""


@dataclass
class Page:
    url: str
    status: int
    title: str
    text: str
    truncated: bool
    content_type: str


def _public(ip) -> bool:
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    if ip.version == 6 and (ip in ipaddress.ip_network("64:ff9b::/96") or ip in ipaddress.ip_network("2002::/16")
                            or ip in ipaddress.ip_network("2001::/32")):
        return False           # NAT64, 6to4 and Teredo can carry a private IPv4 address inside
    return bool(ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_unspecified or ip.is_loopback or ip.is_link_local))


def check_url(url: str, resolver=socket.getaddrinfo):
    """Return (address, port) the connection must use, or raise Blocked."""
    try:
        parts = urlsplit(url.strip())
        port_given = parts.port
    except ValueError:
        raise Blocked(f"not a usable web address: {url[:80]!r}") from None
    if parts.scheme not in ("http", "https"):
        raise Blocked(f"only http and https addresses can be opened, not {parts.scheme or 'a bare path'!r}")
    if not parts.hostname:
        raise Blocked("the address has no host name")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise Blocked("addresses with a login in them are not opened")
    port = port_given or (443 if parts.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise Blocked(f"port {port} is not opened (only {', '.join(map(str, ALLOWED_PORTS))})")
    try:
        infos = resolver(parts.hostname, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as e:
        raise Blocked(f"the name {parts.hostname!r} does not resolve ({e})") from None
    addrs = []
    for info in infos:
        raw = str(info[4][0]).split("%")[0]
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            raise Blocked(f"the name {parts.hostname!r} resolved to {raw!r}, which is not an address") from None
        if not _public(ip):
            raise Blocked(f"{parts.hostname!r} resolves to {raw}, which is not a public address (private, loopback, link-local and reserved "
                          "addresses are not opened)")
        addrs.append(raw)
    if not addrs:
        raise Blocked(f"the name {parts.hostname!r} has no address")
    return addrs[0], port


class _Pinned(http.client.HTTPConnection):
    def __init__(self, ip, port, hostname, timeout):
        super().__init__(hostname, port, timeout=timeout)
        self._ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)


class _PinnedTLS(http.client.HTTPSConnection):
    def __init__(self, ip, port, hostname, timeout):
        super().__init__(hostname, port, timeout=timeout, context=ssl.create_default_context())
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def default_connector(ip, port, scheme, hostname, timeout):
    return (_PinnedTLS if scheme == "https" else _Pinned)(ip, port, hostname, timeout)


class SafeFetcher:
    def __init__(self, resolver=socket.getaddrinfo, connector=default_connector, max_bytes: int = 1_000_000, timeout_s: float = 20.0,
                 max_redirects: int = 5):
        self.resolver, self.connector = resolver, connector
        self.max_bytes, self.timeout_s, self.max_redirects = max_bytes, timeout_s, max_redirects

    def fetch(self, url: str) -> Page:
        deadline = time.monotonic() + self.timeout_s
        current = url
        for hop in range(self.max_redirects + 1):
            ip, port = check_url(current, self.resolver)
            parts = urlsplit(current)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FetchError(f"the fetch took too long (over {self.timeout_s:g} s)")
            conn = self.connector(ip, port, parts.scheme, parts.hostname, remaining)
            try:
                path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
                host_header = parts.hostname if port in (80, 443) and parts.port is None else f"{parts.hostname}:{port}"
                conn.request("GET", path, headers={"Host": host_header, "User-Agent": USER_AGENT, "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1",
                                                   "Accept-Encoding": "identity", "Connection": "close"})
                resp = conn.getresponse()
                if 300 <= resp.status < 400:
                    loc = resp.getheader("Location")
                    resp.read(0)
                    if not loc:
                        raise FetchError(f"the server answered {resp.status} with no Location to follow")
                    current = urljoin(current, loc)
                    continue
                if resp.status >= 400:
                    raise FetchError(f"the server answered HTTP {resp.status} {resp.reason}")
                ctype = (resp.getheader("Content-Type") or "text/plain").split(";")[0].strip().lower()
                if not (ctype.startswith(TEXT_TYPES) or ctype.endswith(("+json", "+xml"))):
                    raise FetchError(f"the page is {ctype}, which is not text the agent can read")
                body, truncated = self._read(resp, deadline)
                return self._page(current, resp.status, ctype, resp.getheader("Content-Type") or "", body, truncated)
            except (socket.timeout, TimeoutError):
                raise FetchError(f"the fetch took too long (over {self.timeout_s:g} s)") from None
            except (OSError, http.client.HTTPException) as e:
                raise FetchError(f"the fetch failed: {e}") from None
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        raise FetchError(f"too many redirects (more than {self.max_redirects})")

    def _read(self, resp, deadline):
        chunks, total = [], 0
        while total <= self.max_bytes:
            if time.monotonic() > deadline:
                raise FetchError(f"the fetch took too long (over {self.timeout_s:g} s)")
            part = resp.read1(min(16384, self.max_bytes + 1 - total))     # read1 returns what has arrived, so the deadline is checked between pieces
            if not part:
                break
            chunks.append(part)
            total += len(part)
        data = b"".join(chunks)
        return data[: self.max_bytes], len(data) > self.max_bytes

    @staticmethod
    def _page(url, status, ctype, raw_ctype, body, truncated) -> Page:
        m = re.search(r"charset=([\w.-]+)", raw_ctype, re.I)
        enc = "utf-8"
        if m:
            try:
                enc = codecs.lookup(m.group(1)).name
            except LookupError:
                enc = "utf-8"
        text = body.decode(enc, "replace")
        if ctype in ("text/html", "application/xhtml+xml"):
            title, text = html_to_text(text)
        else:
            title = ""
        return Page(url=url, status=status, title=title, text=text, truncated=truncated, content_type=ctype)


class _Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self._in_title, self._skip = "", False, 0
        self._blocks, self._cur = [], []

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._cur)).strip()
        if text:
            self._blocks.append(text)
        self._cur = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in SKIP_TAGS:
            self._skip += 1
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._cur.append(data)


def html_to_text(markup: str):
    p = _Extract()
    try:
        p.feed(markup)
        p.close()
    except Exception:          # malformed markup: keep what was read
        pass
    p._flush()
    return re.sub(r"\s+", " ", p.title).strip(), "\n".join(p._blocks)
