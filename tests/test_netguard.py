# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The fetch guard (arex2_agent/netguard.py): which addresses it refuses, and what it does with a page it may read."""
from __future__ import annotations

import http.client
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arex2_agent import netguard

PUBLIC = "93.184.216.34"


def resolver_to(*ips):
    def resolve(host, port, *a, **k):
        return [(2 if ":" not in ip else 10, 1, 6, "", (ip, port)) for ip in ips]
    return resolve


class Site:
    """A local HTTP server with routes {path: (status, headers, body)}; seen[] keeps each request's headers."""
    def __init__(self, routes):
        self.routes, self.seen = routes, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.seen.append((self.path, dict(self.headers)))
                r = outer.routes.get(self.path, (404, {}, b"nope"))
                if callable(r):
                    r = r()
                status, headers, body = r
                if body == "trickle":      # headers at once, then one byte every 0.1 s for a long time
                    self.send_response(status)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.send_header("Content-Length", "100000")
                    self.end_headers()
                    try:
                        for _ in range(60):
                            self.wfile.write(b"x"); self.wfile.flush(); time.sleep(0.1)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown(); self.server.server_close()


@pytest.fixture
def site():
    made = []
    def make(routes):
        s = Site(routes); made.append(s); return s
    yield make
    for s in made:
        s.close()


def fetcher_for(site, ips=(PUBLIC,), **kw):
    """A fetcher whose DNS says the name is public and whose connector reaches the local test server; connects[] records (ip, port)."""
    connects = []
    def connector(ip, port, scheme, hostname, timeout):
        connects.append((ip, port, scheme, hostname))
        return http.client.HTTPConnection("127.0.0.1", site.port, timeout=timeout)
    f = netguard.SafeFetcher(resolver=resolver_to(*ips), connector=connector, **kw)
    f.connects = connects
    return f


# ---- which URLs are refused -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd", "javascript:alert(1)", "gopher://example.com/", "//example.com/x"])
def test_only_http_and_https_are_accepted(url):
    with pytest.raises(netguard.Blocked) as e:
        netguard.check_url(url, resolver=resolver_to(PUBLIC))
    assert "http" in str(e.value)


def test_a_url_with_a_login_in_it_is_refused():
    with pytest.raises(netguard.Blocked):
        netguard.check_url("http://user:pw@example.com/", resolver=resolver_to(PUBLIC))


def test_a_url_with_no_host_is_refused():
    with pytest.raises(netguard.Blocked):
        netguard.check_url("http:///path", resolver=resolver_to(PUBLIC))


@pytest.mark.parametrize("ip", ["127.0.0.1", "127.1.2.3", "10.0.0.5", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.169.254",
                                "100.64.0.1", "0.0.0.0", "224.0.0.1", "240.0.0.1", "255.255.255.255", "::1", "fe80::1", "fc00::1", "fd12:3456::1",
                                "::", "::ffff:127.0.0.1", "::ffff:10.0.0.1", "ff02::1", "64:ff9b::7f00:1", "64:ff9b::808:808", "2002:7f00:1::1", "2001:0:4136:e378:8000:63bf:3fff:fdd2"])
def test_private_loopback_and_link_local_addresses_are_refused(ip):
    with pytest.raises(netguard.Blocked) as e:
        netguard.check_url("http://example.test/", resolver=resolver_to(ip))
    assert ip.replace("::ffff:", "") in str(e.value) or "private" in str(e.value).lower() or "not a public" in str(e.value).lower()


def test_one_private_address_among_public_ones_is_enough_to_refuse():
    with pytest.raises(netguard.Blocked):
        netguard.check_url("http://example.test/", resolver=resolver_to(PUBLIC, "10.1.1.1"))


@pytest.mark.parametrize("url", ["http://2130706433/", "http://0x7f.0.0.1/", "http://017700000001/", "http://127.1/", "http://[::1]/", "http://localhost/",
                                 "http://0/"])
def test_numeric_and_odd_spellings_of_loopback_are_refused_after_real_resolution(url):
    with pytest.raises(netguard.Blocked):
        netguard.check_url(url)


@pytest.mark.parametrize("port", [22, 25, 3306, 5432, 6379, 8000, 8100, 8101, 11434])
def test_ports_other_than_the_web_ports_are_refused(port):
    with pytest.raises(netguard.Blocked) as e:
        netguard.check_url(f"http://example.test:{port}/", resolver=resolver_to(PUBLIC))
    assert "port" in str(e.value).lower()


@pytest.mark.parametrize("url", ["http://example.test/", "https://example.test/x?y=1", "http://example.test:80/", "https://example.test:443/", "http://example.test:8080/"])
def test_ordinary_web_urls_with_public_addresses_pass(url):
    ip, port = netguard.check_url(url, resolver=resolver_to(PUBLIC))
    assert ip == PUBLIC and port in (80, 443, 8080, 8443)


def test_a_name_that_does_not_resolve_is_a_plain_refusal():
    def nxdomain(host, port, *a, **k):
        import socket
        raise socket.gaierror(-2, "Name or service not known")
    with pytest.raises(netguard.Blocked) as e:
        netguard.check_url("http://nope.invalid/", resolver=nxdomain)
    assert "resolve" in str(e.value).lower()


# ---- what a fetch does -----------------------------------------------------------------------------------------------------

HTML = b"""<html><head><title>Test &amp; Title</title><style>.x{color:red}</style><script>var secret = 1;</script></head>
<body><nav>menu menu</nav><h1>Heading</h1><p>First   paragraph
 with &eacute;ntities.</p><script>alert(2)</script><p>Second.</p></body></html>"""


def test_html_becomes_readable_text_without_scripts_or_styles(site):
    s = site({"/p": (200, {"Content-Type": "text/html; charset=utf-8"}, HTML)})
    page = fetcher_for(s).fetch("http://example.test/p")
    assert page.title == "Test & Title"
    assert "Heading" in page.text and "First paragraph with éntities." in page.text and "Second." in page.text
    assert "secret" not in page.text and "alert" not in page.text and "color:red" not in page.text
    assert page.status == 200 and page.url == "http://example.test/p" and not page.truncated


def test_the_connection_goes_to_the_address_that_was_checked(site):
    s = site({"/p": (200, {"Content-Type": "text/plain"}, b"hello")})
    f = fetcher_for(s)
    f.fetch("http://example.test/p")
    assert f.connects == [(PUBLIC, 80, "http", "example.test")]


def test_the_request_carries_no_cookies_or_credentials(site):
    s = site({"/a": (302, {"Location": "/b", "Set-Cookie": "sid=1"}, b""), "/b": (200, {"Content-Type": "text/plain"}, b"ok")})
    fetcher_for(s).fetch("http://example.test/a")
    for _, headers in s.seen:
        low = {k.lower() for k in headers}
        assert "cookie" not in low and "authorization" not in low
    assert s.seen[0][1].get("Accept-Encoding") == "identity"
    assert "User-Agent" in s.seen[0][1]


def test_a_redirect_is_followed_and_the_final_url_reported(site):
    s = site({"/a": (301, {"Location": "http://example.test/b"}, b""), "/b": (200, {"Content-Type": "text/plain"}, b"landed")})
    page = fetcher_for(s).fetch("http://example.test/a")
    assert page.text == "landed" and page.url == "http://example.test/b"


def test_a_redirect_to_a_private_address_is_refused_even_though_the_first_hop_was_fine(site):
    s = site({"/a": (302, {"Location": "http://internal.test/secret"}, b"")})
    answers = {"example.test": PUBLIC, "internal.test": "10.0.0.7"}
    def resolve(host, port, *a, **k):
        return [(2, 1, 6, "", (answers[host], port))]
    f = netguard.SafeFetcher(resolver=resolve, connector=lambda ip, port, scheme, host, timeout: http.client.HTTPConnection("127.0.0.1", s.port, timeout=timeout))
    with pytest.raises(netguard.Blocked) as e:
        f.fetch("http://example.test/a")
    assert "10.0.0.7" in str(e.value)
    assert len(s.seen) == 1


def test_a_redirect_to_another_scheme_is_refused(site):
    s = site({"/a": (302, {"Location": "file:///etc/passwd"}, b"")})
    with pytest.raises(netguard.Blocked):
        fetcher_for(s).fetch("http://example.test/a")


def test_a_redirect_loop_stops_after_a_few_hops(site):
    s = site({"/a": (302, {"Location": "/a"}, b"")})
    with pytest.raises(netguard.FetchError) as e:
        fetcher_for(s, max_redirects=3).fetch("http://example.test/a")
    assert "redirect" in str(e.value).lower()
    assert len(s.seen) == 4


def test_a_redirect_with_no_location_is_an_error_that_says_so(site):
    s = site({"/a": (302, {}, b"")})
    with pytest.raises(netguard.FetchError) as e:
        fetcher_for(s).fetch("http://example.test/a")
    assert "Location" in str(e.value)


def test_a_big_body_is_cut_at_the_cap_and_flagged(site):
    s = site({"/big": (200, {"Content-Type": "text/plain"}, b"a" * 50000)})
    page = fetcher_for(s, max_bytes=1000).fetch("http://example.test/big")
    assert page.truncated and len(page.text) <= 1000


def test_a_binary_type_is_refused_with_its_type_named(site):
    s = site({"/i": (200, {"Content-Type": "image/png"}, b"\x89PNG")})
    with pytest.raises(netguard.FetchError) as e:
        fetcher_for(s).fetch("http://example.test/i")
    assert "image/png" in str(e.value)


def test_json_and_plain_text_are_returned_as_text(site):
    s = site({"/j": (200, {"Content-Type": "application/json"}, b'{"a": 1}')})
    assert fetcher_for(s).fetch("http://example.test/j").text == '{"a": 1}'


def test_an_http_error_status_is_an_error_with_the_status(site):
    s = site({})
    with pytest.raises(netguard.FetchError) as e:
        fetcher_for(s).fetch("http://example.test/missing")
    assert "404" in str(e.value)


def test_a_slow_server_is_cut_off_at_the_deadline(site):
    def slow():
        time.sleep(1.5)
        return 200, {"Content-Type": "text/plain"}, b"late"
    s = site({"/s": slow})
    t0 = time.monotonic()
    with pytest.raises(netguard.FetchError) as e:
        fetcher_for(s, timeout_s=0.4).fetch("http://example.test/s")
    assert time.monotonic() - t0 < 1.2 and "too long" in str(e.value).lower()


def test_text_with_odd_encodings_does_not_crash(site):
    s = site({"/l": (200, {"Content-Type": "text/html; charset=latin-1"}, "<p>caf\xe9</p>".encode("latin-1")),
              "/bad": (200, {"Content-Type": "text/html; charset=nonsense"}, b"<p>ok \xff\xfe</p>")})
    assert "café" in fetcher_for(s).fetch("http://example.test/l").text
    assert "ok" in fetcher_for(s).fetch("http://example.test/bad").text


def test_a_body_that_trickles_in_is_cut_off_at_the_total_deadline(site):
    s = site({"/t": (200, {"Content-Type": "text/plain"}, "trickle")})
    t0 = time.monotonic()
    with pytest.raises(netguard.FetchError) as e:
        fetcher_for(s, timeout_s=0.6).fetch("http://example.test/t")
    assert time.monotonic() - t0 < 2.0 and "too long" in str(e.value).lower()
