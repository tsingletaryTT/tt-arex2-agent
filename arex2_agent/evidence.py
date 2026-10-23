# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""A plain check of an answer against what the agent actually retrieved. It matches strings. It does not read for meaning.

A claim is a sentence of the answer. It is *backed* when some stretch of up to three consecutive sentences in a source the agent fetched
contains at least 60% of the claim's content words and every number in the claim. It is *computed* when the rest is backed and the missing
numbers appear in the output of the calculator or the Python tool. Otherwise it is *unbacked*. Numbers that sit in a different source than
the main stretch count when that other source's stretch shares words with the claim, so a claim built from two pages can still be backed.

Passing means the wording and the numbers appear in the retrieved text. It does not mean the claim is true, that the source is reliable, or
that the claim says what the source says. Failing means the agent did not retrieve text that contains them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import unquote, urlsplit

COVERAGE = 0.6            # the share of a claim's content words a stretch must contain (a chosen value; nothing measured it)
OTHER_STRETCH = 0.25      # the share of words another stretch needs before its numbers count for the claim
MIN_WORDS = 5
WINDOW = 3
STOP = set("""the and for that with this from are was were has have had but not you your its they their them there which who whom what when where
why how into than then also been being can could would should will shall may might must does did done about after before over under between during
through such these those other more most some any each both either one two three four his her she him our out off per via very much many""".split())
HEDGE = re.compile(r"\b(could not|couldn't|cannot|can't|unable to|did not find|didn't find|no evidence|not found|no information|i do not know|i don't know)\b", re.I)
URL = re.compile(r"https?://[^\s)\]>\"']+")
SECTION = re.compile(r"^\s*(?:[#*_ ]*)(sources?|references?|citations?)(?:[*_ ]*):?[*_ ]*$", re.I | re.M)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class Claim:
    text: str
    status: str                       # "backed", "computed" or "unbacked"
    source_url: str = ""
    source_title: str = ""
    source_kind: str = ""
    evidence: str = ""
    missing_numbers: list = field(default_factory=list)
    coverage: float = 0.0
    also_from: list = field(default_factory=list)      # (url, title) of a second source that supplies the words the first one lacks


def _words(text: str) -> set:
    return {w for w in re.findall(r"[^\W\d_]+", NUMBER.sub(" ", text).casefold()) if len(w) >= 3 and w not in STOP}


def _numbers(text: str) -> set:
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+", text) if s.strip()]


def extract_claims(answer: str) -> list:
    m = SECTION.search(answer)
    body = answer[: m.start()] if m else answer
    claims = []
    for line in body.splitlines():
        line = URL.sub("", line)
        line = re.sub(r"^\s*(?:[-*+>]|\d+[.)])\s+", "", line)
        line = re.sub(r"[*_`#]+", "", line).strip()
        line = re.sub(r"^(?:final )?answer\s*:\s*", "", line, flags=re.I)
        for s in _sentences(line):
            if len(s.split()) >= MIN_WORDS and not HEDGE.search(s):
                claims.append(s.strip())
    return claims


class _Stretches:
    """Every run of up to WINDOW consecutive sentences of one source, with its word and number sets."""

    def __init__(self, source):
        self.source = source
        sents = _sentences(source.text)
        self.items = []
        for i in range(len(sents)):
            for n in range(1, WINDOW + 1):
                chunk = " ".join(sents[i:i + n])
                self.items.append((chunk, _words(chunk), _numbers(chunk)))
                if i + n >= len(sents):
                    break


def check_claims(answer: str, sources: Iterable, computed_texts: Iterable[str] = ()) -> list:
    stretches = [_Stretches(s) for s in sources]
    computed, calcs = set(), []
    for t in computed_texts:
        nums = _numbers(t)
        computed |= nums
        calcs.append((t, nums))
    out = []
    for text in extract_claims(answer):
        cw, cn = _words(text), _numbers(text)
        best_key, best_hit = None, None
        for st in stretches:
            for chunk, words, nums in st.items:
                cov = len(cw & words) / len(cw) if cw else 0.0
                key = (cov, -len(cn - nums))              # most shared words first, then fewest missing numbers
                if best_key is None or key > best_key:
                    best_key, best_hit = key, (cov, st, chunk, nums)
        also = []
        if best_hit is not None and best_hit[0] < COVERAGE:
            joined = _second_source(cw, best_hit, stretches)
            if joined is not None:
                best_hit, also = (joined[0], best_hit[1], best_hit[2], best_hit[3]), [(joined[1].source.url, joined[1].source.title)]
        if best_hit is None or best_hit[0] < COVERAGE:
            calc = _stated_calculation(cn, calcs)
            if calc is not None:
                out.append(Claim(text, "computed", evidence=calc.strip()[:300], coverage=best_hit[0] if best_hit else 0.0))
            else:
                out.append(Claim(text, "unbacked", missing_numbers=sorted(cn), coverage=best_hit[0] if best_hit else 0.0))
            continue
        cov, st, chunk, nums = best_hit
        missing = sorted(cn - nums)
        from_computation = False
        for n in list(missing):
            elsewhere = any(n in nm and len(cw & wd) / len(cw) >= OTHER_STRETCH for s2 in stretches for _, wd, nm in s2.items)
            if elsewhere:
                missing.remove(n)
            elif n in computed:
                missing.remove(n)
                from_computation = True
        status = "unbacked" if missing else ("computed" if from_computation else "backed")
        out.append(Claim(text, status, st.source.url, st.source.title, st.source.kind, chunk[:300], missing, cov, also))
    return out


def _second_source(cw: set, best_hit, stretches):
    """When no single stretch covers enough of the claim, look for a stretch of a different source that adds the missing words.
    Returns (combined coverage, that source's stretches) or None."""
    _, st1, _, _ = best_hit
    have = _words(best_hit[2])
    best = None
    for st in stretches:
        if st is st1:
            continue
        for _, words, _nums in st.items:
            union = len(cw & (have | words)) / len(cw)
            if (best is None or union > best[0]) and union > len(cw & have) / len(cw):
                best = (union, st)
    return best if best is not None and best[0] >= COVERAGE else None


def _stated_calculation(claim_numbers: set, calcs: list):
    """The text of a calculator or Python result that holds every number of a claim, when the claim has at least two numbers; else None.
    A sentence like "The difference between 1973 and 1918 is 55 years" has few words a source could match, and the calculation backs it."""
    if len(claim_numbers) < 2:
        return None
    for text, nums in calcs:
        if claim_numbers <= nums:
            return text
    return None


def summarize(claims) -> dict:
    return {"total": len(claims), "backed": sum(c.status == "backed" for c in claims), "computed": sum(c.status == "computed" for c in claims),
            "unbacked": sum(c.status == "unbacked" for c in claims)}


def _norm_url(url: str) -> str:
    p = urlsplit(url.strip())
    return (p.netloc.lower() + unquote(p.path).rstrip("/") + (("?" + p.query) if p.query else "")).casefold()


def cited_addresses(answer: str, sources: Iterable) -> dict:
    """Which web addresses the answer cites, and which of them the agent never opened (a model can invent a plausible link)."""
    opened = {_norm_url(s.url) for s in sources}
    seen, ok, missing = set(), 0, []
    for raw in URL.findall(answer):
        url = raw.rstrip(".,;:!?)")
        key = _norm_url(url)
        if key in seen:
            continue
        seen.add(key)
        if key in opened:
            ok += 1
        else:
            missing.append(url)
    return {"opened": ok, "never_opened": missing}
