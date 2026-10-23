#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Build arex2_agent/data/missions.json and corpus.json by reading Wikipedia and arXiv, and fail when a preset's expected answer is not supported.

    python tools/build_missions.py            # fetch, verify, write the two files
    python tools/build_missions.py --check    # fetch and verify the existing files still hold (no writes)

Each preset mission has facts the answer depends on. This script fetches the pages, requires a regular expression to match in each, keeps the matching
sentence as a quote with the date, and computes the expected answer from the numbers it found, so no fact in a preset is written from memory.
The offline corpus is trimmed text of the same pages (Wikipedia text is CC BY-SA 4.0; each corpus entry keeps its address and the date read).
Uses the app's own tools, so it exercises them against the real sites too.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arex2_agent import tools  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "arex2_agent" / "data"
KEEP = 2600


def sentence_with(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"fact not found: {pattern!r}")
    start = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
    end_candidates = [i for i in (text.find(". ", m.end()), text.find("\n", m.end())) if i != -1]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return re.sub(r"\s+", " ", text[start:end]).strip()


CACHE = Path(__file__).resolve().parent / ".cache"       # fetched pages, so a re-run after a rate limit only fetches what is missing


class Reader:
    def __init__(self):
        self.tb = tools.Toolbox(env={})
        self.pages = {}

    def wiki(self, title: str):
        if title not in self.pages:
            f = CACHE / (re.sub(r"\W+", "_", title) + ".json")
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                self.pages[title] = tools.Source(d["url"], d["title"], d["kind"], d["text"], d.get("tool", ""))
                return self.pages[title]
            time.sleep(4)                    # Wikipedia rate-limits quick bursts
            r = self.tb.run("wikipedia_page", json.dumps({"title": title}))
            if not r.ok:
                raise SystemExit(f"cannot read {title!r}: {r.text}")
            src = r.sources[0]
            CACHE.mkdir(exist_ok=True)
            f.write_text(json.dumps(src.__dict__, ensure_ascii=False), encoding="utf-8")
            self.pages[title] = src
        return self.pages[title]

    def arxiv(self, query: str):
        r = self.tb.run("arxiv_search", json.dumps({"query": query, "max_results": 1}))
        if not r.ok or not r.sources:
            raise SystemExit(f"arXiv lookup failed for {query!r}: {r.text}")
        return r


def build():
    rd, today = Reader(), datetime.date.today().isoformat()
    corpus, missions = [], []

    def add_doc(kind, title, url, facts, extra_text=""):
        lead = extra_text
        keep = [lead] if lead else []
        for q in facts:
            if q not in " ".join(keep):
                keep.append(q)
        text = " ".join(keep)[:KEEP]
        corpus.append({"id": re.sub(r"\W+", "_", title.lower()).strip("_"), "title": title, "url": url, "kind": kind, "text": text,
                       "license": "CC BY-SA 4.0 (Wikipedia contributors)" if kind == "wikipedia" else "arXiv abstract", "read": today})

    def wiki_facts(title, patterns, lead_chars=700):
        src = rd.wiki(title)
        quotes = [sentence_with(src.text, p) for p in patterns]
        add_doc("wikipedia", title, src.url, quotes, extra_text=src.text[:lead_chars].rsplit(" ", 1)[0])
        return src, quotes

    # 1. Sydney Opera House and its architect
    opera, q1 = wiki_facts("Sydney Opera House", [r"formally opened by Queen Elizabeth II on \d+ \w+ \d{4}"])
    utzon, q2 = wiki_facts("Jørn Utzon", [r"\d+ April \d{4} – \d+ \w+ \d{4}\) was a Danish architect"])
    opened = int(re.search(r"opened by Queen Elizabeth II on \d+ \w+ (\d{4})", opera.text).group(1))
    born = int(re.search(r"(\d+) April (\d{4}) –", utzon.text).group(2))
    missions.append({"id": "opera-utzon", "title": "How old was the architect when the building opened?", "kind": "research",
                     "prompt": "The architect of the Sydney Opera House was born in one year and the building was formally opened in another. How many years before the "
                               "opening was he born? Give both years and the difference, with your sources.",
                     "expected": f"{opened - born} years (born {born}, formally opened {opened})", "facts": [str(born), str(opened), str(opened - born)], "derived": [str(opened - born)],
                     "needs": ["wikipedia_page", "calculator"], "plain_chat_risk": "It may remember one of the years wrongly, or the difference.",
                     "verified": {"read": today, "quotes": [{"url": opera.url, "quote": q1[0]}, {"url": utzon.url, "quote": q2[0]}]}})

    # 2. Ada Lovelace and her father
    ada, q3 = wiki_facts("Ada Lovelace", [r"née Byron; \d+ \w+ \d{4} – \d+ \w+ \d{4}\)"])
    byron, q4 = wiki_facts("Lord Byron", [r"\(\d+ \w+ \d{4} – \d+ \w+ \d{4}\), was a British poet"])
    ada_born = re.search(r"née Byron; (\d+) (\w+) (\d{4})", ada.text)
    byron_died = re.search(r"– (\d+) (\w+) (\d{4})\), was a British poet", byron.text)
    months = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
    b = datetime.date(int(ada_born.group(3)), months[ada_born.group(2)], int(ada_born.group(1)))
    d = datetime.date(int(byron_died.group(3)), months[byron_died.group(2)], int(byron_died.group(1)))
    age = d.year - b.year - ((d.month, d.day) < (b.month, b.day))
    missions.append({"id": "ada-byron", "title": "Ada Lovelace's age when her father died", "kind": "research",
                     "prompt": "How old was Ada Lovelace when her father died? Say who her father was, give the two dates you used, and work out her age in whole years.",
                     "expected": f"{age} years old (born {b.isoformat()}, her father Lord Byron died {d.isoformat()})",
                     "facts": ["Byron", str(b.year), str(d.year), str(age)], "derived": [str(age)], "needs": ["wikipedia_page", "calculator"],
                     "plain_chat_risk": "It may know the dates roughly and get the age or the exact dates wrong.",
                     "verified": {"read": today, "quotes": [{"url": ada.url, "quote": q3[0]}, {"url": byron.url, "quote": q4[0]}]}})

    # 3. Everest and K2
    ev, q5 = wiki_facts("Mount Everest", [r"measured in \d{4} .*?as 8,\d{3}\.\d+ m"])
    k2, q6 = wiki_facts("K2", [r"at 8,\d{3} metres .*? above sea level"])
    ev_h = float(re.search(r"as (8,\d{3}\.\d+) m", ev.text).group(1).replace(",", ""))
    k2_h = float(re.search(r"at (8,\d{3}) metres", k2.text).group(1).replace(",", ""))
    diff = round(ev_h - k2_h, 2)
    pct = round(diff / ev_h * 100, 1)
    missions.append({"id": "everest-k2", "title": "Everest against K2", "kind": "research",
                     "prompt": "According to Wikipedia, by how many metres is Mount Everest higher than K2, and what is that difference as a percentage of Everest's height "
                               "(one decimal place)? Quote both heights.",
                     "expected": f"{diff} m, about {pct}% (Everest {ev_h} m, K2 {k2_h:g} m, as Wikipedia states them on {today})",
                     "facts": [f"{ev_h:g}", f"{k2_h:g}", str(pct)], "derived": [str(pct)], "needs": ["wikipedia_page", "calculator"],
                     "plain_chat_risk": "Heights have several published values; the mission asks for the ones on the page, and the percentage needs exact arithmetic.",
                     "verified": {"read": today, "quotes": [{"url": ev.url, "quote": q5[0]}, {"url": k2.url, "quote": q6[0]}], "note": "Wikipedia figures can change; the expected answer is as of the date read."}})

    # 4. Best Picture, its director, his birth year
    oscars, q7 = wiki_facts("70th Academy Awards", [r"Titanic won 11 awards, including Best Picture"])
    titanic, q8 = wiki_facts("Titanic (1997 film)", [r"written and directed by James Cameron"])
    cam, q9 = wiki_facts("James Cameron", [r"James Francis Cameron \(born \w+ \d+, \d{4}\)"])
    cam_born = int(re.search(r"\(born \w+ \d+, (\d{4})\)", cam.text).group(1))
    missions.append({"id": "oscars-director", "title": "The Best Picture director's birth year", "kind": "research",
                     "prompt": "Which film won Best Picture at the 70th Academy Awards, who directed it, and in what year was that director born? "
                               "Then say how many years before the year 2000 that was.",
                     "expected": f"Titanic, directed by James Cameron, born {cam_born}; {2000 - cam_born} years before 2000", "facts": ["Titanic", "Cameron", str(cam_born), str(2000 - cam_born)], "derived": [str(2000 - cam_born)],
                     "needs": ["wikipedia_page", "calculator"], "plain_chat_risk": "Three lookups in a chain; a slip in the first one changes everything after it.",
                     "verified": {"read": today, "quotes": [{"url": oscars.url, "quote": q7[0]}, {"url": titanic.url, "quote": q8[0]}, {"url": cam.url, "quote": q9[0]}]}})

    # 5. an arXiv paper
    r = rd.arxiv('ti:"Attention Is All You Need"')
    src = r.sources[0]
    authors = re.search(r"^\s*Authors: (.*)$", r.text, re.M).group(1).split(", ")
    if len(authors) != 8 or authors[-1] != "Illia Polosukhin":
        raise SystemExit(f"unexpected author list for the attention paper: {authors}")
    posted = re.search(r"\((\d{4}-\d{2}-\d{2})\)", r.text).group(1)
    corpus.append({"id": "attention_is_all_you_need", "title": src.title, "url": re.sub(r"v\d+$", "", src.url), "kind": "arxiv",
                   "text": (src.text.split(". Authors:", 1)[0] + f". Posted {posted}. Authors:" + src.text.split(". Authors:", 1)[1])[:KEEP],
                   "license": "arXiv abstract", "read": today})
    missions.append({"id": "attention-authors", "title": "Who wrote the Transformer paper", "kind": "research",
                     "prompt": "Find the arXiv paper 'Attention Is All You Need'. How many authors does it list, who is listed last, and in what year was the first version posted?",
                     "expected": f"{len(authors)} authors, last {authors[-1]}; first posted {posted[:4]}", "facts": [str(len(authors)), authors[-1], posted[:4]], "derived": [str(len(authors))], "needs": ["arxiv_search"],
                     "plain_chat_risk": "A model often knows the paper and still miscounts or misorders the author list.",
                     "verified": {"read": today, "quotes": [{"url": src.url, "quote": src.text[:200]}]}})

    # 6. a computation
    digits = sum(int(c) for c in str(2 ** 1000))
    missions.append({"id": "digits-2-1000", "title": "Digits of a large power", "kind": "compute",
                     "prompt": "What is the sum of the decimal digits of 2 to the power 1000? Then say whether 2 to the power 1000 is divisible by 7, and show how you checked.",
                     "expected": f"{digits}; not divisible by 7 (2^n mod 7 is 1, 2 or 4)", "facts": [str(digits), "not"], "derived": [str(digits)], "needs": ["run_python"],
                     "plain_chat_risk": "A 302-digit number cannot be added up reliably from memory or by reasoning alone.",
                     "verified": {"read": today, "quotes": [{"url": "", "quote": f"computed here: sum of digits of 2**1000 = {digits}; 2**1000 % 7 = {2 ** 1000 % 7}"}]}})

    # distractors, so the offline search has more than the answers to choose from
    wiki_facts("Eiffel Tower", [], 1400)
    wiki_facts("Charles Babbage", [], 1400)
    return corpus, missions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the existing files against the sites and write nothing")
    a = ap.parse_args(argv)
    corpus, missions = build()
    if a.check:
        old = json.loads((DATA / "missions.json").read_text())
        changed = [m["id"] for m, o in zip(missions, old) if m["expected"] != o["expected"]]
        print("expected answers changed since the files were built: " + (", ".join(changed) or "none"))
        return 1 if changed else 0
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "corpus.json").write_text(json.dumps(corpus, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (DATA / "missions.json").write_text(json.dumps(missions, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(missions)} missions and {len(corpus)} corpus entries ({sum(len(c['text']) for c in corpus)} characters)")
    for m in missions:
        print(f"- {m['id']}: {m['expected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
