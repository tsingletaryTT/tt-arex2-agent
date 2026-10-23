# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""The claim check (arex2_agent/evidence.py): is each claim in an answer backed by text the agent actually retrieved?"""
from __future__ import annotations

from arex2_agent import evidence
from arex2_agent.tools import Source

OPERA = Source("https://en.wikipedia.org/wiki/Sydney_Opera_House", "Sydney Opera House", "page",
               "The Sydney Opera House is a multi-venue performing arts centre in Sydney. It was designed by Danish architect Jørn Utzon. "
               "It was formally opened on 20 October 1973 by Queen Elizabeth II. The building cost 102 million Australian dollars.")
UTZON = Source("https://en.wikipedia.org/wiki/J%C3%B8rn_Utzon", "Jørn Utzon", "page",
               "Jørn Oberg Utzon was a Danish architect, born on 9 April 1918 in Copenhagen. He is best known for the Sydney Opera House.")


def status(answer, sources, computed=()):
    return [(c.text, c.status) for c in evidence.check_claims(answer, sources, computed)]


def test_a_claim_whose_words_and_numbers_sit_together_in_a_source_is_backed_and_names_it():
    (c,) = evidence.check_claims("The Sydney Opera House was formally opened on 20 October 1973.", [OPERA, UTZON], [])
    assert c.status == "backed" and c.source_url == OPERA.url and "20 October 1973" in c.evidence


def test_the_same_claim_with_a_wrong_number_is_not_backed_and_the_missing_number_is_named():
    (c,) = evidence.check_claims("The Sydney Opera House was formally opened on 20 October 1975.", [OPERA], [])
    assert c.status == "unbacked" and "1975" in c.missing_numbers


def test_a_claim_about_something_no_source_mentions_is_unbacked():
    (c,) = evidence.check_claims("The Golden Gate Bridge was painted orange in a famous campaign during the thirties.", [OPERA, UTZON], [])
    assert c.status == "unbacked" and c.source_url == ""


def test_a_number_the_model_computed_counts_as_computed_when_the_rest_is_backed():
    answer = "Jørn Utzon was born in 1918, which is 55 years before the Sydney Opera House opened in 1973."
    (c,) = evidence.check_claims(answer, [OPERA, UTZON], ["1973 - 1918 = 55"])
    assert c.status == "computed" and c.missing_numbers == []


def test_a_computed_number_does_not_rescue_a_claim_whose_words_are_unsupported():
    (c,) = evidence.check_claims("The Eiffel Tower is repainted every seven years by a team of sixty workers.", [OPERA], ["7 * 60 = 420"])
    assert c.status == "unbacked"


def test_a_number_that_is_neither_in_a_source_nor_computed_is_not_rescued():
    (c,) = evidence.check_claims("Jørn Utzon was born in 1918, which is 56 years before the Sydney Opera House opened in 1973.", [OPERA, UTZON], ["1973 - 1918 = 55"])
    assert c.status == "unbacked" and "56" in c.missing_numbers


def test_numbers_match_across_thousands_separators():
    src = Source("https://x.example/", "X", "page", "The tower has 2,722 steps between the base and the top platform.")
    (c,) = evidence.check_claims("The tower has 2722 steps between the base and the top platform.", [src], [])
    assert c.status == "backed"


def test_a_claim_needs_its_numbers_and_words_in_the_same_stretch_of_a_page_not_scattered():
    src = Source("https://x.example/", "X", "page", "The bridge opened to traffic. " + "Filler sentence about weather. " * 30 + "Seventy engineers worked there in 1937.")
    (c,) = evidence.check_claims("The bridge opened to traffic in 1937.", [src], [])
    assert c.status == "unbacked"


def test_sentences_that_only_say_the_evidence_is_missing_are_not_claims():
    ans = "I could not find the founding year of the company. No evidence was found for the second part."
    assert evidence.check_claims(ans, [OPERA], []) == []


def test_the_sources_section_and_urls_are_not_checked_as_claims():
    ans = "Answer: The opera house opened in 1973 and was designed by Jørn Utzon.\n\nSources:\n- https://en.wikipedia.org/wiki/Sydney_Opera_House\n- https://en.wikipedia.org/wiki/J%C3%B8rn_Utzon"
    out = evidence.check_claims(ans, [OPERA, UTZON], [])
    assert len(out) == 1 and "http" not in out[0].text


def test_short_fragments_and_headings_are_skipped():
    assert evidence.check_claims("Answer: 55\n\nExplanation:\nYes.", [OPERA], []) == []


def test_bullets_and_markdown_marks_are_stripped_from_claim_text():
    (c,) = evidence.check_claims("- **The Sydney Opera House** was designed by the Danish architect Jørn Utzon.", [OPERA], [])
    assert c.text.startswith("The Sydney Opera House") and "*" not in c.text and c.status == "backed"


def test_the_best_matching_source_is_named_not_the_first():
    (c,) = evidence.check_claims("Jørn Utzon was born on 9 April 1918 in Copenhagen.", [OPERA, UTZON], [])
    assert c.source_url == UTZON.url and c.status == "backed"


def test_no_sources_means_nothing_is_backed():
    assert status("The Sydney Opera House was designed by Jørn Utzon.", []) == [("The Sydney Opera House was designed by Jørn Utzon.", "unbacked")]


def test_an_empty_answer_has_no_claims():
    assert evidence.check_claims("", [OPERA], []) == [] and evidence.check_claims("   \n", [OPERA], []) == []


def test_the_summary_counts_each_status():
    cl = evidence.check_claims("The Sydney Opera House was designed by Danish architect Jørn Utzon. The Golden Gate Bridge was painted orange in a famous campaign.", [OPERA], [])
    s = evidence.summarize(cl)
    assert s == {"total": 2, "backed": 1, "computed": 0, "unbacked": 1}


def test_a_search_snippet_alone_is_weaker_and_is_reported_as_such():
    snip = Source("https://en.wikipedia.org/wiki/Sydney_Opera_House", "Sydney Opera House", "snippet", "Designed by Danish architect Jørn Utzon.")
    (c,) = evidence.check_claims("The Sydney Opera House was designed by the Danish architect Jørn Utzon.", [snip], [])
    assert c.status == "backed" and c.source_kind == "snippet"


# ---- cited addresses ------------------------------------------------------------------------------------------------------------

def test_a_cited_address_that_was_fetched_is_ok_even_with_a_trailing_slash_fragment_or_other_scheme():
    ans = "See https://en.wikipedia.org/wiki/Sydney_Opera_House/ and http://en.wikipedia.org/wiki/J%C3%B8rn_Utzon#Early_life for details."
    r = evidence.cited_addresses(ans, [OPERA, UTZON])
    assert r["opened"] == 2 and r["never_opened"] == []


def test_a_cited_address_the_agent_never_opened_is_listed():
    ans = "Sources:\n- https://en.wikipedia.org/wiki/Sydney_Opera_House\n- https://example.com/made-up-page"
    r = evidence.cited_addresses(ans, [OPERA])
    assert r["opened"] == 1 and r["never_opened"] == ["https://example.com/made-up-page"]


def test_trailing_punctuation_is_not_part_of_a_cited_address():
    r = evidence.cited_addresses("(see https://en.wikipedia.org/wiki/Sydney_Opera_House).", [OPERA])
    assert r["opened"] == 1


def test_an_answer_that_cites_nothing_reports_zero():
    assert evidence.cited_addresses("No links here.", [OPERA]) == {"opened": 0, "never_opened": []}


def test_the_same_address_cited_twice_is_counted_once():
    ans = "https://en.wikipedia.org/wiki/Sydney_Opera_House and again https://en.wikipedia.org/wiki/Sydney_Opera_House"
    assert evidence.cited_addresses(ans, [OPERA])["opened"] == 1


# ---- guards found by mutation ------------------------------------------------------------------------------------------------------

def test_a_claim_with_only_some_of_its_words_in_the_source_is_unbacked():
    # 10 content words; the Opera House stretch holds 4 of them (sydney, opera, house, designed)
    (c,) = evidence.check_claims("The Sydney Opera House was designed by a committee of Brazilian sculptors using recycled aluminium panels.", [OPERA], [])
    assert c.status == "unbacked" and c.coverage < 0.6


def test_text_in_the_sources_section_is_not_checked_as_a_claim():
    ans = "Answer: The Sydney Opera House was designed by Jørn Utzon.\n\nSources:\n- The Wikipedia article about the Sydney Opera House and its architect\n"
    out = evidence.check_claims(ans, [OPERA], [])
    assert len(out) == 1 and "article" not in out[0].text


def test_a_bold_sources_heading_also_ends_the_claims():
    ans = "The Sydney Opera House was designed by Jørn Utzon.\n\n**Sources**\n- The Wikipedia article about the Sydney Opera House and its architect\n"
    assert len(evidence.check_claims(ans, [OPERA], [])) == 1


def test_an_address_inside_a_claim_is_removed_from_the_claim_text():
    (c,) = evidence.check_claims("The Sydney Opera House was designed by Jørn Utzon (https://en.wikipedia.org/wiki/Sydney_Opera_House).", [OPERA], [])
    assert "http" not in c.text and c.status == "backed"


def test_the_answer_label_is_not_part_of_the_claim():
    (c,) = evidence.check_claims("Answer: The Sydney Opera House was designed by Jørn Utzon.", [OPERA], [])
    assert not c.text.lower().startswith("answer")


def test_when_two_stretches_share_as_many_words_the_one_with_the_numbers_wins():
    no_number = Source("https://a.example/", "A", "page", "The opera house opened to the public after a long construction.")
    with_number = Source("https://b.example/", "B", "page", "The opera house opened to the public in 1973 after a long construction.")
    (c,) = evidence.check_claims("The opera house opened to the public in 1973 after a long construction.", [no_number, with_number], [])
    assert c.status == "backed" and c.source_url == "https://b.example/"


def test_a_claim_built_from_two_sources_is_backed_when_each_number_is_in_a_source_that_shares_its_words():
    (c,) = evidence.check_claims("Jørn Utzon was born in 1918 and the Sydney Opera House opened in 1973.", [OPERA, UTZON], [])
    assert c.status == "backed" and c.missing_numbers == []


def test_the_summary_counts_computed_claims_separately():
    cl = evidence.check_claims("Jørn Utzon was born in 1918, which is 55 years before the Sydney Opera House opened in 1973.", [OPERA, UTZON], ["1973 - 1918 = 55"])
    assert evidence.summarize(cl) == {"total": 1, "backed": 0, "computed": 1, "unbacked": 0}


def test_a_cited_address_written_with_the_real_characters_matches_the_percent_encoded_source():
    r = evidence.cited_addresses("See https://en.wikipedia.org/wiki/Jørn_Utzon for his life.", [UTZON])
    assert r["opened"] == 1


def test_a_sentence_ending_full_stop_is_not_part_of_the_cited_address():
    r = evidence.cited_addresses("The page is https://en.wikipedia.org/wiki/Sydney_Opera_House.", [OPERA])
    assert r["opened"] == 1 and r["never_opened"] == []


def test_the_evidence_quote_is_cut_to_a_readable_length():
    long = Source("https://x.example/", "X", "page", "The harbour bridge carried " + "very heavy traffic " * 60 + "daily.")
    (c,) = evidence.check_claims("The harbour bridge carried very heavy traffic daily.", [long], [])
    assert c.status == "backed" and len(c.evidence) <= 300


# ---- a claim whose words are split over two sources -------------------------------------------------------------------------------------

ONLY_OPERA = Source("https://en.wikipedia.org/wiki/Sydney_Opera_House", "Sydney Opera House", "page",
                    "The Sydney Opera House is a performing arts centre. It was formally opened by Queen Elizabeth II on 20 October 1973.")
ONLY_UTZON = Source("https://en.wikipedia.org/wiki/J%C3%B8rn_Utzon", "Jørn Utzon", "page", "Jørn Utzon was a Danish architect, born on 9 April 1918 in Copenhagen.")


def test_a_claim_whose_words_are_split_over_two_sources_is_backed_by_both_and_names_both():
    (c,) = evidence.check_claims("Jørn Utzon was born in 1918, 55 years before the Sydney Opera House was formally opened in 1973.", [ONLY_OPERA, ONLY_UTZON], ["1973 - 1918 = 55"])
    assert c.status == "computed" and c.source_url and len(c.also_from) == 1
    assert {c.source_title, c.also_from[0][1]} == {"Sydney Opera House", "Jørn Utzon"}


def test_two_sources_that_together_still_cover_too_little_are_not_enough():
    (c,) = evidence.check_claims("Jørn Utzon was born in 1918 and later painted every ship in Copenhagen harbour for the Danish royal navy.", [ONLY_OPERA, ONLY_UTZON], [])
    assert c.status == "unbacked" and c.also_from == []


def test_a_claim_one_source_covers_alone_names_only_that_source():
    (c,) = evidence.check_claims("The Sydney Opera House was formally opened by Queen Elizabeth II on 20 October 1973.", [ONLY_OPERA, ONLY_UTZON], [])
    assert c.status == "backed" and c.also_from == []


def test_the_second_source_must_add_words_to_count():
    twin = Source("https://other.example/opera", "Opera copy", "page", ONLY_OPERA.text)
    (c,) = evidence.check_claims("The Sydney Opera House was formally opened by Queen Elizabeth II and later repainted completely in bright orange stripes.", [ONLY_OPERA, twin], [])
    assert c.status == "unbacked" and c.also_from == []


def test_words_scattered_over_one_source_far_apart_do_not_count_as_covered():
    src = Source("https://x.example/", "X", "page", "The bridge opened to traffic. " + "Filler sentence about the weather. " * 20 + "The engineers celebrated with a large parade.")
    (c,) = evidence.check_claims("The bridge opened to traffic and the engineers celebrated with a large parade.", [src], [])
    assert c.status == "unbacked" and c.also_from == []


def test_two_sources_that_add_only_a_little_together_stay_below_the_bar():
    a = Source("https://a.example/", "A", "page", "Copenhagen harbour has many ships.")
    b = Source("https://b.example/", "B", "page", "The navy owns a royal yacht.")
    (c,) = evidence.check_claims("Copenhagen harbour hosts every painted ship owned by the navy under royal protection through winter.", [a, b], [])
    assert c.status == "unbacked" and c.also_from == []
