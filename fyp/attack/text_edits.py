"""Text-level edit operators (attack v1 text realisation, plan section 3.6).

Every operator maps a catalogue text ("Title: ...; Brand: ...; Categories:
[...]; Price: ...;") to a new text. Two families:

- **Benign operators** are meaning-preserving (spelling variants, word order,
  light word dropout, casing/punctuation, field order). They stand in for
  paraphrases until the LLM-rewrite stage and calibrate how much a real text
  edit moves the Sentence-T5 embedding.
- **Steering operators** are the kind of edit a seller can make to their own
  listing: append/prepend promotional or category keywords, swap the brand.

Rule-based (no LLM) so the experiment is deterministic and free; LLM rewrites
plug into the same ``text -> text`` interface later.
"""

import re
from dataclasses import dataclass

_FIELD_RE = re.compile(
    r"Title: (?P<title>.*?); Brand: (?P<brand>.*?); "
    r"Categories: (?P<cats>\[.*?\]); Price: (?P<price>.*?);?\s*$"
)

# US/UK spelling variants — meaning-preserving by construction.
_SPELLING = {
    "moisturizer": "moisturiser",
    "color": "colour",
    "colors": "colours",
    "odor": "odour",
    "anti-aging": "anti-ageing",
    "fiber": "fibre",
}

# Seller-vocabulary filler for the steering attack (plus title words mined
# from items already on the target prefix — see run_text_eval).
PROMO_PHRASES = ["bestseller", "top rated", "professional", "gift set", "new"]

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class ParsedText:
    title: str
    brand: str
    cats: str
    price: str

    def render(self) -> str:
        return (
            f"Title: {self.title}; Brand: {self.brand}; "
            f"Categories: {self.cats}; Price: {self.price}; "
        )


def parse(text: str) -> ParsedText | None:
    m = _FIELD_RE.match(text)
    if not m:
        return None
    return ParsedText(m["title"], m["brand"], m["cats"], m["price"])


# --- benign operators (meaning-preserving) ---


def spelling_variants(p: ParsedText, rng) -> str:
    words = [_SPELLING.get(w.lower(), w) for w in p.title.split()]
    p.title = " ".join(words)
    return p.render()


def word_shuffle(p: ParsedText, rng) -> str:
    words = p.title.split()
    rng.shuffle(words)
    p.title = " ".join(words)
    return p.render()


def word_dropout(p: ParsedText, rng, rate: float = 0.15) -> str:
    words = p.title.split()
    if len(words) > 4:
        keep = [w for w in words if rng.random() > rate]
        words = keep if len(keep) >= 3 else words[:3]
    p.title = " ".join(words)
    return p.render()


def case_punct(p: ParsedText, rng) -> str:
    p.title = " ".join(_WORD_RE.findall(p.title.lower()))
    return p.render()


def field_reorder(p: ParsedText, rng) -> str:
    return (
        f"Brand: {p.brand}; Categories: {p.cats}; Title: {p.title}; "
        f"Price: {p.price}; "
    )


BENIGN_OPS = {
    "spelling": spelling_variants,
    "word_shuffle": word_shuffle,
    "word_dropout": word_dropout,
    "case_punct": case_punct,
    "field_reorder": field_reorder,
}


# --- steering operators (seller edits) ---


def append_keywords(p: ParsedText, keywords: list[str]) -> str:
    p.title = f"{p.title} {' '.join(keywords)}"
    return p.render()


def prepend_keywords(p: ParsedText, keywords: list[str]) -> str:
    p.title = f"{' '.join(keywords)} {p.title}"
    return p.render()


def swap_brand(p: ParsedText, brand: str) -> str:
    p.brand = brand
    return p.render()


def title_words(texts: list[str], stop: set[str] | None = None) -> list[str]:
    """Most frequent title words across texts (used to mine target-prefix vocab)."""
    from collections import Counter

    stop = stop or set()
    c = Counter()
    for t in texts:
        p = parse(t)
        if p:
            c.update(w.lower() for w in _WORD_RE.findall(p.title) if len(w) > 2)
    return [w for w, _ in c.most_common() if w not in stop]
