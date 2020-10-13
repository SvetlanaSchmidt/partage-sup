from typing import Iterable, List, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class STag:
    """Supertag"""
    stag_str: str


@dataclass(frozen=True)
class Token:
    """Token"""
    tok_id: int     # ID of the token (basically its position in the sentence)
    word_form: str  # Word form
    head_dist: Dict[int, float]
                    # Head distribution
    stag_dist: Dict[STag, float]
                    # Supertag distribution


@dataclass(frozen=True)
class Sent:
    """Sentence"""
    tokens: List[Token]


def read_supertags(path: str) -> Iterable[Sent]:
    """Read the list of sentence in a .conllu file."""
    with open(path, "r", encoding="utf-8") as data_file:
        sent = []
        for line in data_file:
            if line.strip():
                tok_id, word, heads, *stags = line.split("\t")
                sent.append(Token(
                    tok_id=int(tok_id),
                    word_form=word,
                    head_dist=parse_head_dist(heads),
                    stag_dist=parse_stag_dist(stags),
                ))
            else:
                yield sent
                sent = []


def parse_head_dist(heads: str) -> Dict[int, float]:
    """Parse the head distribution."""
    xs = []
    for pair in heads.split("|"):
        head, prob = pair.split(":")
        xs.append((int(head), float(prob)))
    return dict(xs)


def parse_stag_dist(stags: List[str]) -> Dict[STag, float]:
    """Parse the supertag distribution."""
    xs = []
    for pair in stags:
        stag, prob = pair.split(":")
        xs.append((STag(stag), float(prob)))
    return dict(xs)
