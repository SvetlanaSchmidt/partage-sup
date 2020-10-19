from typing import Iterable, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

import discodop.tree as disco  # type: ignore


# Anchor symbol
ANCHOR = "<>"


@dataclass(frozen=True)
class STag:
    """Supertag"""
    stag_str: str

    def as_tree(self) -> Tuple[disco.ParentedTree, List[Optional[str]]]:
        """Present the supertag as a (disco-dop) tree."""
        return disco.brackettree(self.stag_str)


def tree_pos(
    tree: disco.ParentedTree,
    sent: List[Optional[str]]
) -> Optional[str]:
    """Retrieve the POS tag of the given tree."""
    for child in tree.children:
        if isinstance(child, int):
            if sent[child] == ANCHOR:
                return tree.label
        else:
            may_pos = tree_pos(child, sent)
            if may_pos is not None:
                return may_pos
    return None


@dataclass(frozen=True)
class Token:
    """Token"""
    tok_id: int     # ID of the token (basically its position in the sentence)
    word_form: str  # Word form
    head_dist: Dict[int, float]     # Head distribution (non-empty)
    stag_dist: Dict[STag, float]    # Supertag distribution (non-empty)

    def best_head(self) -> int:
        """Retrieve the head with the highest probability."""
        return self.dist2list(self.head_dist)[0][0]

    def best_stag(self) -> STag:
        """Retrieve the supertag with the highest probability."""
        return self.dist2list(self.stag_dist)[0][0]

    def render(self, decimal: int = 6) -> str:
        head_str = '|'.join(
            f"{head}:{prob:.{decimal}f}"
            for head, prob in self.dist2list(self.head_dist)
        )
        stag_str = '\t'.join(
            f"{stag.stag_str}:{prob:.{decimal}f}"
            for stag, prob in self.dist2list(self.stag_dist)
        )
        return f"{self.tok_id}\t{self.word_form}\t{head_str}\t{stag_str}"

    def dist2list(self, dist: Dict[Any, float]) -> List[Tuple[Any, float]]:
        def snd(tup): return tup[1]
        return sorted(dist.items(), key=snd, reverse=True)

# @dataclass(frozen=True)
# class Sent:
#     """Sentence"""
#     tokens: List[Token]
Sent = List[Token]  # noqa E305


def read_supertags(path: str) -> Iterable[Sent]:
    """Read the list of sentence in a .conllu file."""
    with open(path, "r", encoding="utf-8") as data_file:
        sent = []
        for line in data_file:
            if line.strip():
                try:
                    tok_id, word, heads, *stags = line.split("\t")
                    sent.append(Token(
                        tok_id=int(tok_id),
                        word_form=word,
                        head_dist=parse_head_dist(heads),
                        stag_dist=parse_stag_dist(stags),
                    ))
                except:     # noqa E722
                    raise RuntimeError(f"Couldn't parse line: {line}")
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
        # stag, prob = pair.split(":")
        parts = pair.split(":")
        stag = ':'.join(parts[:-1])
        prob = parts[-1]
        xs.append((STag(stag), float(prob)))
    return dict(xs)


def render_sent(sent: Sent) -> str:
    """Render the sentence in the .supertags format."""
    return "\n".join(tok.render() for tok in sent)
