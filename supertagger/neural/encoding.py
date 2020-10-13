from typing import Iterable, Any, Dict


class Encoding:

    """A class which represents a mapping between (hashable) objects
    and unique atoms (represented as ints).

    >>> objects = ["English", "German", "French"]
    >>> enc = Encoding(objects)
    >>> assert "English" == enc.decode(enc.encode("English"))
    >>> assert "German" == enc.decode(enc.encode("German"))
    >>> assert "French" == enc.decode(enc.encode("French"))
    >>> set(range(3)) == set(enc.encode(ob) for ob in objects)
    True
    >>> for ob in objects:
    ...     ix = enc.encode(ob)
    ...     assert 0 <= ix <= enc.obj_num
    ...     assert ob == enc.decode(ix)
    """

    # TODO: enforce (at the level of the types) that the objects are hashable?
    def __init__(self, objects: Iterable[Any]):
        obj_set = set(ob for ob in objects)
        self.obj_num = len(obj_set)
        self.obj_to_ix: Dict[Any, int] = {}
        self.ix_to_obj: Dict[int, Any] = {}
        for (ix, ob) in enumerate(sorted(obj_set)):
            self.obj_to_ix[ob] = ix
            self.ix_to_obj[ix] = ob

    def __eq__(self, other):
        return self.obj_to_ix == other.obj_to_ix and \
            self.ix_to_obj == other.ix_to_obj

    def encode(self, ob: str) -> int:
        return self.obj_to_ix[ob]

    def decode(self, ix: int) -> str:
        return self.ix_to_obj[ix]
