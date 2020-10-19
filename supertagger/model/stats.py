from typing import List
from dataclasses import dataclass

from supertagger.neural.proto import ScoreStats


"""Stats calculated when training the model"""


@dataclass
class AccStats(ScoreStats):
    """Accuracy statistics"""

    tp_num: int
    all_num: int

    def add(self, other):
        return AccStats(
            self.tp_num + other.tp_num,
            self.all_num + other.all_num
        )

    def acc(self):
        return 100 * self.tp_num / self.all_num

    def report(self):
        return f'{self.acc():2.2f}'


@dataclass
class FullStats(ScoreStats):
    """Full statistics (joint model)"""

    pos_stats: AccStats
    uas_stats: AccStats
    stag_stats: AccStats

    def add(self, other: 'FullStats') -> 'FullStats':
        return FullStats(
            pos_stats=self.pos_stats.add(other.pos_stats),
            uas_stats=self.uas_stats.add(other.uas_stats),
            stag_stats=self.stag_stats.add(other.stag_stats),
        )

    def report(self):
        pos_acc = self.pos_stats.acc()
        uas = self.uas_stats.acc()
        stag_acc = self.stag_stats.acc()
        return f"[POS={pos_acc:05.2f} UAS={uas:05.2f} STag={stag_acc:05.2f}]"
