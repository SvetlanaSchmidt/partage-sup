from __future__ import annotations

from typing import List, Tuple, Iterable, List, TypedDict

from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset

# import supertagger.neural.proto
from supertagger.neural.proto import \
    ScoreStats, Neural, Inp, Out, Y, S
from supertagger.neural.utils import \
    batch_loader, simple_loader, eval_on


def batch_score(
    neural: Neural[Inp, Out, Y, S],
    batch: Iterable[Tuple[Inp, Out]]
) -> S:
    with torch.no_grad():
        with eval_on(neural.module()):
            total = None
            for inp, gold in batch:
                pred_enc = neural.forward(inp)
                pred = neural.decode(pred_enc)
                score = neural.score(gold, pred)
                if total is None:
                    total = score
                else:
                    total = total.add(score)
            assert total is not None
            return total


class StageConfig(TypedDict):
    """Training stage configuration"""
    epoch_num: int
    learning_rate: float


class TrainConfig(TypedDict):
    """Training configuration"""
    stages: List[StageConfig]
    report_rate: int
    batch_size: int
    shuffle: bool
    # weight_decay: float = 0.01
    # clip: float = 5.0


def train(
    neural: Neural[Inp, Out, Y, S],
    train_set: Dataset[Tuple[Inp, Out]],
    dev_set: Dataset[Tuple[Inp, Out]],
    learning_rate: float = 2e-3,
    # weight_decay: float = 0.01,
    # clip: float = 5.0,
    epoch_num: int = 60,
    batch_size: int = 64,
    report_rate: int = 10,
    shuffle: bool = True,
):
    # choose Adam for optimization
    # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    optimizer = Adam(
        neural.module().parameters(),
        lr=learning_rate,
        # https://github.com/kawu/mwe-collab/issues/54
        # weight_decay=weight_decay,
    )

    # create batched loader
    batches = batch_loader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )

    # # activate gpu usage for the model if possible
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = neural.model.to(device)

    # Perform SGD in a loop
    for t in range(epoch_num):
        time_begin = datetime.now()

        train_loss: float = 0.0

        # We use a PyTorch DataLoader to provide a stream of
        # dataset batches
        for batch in batches:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward, backward
            loss = neural.loss(batch)
            loss.backward()

            # # scaling the gradients down, places a limit on the size of the parameter updates
            # # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
            # nn.utils.clip_grad_norm_(neural.module().parameters(), clip)

            # optimize
            optimizer.step()

            # save for statistics
            train_loss += loss.item()

            # reduce memory usage by deleting loss after calculation
            # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
            del loss

       # reporting (every `report_rate` epochs)
        if (t + 1) % report_rate == 0:
            # with torch.no_grad():

            # divide by number of batches to make it comparable across
            # different datasets (assuming that batch loss is averaged
            # over the batch)
            train_loss = train_loss * batch_size / len(train_set)

            # training score
            train_score = batch_score(neural, simple_loader(train_set))

            # dev score
            dev_score = None
            if dev_set:
                dev_score = batch_score(neural, simple_loader(dev_set))

            # print stats
            msg = (
                "@{k}: \t"
                "loss(train)={tl:f} \t"
                "score(train)={ta} \t"
                "score(dev)={da} \t"
                "time(epoch)={ti}"
            )

            print(
                msg.format(
                    k=t + 1,
                    tl=train_loss,
                    ta=train_score.report(),
                    da=dev_score.report() if dev_score else "n/a",
                    ti=datetime.now() - time_begin
                )
            )
