from __future__ import annotations

from typing import List, Tuple, Iterable  # , Type

from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset

# import supertagger.neural.proto
from supertagger.neural.proto import \
    Score, Neural, Inp, Out, Y, Z, S
from supertagger.neural.utils import \
    batch_loader, simple_loader, eval_on


def batch_loss(
    neural: Neural[Inp, Out, Y, Z, S],
    batch: Iterable[Tuple[Inp, Out]]
) -> torch.Tensor:
    loss = torch.tensor(0.0, dtype=torch.float)
    for inp, out in batch:
        # x = neural.embed(inp)
        y = neural.forward(inp)
        z = neural.encode(out)
        loss = loss + neural.loss(z, y)
    return loss


def batch_score(
    neural: Neural[Inp, Out, Y, Z, S],
    batch: Iterable[Tuple[Inp, Out]]
) -> S:
    with torch.no_grad():
        with eval_on(neural.model):
            total = None
            for inp, gold in batch:
                pred = neural.decode(neural.forward(inp))
                score = neural.score(gold, pred)
                if total is None:
                    total = score
                else:
                    total = total.add(score)
            assert total is not None
            return total


def train(
    neural: Neural[Inp, Out, Y, Z, S],
    train_set: Dataset[Tuple[Inp, Out]],
    dev_set: Dataset[Tuple[Inp, Out]],
        # model: nn.Module,
        # train_set: IterableDataset,
        # dev_set: IterableDataset,
        # batch_loss: callable,
        # accuracies: List[callable],
    learning_rate: float = 2e-3,
    weight_decay: float = 0.01,
    clip: float = 5.0,
    epoch_num: int = 60,
    batch_size: int = 64,
    report_rate: int = 10,
    shuffle: bool = True,
):
    # internal config
    round_decimals: int = 4

    # choose Adam for optimization
    # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    optimizer = Adam(
        neural.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # create batched loader
    batches = batch_loader(
        # Use no shuffling, it doesn't work with iterable datasets
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
            loss = batch_loss(neural, batch)
            loss.backward()

            # # scaling the gradients down, places a limit on the size of the parameter updates
            # # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
            # nn.utils.clip_grad_norm_(model.parameters(), clip)

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

            # # dividing by length of train_set making it comparable
            # train_loss /= len(train_set)

            # training score
            train_score = batch_score(neural, simple_loader(train_set))

            # dev score
            dev_score = None
            if dev_set:
                dev_score = batch_score(neural, simple_loader(dev_set))

            # create message object
            msg = (
                "@{k}: \t"
                "loss(train)={tl:f} \t"
                "score(train)={ta:f} \t"
                "score(dev)={da:f} \t"
                "time(epoch)={ti}"
            )

            # def format(x):
            #     return round(x, round_decimals)

            # print and format
            print(
                msg.format(
                    k=t + 1,
                    tl=train_loss,
                    ta=train_score.as_float(),
                    da=dev_score.as_float() if dev_score else 0.,
                    ti=datetime.now() - time_begin
                )
            )
