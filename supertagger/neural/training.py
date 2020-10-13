from typing import List

from datetime import datetime

import torch
import torch.nn as nn

from torch.optim import Adam

from supertagger.neural.utils import batch_loader, IterableDataset


def train(
        model: nn.Module,
        train_set: IterableDataset,
        dev_set: IterableDataset,
        batch_loss: callable,
        accuracies: List[callable],
        learning_rate: float = 2e-3,
        weight_decay: float = 0.01,
        clip: float = 5.0,
        epoch_num: int = 60,
        batch_size: int = 64,
        report_rate: int = 10,
        shuffle: bool = True,
):
    """Train the model on the given dataset w.r.t. the batch_loss function.
    The model parameters are updated in-place.

    Args:
        model: the neural model to be trained
        train_set: the dataset to train on
        dev_set: the development dataset (can be None)
        batch_loss: the objective function we want to minimize;
            note that this function must support backpropagation!
        accuracy: accuracy of the model over the given dataset
        learning_rate: hyper-parameter of the SGD method
        decay: learning rate decay applied with scheduler
        clip: gradient clip size, limit learning rate over epoch
        epoch_num: the number of epochs of the training procedure
        batch_size: size of the SGD batches
        report_rate: how often to report the loss/accuracy on train/dev
        shuffle: shuffled data from dataloader
    """
    # internal config
    round_decimals: int = 4

    # choose Adam for optimization
    # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    optimizer = Adam(
        model.parameters(),
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

    # activate gpu usage for the model if possible, else nothing will change
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Perform SGD in a loop
    for t in range(epoch_num):
        time_begin = datetime.now()
        train_loss: float = 0.0

        # We use a PyTorch DataLoader to provide a stream of
        # dataset element batches
        for batch in batches:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward, backward
            loss = batch_loss(model, batch)
            loss.backward()

            # scaling the gradients down, places a limit on the size of the parameter updates
            # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # optimize
            optimizer.step()

            # save for statistics
            train_loss += loss.item()

            # reduce memory usage by deleting loss after calculation
            # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
            del loss

        # reporting (every `report_rate` epochs)
        if (t + 1) % report_rate == 0:
            with torch.no_grad():

                # dividing by length of train_set making it comparable
                train_loss /= len(train_set)

                # get train acc
                train_acc = [acc(model, train_set) for acc in accuracies]

                # get dev accurracy if given
                if dev_set:
                    dev_acc = [acc(model, dev_set) for acc in accuracies]
                else:
                    dev_acc = [0.0]

                # create message object
                msg = (
                    "@{k}: \t loss(train)={tl:f} \t acc(train)={ta} \t"
                    "acc(dev)={da} \t time(epoch)={ti}"
                )

                def format(x):
                    return round(x, round_decimals)

                # print and format
                print(
                    msg.format(k=t + 1,
                               tl=format(train_loss),
                               ta=list(map(format, train_acc)),
                               da=list(map(format, dev_acc)),
                               ti=datetime.now() - time_begin))
