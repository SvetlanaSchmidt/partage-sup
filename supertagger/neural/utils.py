from typing import Union, List

import torch.nn.utils.rnn as rnn

import torch
import torch.nn as nn

from torch.utils.data import Dataset, IterableDataset, DataLoader

#
#
#  -------- Tensor alias -----------
#
TT = torch.TensorType


#
#
#  -------- eval_on -----------
#
def eval_on(model: nn.Module):
    """
    Enter the evaluation mode.

    Should be used like the `torch.no_grad` function.

    with eval_on(model):
        ...
    """
    class EvalMode:
        def __enter__(self):
            self.mode = model.training
            model.train(False)

        def __exit__(self, _tp, _val, _tb):
            model.train(self.mode)

    return EvalMode()


#
#
#  -------- batch_loader -----------
#
def batch_loader(data_set: Union[IterableDataset, Dataset],
                 batch_size: int = 32,
                 shuffle: bool = False,
                 num_workers: int = 10) -> DataLoader:
    """
    Create a batch data loader from the given data set.
    """
    return DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        shuffle=shuffle,
        num_workers=num_workers,
    )


#
#
#  -------- unpack -----------
#
def unpack(pack: rnn.PackedSequence) -> List[TT]:
    """Convert the given packaged sequence into a list of vectors."""
    padded_pack, padded_len = rnn.pad_packed_sequence(pack, batch_first=True)
    return unpad(padded_pack, padded_len)


#
#
#  -------- unpad -----------
#
def unpad(padded: TT, length: TT) -> List[TT]:
    """Convert the given packaged sequence into a list of vectors."""
    output = []
    for v, n in zip(padded, length):
        output.append(v[:n])
    return output


#
#
#  -------- get_device -----------
#
def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
