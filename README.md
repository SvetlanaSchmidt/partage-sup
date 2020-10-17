TWG Supertagger
===============

This repository contains a [PyTorch](pytorch)-based implementation of a TAG/TWG
supertagger.


**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)


Installation
------------

The tool requires Python 3.8+.  If you use conda, you can set up an appropriate
environment using the following commands (substituting `<env-name>` for the
name of the environment):
```bash
conda create --name <env-name> python=3.8
conda activate <env-name>
```
Then, to install (most of) the dependencies:
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
<!---
otherwise:
```bash
pip install -r requirements-gpu.txt
```
-->
Finally, install `disco-dop` from its [github
repository](https://github.com/andreasvc/disco-dop#installation) (the [version
on PyPI](https://pypi.org/project/disco-dop/) is outdated).
<!---
(**warning**: if you use conda, you should probably *not* use `-\-user` when
`pip`-installing disco-dop).

Discodop require `make install`, is it possible to put it in `requirements.txt`?
-->


Usage
-----

#### Data format

The tool supports the same format as [partage][partage-format].

#### Configuration

The model (embedding size, BiLSTM depth, etc.) and training (number of epochs,
learning rates, etc.) configuration is currently hard-coded in
[supertagger/config.py](supertagger/config.py).  It be replaced during training
by providing appropriate `.json` configuration files.

#### Training

To train a supertagging model, you will need:
* `fastText.bin`: a binary [fastText][fastText] model (**important**: the size
  of the fastText model must be specified in the
  [configuration](#configuration))
* `train.supertags`: a training dataset (see [data format](#data-format))
* `dev.supertags` (optional): a development dataset

Then, to train a model and save it in `model.pth`:
```bash
python -m supertagger train -f fastText.bin -t train.supertags -d dev.supertags --save model.pth
```
See `python -m supertagger train --help` for additional training options.

#### Tagging

To use an existing model to supertag a given `input.supertags` file:
```bash
python -m supertagger tag -f fastText.bin -i input.supertags
```

#### TODO: Blind

Add a command to remove supertagging information from a given supertagging
file, e.g.:
```bash
python -m supertagger blind -i input.supertags > input.blind.supertags
```


Experiments
-----------

First make sure that the `TWG` sub-module is available.  If not sure, run:
```bash
git submodule update --init --recursive
```

To run the supertagger on the French TWG dataset:
```bash
produce data/exp.300/dev.eval
```
This will download the corresponding [fastText](fastText) French model, train
the supertagging model, and output the evaluation scores on the dev set (you
can replace `dev` with `test` in the command above to get the results on the
test set).

To speed up the process, you can alternatively run an experiment with a [smaller,
100-dimensional fastText model](fastText-fr-small) using the following command:
```bash
produce data/exp.100/dev.eval
```
Note that, due to a reduced size of the embedding model, the evaluation scores
may be significantly lower in this setting.  Official [fastText](fastText)
models are 300-dimensional and the one required for this experiment was
obtained using the Python `fasttext.util.reduce_model` function (see [this
thread](fastText-reduction)).



[partage-format]: https://github.com/kawu/partage#data-format "ParTAGe data format"
[fastText]: https://fasttext.cc/ "fastText"
[pytorch]: https://pytorch.org/ "PyTorch"
[fastText-reduction]: https://stackoverflow.com/questions/58930298/reducing-size-of-facebooks-fasttext-word2vec "Reducing the size of fastText models"
[fastText-fr-small]: https://user.phil.hhu.de/~waszczuk/treegrasp/fasttext/cc.fr.100.bin.gz "100-dimensional fastText French model"
