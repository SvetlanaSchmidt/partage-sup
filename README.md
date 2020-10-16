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

#### Blind

TODO: add a command to remove supertagging information from a given
supertagging file:
```bash
python -m supertagger blind -i input.supertags > input.blind.supertags
```


Experiments
-----------

If you installed all the dependencies and pulled all sub-modules, just run:
```bash
produce
```

<!---
Make sure all sub-modules are pulled:
```bash
git submodule update -\-init -\-recursive
```
-->



[partage-format]: https://github.com/kawu/partage#data-format "ParTAGe data format"
[fastText]: https://fasttext.cc/ "fastText"
[pytorch]: https://pytorch.org/ "PyTorch"

