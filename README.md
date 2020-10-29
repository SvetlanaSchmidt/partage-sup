TAG/TWG Supertagger
===================

This repository contains a [ParTAGe][partage-twg]-compliant,
[PyTorch][pytorch]-based implementation of a TAG/TWG supertagger.


**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)


Installation
------------

The tool requires Python 3.8+.  If you use conda, you can set up an appropriate
environment using the following commands (substituting `<env-name>` for the
name of the environment):
```bash
conda create --name <env-name> python=3.8
conda activate <env-name>
```
Then, to install the tool (together with its dependencies), run:
```bash
pip install .
```
<!--
Then, to install the dependencies, including PyTorch with CPU support:
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
If you want to install PyTorch with CUDA support, use the following command
instead:
```bash
pip install -r requirements-gpu.txt
```
-->
Finally, install `disco-dop` from its [github
repository](https://github.com/andreasvc/disco-dop#installation).


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


[partage-twg]: https://github.com/kawu/partage-twg "ParTAGe for TWG repository"
[partage-format]: https://github.com/kawu/partage-twg#data-format "ParTAGe data format"
[fastText]: https://fasttext.cc/ "fastText"
[pytorch]: https://pytorch.org/ "PyTorch"
