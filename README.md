# Install

```bash
conda create -n stag python=3.8
conda activate stag
```

If you do not need GPU support:
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
otherwise:
```bash
pip install -r requirements-gpu.txt
```

Then install [disco-dop](https://github.com/andreasvc/disco-dop#installation).
**WARNING**: if you use conda, you should probably not use `--user` when
`pip`-installing disco-dop.
<!---
Discodop require `make install`, is it possible to put it in `requirements.txt`?
-->
