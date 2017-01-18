
# Decomposable attention model

## Introduction

Neural architecture for natural language inference. This is a Chainer implementation of
["A Decomposable Attention Model for Natural Language Inference" by Parikh, et al.](https://arxiv.org/abs/1606.01933).

## How to run

### Prerequisite

```python
pip install -r requirements.py
```

Download word2vec model from http://nlp.stanford.edu/data/glove.840B.300d.zip  which is distributed in
http://nlp.stanford.edu/projects/glove/ .

### Running

```
python bin/train.py
```

Currently all the parameters are written inside `train.py` thus you need to modify file.
I checked that implementation is movin in the same way as the original paper, but I have not checked the performance of the model against the original implementation.

## Licence

I distribute this code under the Unlicence.
But it would be nice if you can link to this repo if you are using this implementation.