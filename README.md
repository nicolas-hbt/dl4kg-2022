# Knowledge Graph Embeddings for Link Prediction: Beware of Semantics!

## Abstract
To do.

## Summary
To do.

## Requirements

Below are the library versions used for running the experiments. For better reproducibility, it is recommended sticking to the following versions, although the experiments may well run smoothly with different library versions.

`Python` version 3.9.7

`Numpy` version 1.21.5

`Pandas` version 1.1.3

`PyTorch` version 1.11.0


## Usage

To run your model on a given dataset, the following parameters are to be defined:

`ne`: number of epochs

`lr`: learning rate

`reg`: regularization parameter

`dataset`: the dataset to be used

`emb_dim`: embedding dimension

`neg_ratio`: number of negative examples per positive example

`neg_sampler`: negative sampling strategy

`batch_size`: batch size

`save_each`: validate every k epochs

`criterion_validation`: criterion for tracking the best epoch during validation

`metrics`: metrics to compute on test set

`pipeline`: whether training or testing your model from a pre-trained model (or both)

`device`: cpu or cuda

### Full command-line

Template: `python main.py -ne ne -lr lr -reg reg -dataset dataset -emb_dim emb_dim -neg_ratio neg_ratio -neg_sampler neg_sampler 
-batch_size batch_size -save_each save_each -criterion_validation criterion_validation -metrics metrics -pipeline pipeline -device device`

Example: `python main.py -ne 100 -lr 0.1 -reg 0.0 -dataset EduKG -emb_dim 100 -neg_ratio 1 -neg_sampler rns 
-batch_size 1280 -save_each 10 -criterion_validation mrr-sem -metrics all -pipeline both -device cpu`

### During Training

Every `save_each`, the model parameters will be stored in a particular folder. Given a dataset `dataset` and a model `model`, the following path will be used:

    $ <Current Directory>/models/dataset/model
    
For instance, the parameters of `TransE` on `EduKG` at the epoch 10 can be retrieved at the following address:

    $ <Current Directory>/models/EduKG/TransE/10.pt

## Best Found Hyperparameters

|                          |          | $k$ | $\eta$   | $\gamma$ |
|--------------------------|----------|-------------------------|----------|----------|
| EduKG     | TransE   | $100$                   | $0.003$  | $5$      |
|                          | DistMult | $100$                   | $0.0003$ | $5$      |
|                          | ComplEx  | $100$                   | $0.0003$ | $5$      |
| FB15K-237 | TransE   | $200$                   | $0.003$  | $5$      |
|                          | DistMult | $200$                   | $0.0005$ | $5$      |
|                          | ComplEx  | $200$                   | $0.0005$ | $5$      |
| KG20C     | TransE   | $200$                   | $0.01$   | $5$      |
|                          | DistMult | $200$                   | $0.001$  | $5$      |
|                          | ComplEx  | $200$                   | $0.001$  | $5$      |

## Cite

To update when appropriate.

## Contact
Nicolas Hubert

University of Lorraine, CNRS, Inria, LORIA, France

University of Lorraine, ERPI, France 

<nicolas.hubert@loria.fr>
