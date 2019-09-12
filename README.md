# DFNets: Distributed Feedback-Looped Networks

This is a keras implementation of DFNets for semi-supervised classification.

Asiri Wijesinghe, Qing Wang, [DFNets: Spectral CNNs for Graphs with feedback-looped Filters](NeurIPS 2019).

## Requirements

* keras (>= 2.2.2)
* TensorFlow (>= 1.9.0)
* sklearn (>= 0.19.1)
* cvxpy (>= 1.0.10)
* networkx (>= 2.2)

## Models and dataset references

We use the same data splitting for each dataset as in Yang et al. [Revisiting semi-supervised learning with graph embeddings](https://arxiv.org/pdf/1603.08861.pdf).

We evaluate our method using 3 different models on Cora, Citeseer, Pubmed, and NELL datasets:

* `DFNet`: A densely connected spectral CNN with feedback-looped filters.
* `DFNet-ATT`:  A self-attention based densely connected spectral CNN with feedback-looped filters.
* `DF-ATT`: A self-attention based CNN model with feedback-looped filters.

## Files description

* dfnets_layer.py - DFNets spectral CNN layer.
* utils.py - data preprocessing, data spliting, and etc.
* dfnets_optimizer.py - coefficients optimizer.
* dfnets_conv_op.py - convolutional operation with feedback-looped filters.
* dfnets_example.ipynb - demo code for dfnets.
