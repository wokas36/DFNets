# DFNets: Distributed Feedback-Looped Networks

This is a Keras implementation of DFNets for semi-supervised classification of nodes on graphs.

Asiri Wijesinghe, Qing Wang, [DFNets: Spectral CNNs for Graphs with feedback-Looped Filters](NeurIPS 2019).

## Cite

Please cite our paper if you use this code in your research work.

```
@inproceedings{asiri2019dfnets,
  title={DFNets: Spectral CNNs for Graphs with Feedback-Looped Filters}, 
  author={Wijesinghe, Asiri and Wang, Qing}, 
  booktitle={NeurIPS},
  year={2019}
}
```

## Requirements

* keras (>= 2.2.2)
* tensorflow (>= 1.9.0)
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
* dfnets_optimizer.py - coefficients optimizer.
* dfnets_conv_op.py - convolution operation with feedback-looped filters.
* utils.py - data preprocessing, data spliting, and etc.
* dfnets_example.ipynb - demo code for dfnets.

## Contact for DFNets Issues
Please contact me: asiri.wijesinghe@anu.edu.au if you have any questions / submit a Github issue if you find any bugs.