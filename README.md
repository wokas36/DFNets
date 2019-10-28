# DFNets: Distributed Feedback-Looped Networks

This is a Keras implementation of DFNets for semi-supervised classification of nodes on graphs.

Asiri Wijesinghe, Qing Wang, [DFNets: Spectral CNNs for Graphs with feedback-Looped Filters](https://arxiv.org/abs/1910.10866).

## Ranking
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dfnets-spectral-cnns-for-graphs-with-feedback/node-classification-on-nell)](https://paperswithcode.com/sota/node-classification-on-nell?p=dfnets-spectral-cnns-for-graphs-with-feedback) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dfnets-spectral-cnns-for-graphs-with-feedback/node-classification-on-pubmed)](https://paperswithcode.com/sota/node-classification-on-pubmed?p=dfnets-spectral-cnns-for-graphs-with-feedback) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dfnets-spectral-cnns-for-graphs-with-feedback/node-classification-on-cora)](https://paperswithcode.com/sota/node-classification-on-cora?p=dfnets-spectral-cnns-for-graphs-with-feedback) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dfnets-spectral-cnns-for-graphs-with-feedback/node-classification-on-citeseer)](https://paperswithcode.com/sota/node-classification-on-citeseer?p=dfnets-spectral-cnns-for-graphs-with-feedback)

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

## Citation

Please cite our paper if you use this code in your research work.

```
@inproceedings{asiri2019dfnets,
  title={DFNets: Spectral CNNs for Graphs with Feedback-Looped Filters}, 
  author={Wijesinghe, Asiri and Wang, Qing}, 
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

## License

MIT License

## Contact for DFNets issues
Please contact me: asiri.wijesinghe@anu.edu.au if you have any questions / submit a Github issue if you find any bugs.
