# Forcast Federated Learning

Forcast Federated Learning (FFL) is an open-source [Pytorch](https://pytorch.org/) based framework for machine learning on decentralized data. FFL has been developed to facilitate open experimentation with [Federated Learning (FL)](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html), an approach to machine learning where a shared global model is trained across
many participating clients that keep their training data locally. For example,
FL has been used to train
[prediction models for mobile keyboards](https://arxiv.org/abs/1811.03604)
without uploading sensitive typing data to servers.

FFL enables developers to use low level model aggregation into a federated model. Explicitly using individual data per client and sharing only the local models or model gradients. This helps bridge the gap from simulation, into simulation with isolated clients and private data and onto deployment.

## Installation

See the [install](docs/install.md) documentation for instructions on how to
install FFL as a package or build FFL from
source.

## Getting Started

See the [get started](docs/get_started.md) documentation for instructions on
how to use FFL.