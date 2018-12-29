# meshnet

This is a neural network where neurons are arranged in N-d space and are connected by springs. The synaptic weights are determined by the forces on the neurons from each spring. The learnable parameters are the positions of the neurons.

At its heart, this is just a matrix factorization algorithm. We represent the weight matrix in terms of N-d particle positions, rather than in terms of the weights themselves. This means that we can control the number of parameters by varying N.
