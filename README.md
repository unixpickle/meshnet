# meshnet

This is a neural network where neurons are arranged in N-d space and connected by springs, and the synaptic weights are determined by the forces on the springs. The parameters are the positions of the neurons.

At its heart, this is just a matrix factorization algorithm. We represent the weight matrix in terms of N-d particle positions, rather than in terms of the weights themselves. This means that we can control the number of parameters by varying N.
