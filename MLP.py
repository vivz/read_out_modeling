#!/usr/bin/env python

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers, activation_fn='relu',
                 activation_fn_out=None, include_bias=True):
        super().__init__()
        self.input_dim = input_dim  # The number of features in the input to each RNN cell
        self.hidden_dim = hidden_dim  # Number of nodes in each hidden layer
        self.output_dim = output_dim  # The number of items in the output
        self.n_hidden_layers = n_hidden_layers  # Number of hidden layers per cell

        self.activation_fn = nn.ReLU()
        self.activation_fn_out = nn.Sigmoid()
        self.include_bias = include_bias

        layers = []
        for L in range(n_hidden_layers):
            layers.append(nn.Linear(input_dim if L == 0 else hidden_dim, hidden_dim, bias=include_bias))
            layers.append(self.activation_fn)
        layers.append(nn.Linear(hidden_dim, output_dim, bias=include_bias))
        layers.append(self.activation_fn_out)
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
