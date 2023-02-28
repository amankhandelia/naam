#!/usr/bin/env python
# coding: utf-8

import random
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import grad, jit, vmap

from parse_names import generate_grapheme_mapping, get_top_n_names

from jax_smi import initialise_tracking

initialise_tracking()


# build the dataset
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in list(w) + [purna_virama]:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


# Define the loss function with vmap
@partial(vmap, in_axes=(None, 0, 0))
def loss_fn(params, X, Y):
    C, W1, b1, W2, b2 = params
    emb = C[X]
    h = jnp.tanh(emb.reshape(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Y.shape[0]), Y], axis=-1)
    return loss


# Define the update function
@jit
def update(params, X, Y, lr):
    def reduce_loss_fn(params, X_reshaped, Y_reshaped):
        loss = jnp.mean(loss_fn(params, X_reshaped, Y_reshaped))
        return loss

    grad_fn = jit(grad(reduce_loss_fn))
    X_reshaped = jnp.reshape(X, (device_count, -1, block_size))
    Y_reshaped = jnp.reshape(Y, (device_count, -1))
    loss = loss_fn(params, X_reshaped, Y_reshaped)
    grads = grad_fn(params, X_reshaped, Y_reshaped)
    params = [p - lr * g for p, g in zip(params, grads)]
    return params, loss


# read in all the words
words = get_top_n_names("naam.csv")

stoi, itos = generate_grapheme_mapping(words)
unique_syallable_count = len(stoi)

block_size = 3
purna_virama = "।"


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


# Define the shapes of the parameters
emb_dim = 10
block_size = 3
input_dim = emb_dim * block_size
params_shapes = [
    (unique_syallable_count, emb_dim),
    (input_dim, 200),
    (200,),
    (200, unique_syallable_count),
    (unique_syallable_count,),
]


# Initialize the parameters with random values
key = jax.random.PRNGKey(2147483647)
params = []
for shape in params_shapes:
    key, subkey = jax.random.split(key)
    params.append(jax.random.normal(subkey, shape))

param_count = sum(x.size for x in jax.tree_leaves(params))

device_count = jax.device_count()


# Convert the PyTorch tensors to numpy arrays
Xtr = np.array(Xtr)
Ytr = np.array(Ytr)

# Define the lists to store the training statistics
stepi = []
lossi = []


# Train the model
batch_size = 128
minibatch_count = 200000
for i in range(minibatch_count):
    # Select a minibatch
    ix = np.random.choice(Xtr.shape[0], size=batch_size, replace=False)
    X, Y = Xtr[ix], Ytr[ix]

    # Update the learning rate
    lr = 0.1 if i < 100000 else 0.01

    # Update the parameters and record the loss
    params, loss = update(params, X, Y, lr)
    if not i % 100:
        print(f"At Step {i}: {loss}")

    # Record the statistics
    stepi.append(i)
    lossi.append(jnp.log10(loss))
