import jax
import jax.numpy as jnp

from jax import grad, jit, pmap, random

from jax_smi import initialise_tracking

initialise_tracking()

key = random.PRNGKey(0)
num_devices = jax.device_count()


# Define MLP
def mlp(params, x):
    for W, b in params:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x)
    return x


# Initialize parameters
def init_params(layer_sizes, key):
    params = []
    for i in range(1, len(layer_sizes)):
        key, subkey = random.split(key)
        input_size, output_size = layer_sizes[i - 1], layer_sizes[i]
        W = random.normal(subkey, (input_size, output_size))
        b = jnp.zeros(output_size)
        params.append((W, b))
    return params


# Define loss function
def loss(params, batch):
    x, y = batch
    logits = mlp(params, x)
    loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * y, axis=1))
    return jax.lax.pmean(loss, axis_name="batch")


# Define accuracy function
def accuracy(params, batch):
    x, y = batch
    predicted_class = jnp.argmax(mlp(params, x), axis=1)
    return jnp.mean(predicted_class == jnp.argmax(y, axis=1))


# Update step
def update(params, batch, lr):
    # calculate the gradient indenpendently across each device
    grads = grad(loss)(params, batch)

    # synchronize the gradients
    grads = jax.lax.pmean(grads, axis_name="batch")

    # update params
    return [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]


# Train function
def train(params, data, epochs, batch_size, lr):
    num_batches = data[0].shape[0] // batch_size

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(num_batches):
            batch_start, batch_end = i * batch_size, (i + 1) * batch_size
            batch = (data[0][batch_start:batch_end], data[1][batch_start:batch_end])
            x, y = batch
            x = jnp.reshape(x, (num_devices, -1, *x.shape[1:]))
            y = jnp.reshape(y, (num_devices, -1, *y.shape[1:]))
            batch_ = (x, y)
            params = pmap(lambda p, b: update(p, b, lr), axis_name="batch")(params, batch_)
            epoch_loss += jnp.mean(pmap(lambda p, b: loss(p, b), axis_name="batch")(params, batch_))

        epoch_loss = epoch_loss / num_batches
        epoch_acc = accuracy(jax.tree_map(lambda x: x[0], params), data)
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.3f}")
    return params


# Generate some toy data
train_data = (random.normal(key, (1000, 784)), jax.nn.one_hot(random.randint(key, (1000,), 0, 10), 10))
test_data = (random.normal(key, (100, 784)), jax.nn.one_hot(random.randint(key, (100,), 0, 10), 10))

# Define hyperparameters
layer_sizes = [784, 10]
epochs = 10
batch_size = 32
learning_rate = 0.1 / num_devices

# Initialize parameters
params = init_params(layer_sizes, key)

# Put initial params on each device
params = jax.tree_map(lambda x: jax.device_put_replicated(x, jax.devices()), params)

# Train the model
params = train(params, train_data, epochs, batch_size, learning_rate)

# Evaluate the model
test_acc = accuracy(jax.tree_map(lambda x: x[0], params), test_data)
print(f"Test Accuracy: {test_acc:.3f}")
