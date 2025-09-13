import jax
import jax.numpy as jnp
from flax import nnx


class DynamicsEncoder(nnx.Module):
    def __init__(
        self,
        din: int,
        layer_width: int,
        n_layers: int,
        dout: int,
        rngs: nnx.Rngs,
    ) -> None:
        assert n_layers >= 2, "n_layers must be at least 2"

        linear1 = nnx.Linear(din, layer_width, rngs=rngs)

        self.layers = []
        self.layers.append(linear1)
        for _ in range(n_layers - 2):
            layer = nnx.Linear(layer_width, layer_width, rngs=rngs)
            self.layers.append(layer)

        self.linear_last = nnx.Linear(layer_width, dout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        x = self.linear_last(x)
        return x


class DynamicsForward(nnx.Module):
    def __init__(
        self,
        din: int,
        layer_width: int,
        n_layers: int,
        dout: int,
        n_actions: int,
        rngs: nnx.Rngs,
    ) -> None:
        assert n_layers >= 2, "n_layers must be at least 2"

        self.n_actions = n_actions

        linear1 = nnx.Linear(din + n_actions, layer_width, rngs=rngs)

        self.layers = []
        self.layers.append(linear1)
        for _ in range(n_layers - 2):
            layer = nnx.Linear(layer_width, layer_width, rngs=rngs)
            self.layers.append(layer)

        self.linear_last = nnx.Linear(layer_width, dout, rngs=rngs)

    def __call__(
        self,
        latent: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        action_1hot = jax.nn.one_hot(action, num_classes=self.n_actions)
        x = jnp.concatenate((latent, action_1hot), axis=-1)
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        x = self.linear_last(x)
        return x


class DynamicsInverse(nnx.Module):
    def __init__(
        self,
        din: int,  # concatenated latent and next_latent
        layer_width: int,
        n_layers: int,
        n_actions: int,
        rngs: nnx.Rngs,
    ) -> None:
        assert n_layers >= 2, "n_layers must be at least 2"

        linear1 = nnx.Linear(din, layer_width, rngs=rngs)

        self.layers = []
        self.layers.append(linear1)
        for _ in range(n_layers - 2):
            layer = nnx.Linear(layer_width, layer_width, rngs=rngs)
            self.layers.append(layer)

        self.linear_last = nnx.Linear(layer_width, n_actions, rngs=rngs)

    def __call__(
        self,
        latent: jax.Array,
        next_latent: jax.Array,
    ) -> jax.Array:
        x = jnp.concatenate((latent, next_latent), axis=-1)
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        action_raw = self.linear_last(x)
        action_logits = jax.nn.log_softmax(action_raw)
        return action_logits
