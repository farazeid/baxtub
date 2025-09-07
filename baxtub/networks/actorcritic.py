import distrax
import jax
import jax.numpy as jnp
from flax import nnx


class ActorCritic(nnx.Module):
    def __init__(
        self,
        din: int,
        layer_width: int,
        dout: int,
        rngs: nnx.Rngs,
        activation: str = "tanh",
    ) -> None:
        self.activation = activation
        self.linear1 = nnx.Linear(
            din,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            layer_width,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            layer_width,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear4 = nnx.Linear(
            layer_width,
            dout,
            kernel_init=nnx.initializers.orthogonal(0.01),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

        self.linear1_critic = nnx.Linear(
            din,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear2_critic = nnx.Linear(
            layer_width,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear3_critic = nnx.Linear(
            layer_width,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear4_critic = nnx.Linear(
            layer_width,
            1,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
    ) -> tuple[distrax.Categorical, jax.Array]:
        activation = nnx.relu if self.activation == "relu" else nnx.tanh

        actor_mean = self.linear1(x)
        actor_mean = activation(actor_mean)
        actor_mean = self.linear2(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = self.linear3(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = self.linear4(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = self.linear1_critic(x)
        critic = activation(critic)
        critic = self.linear2_critic(critic)
        critic = activation(critic)
        critic = self.linear3_critic(critic)
        critic = activation(critic)
        critic = self.linear4_critic(critic)

        return pi, jnp.squeeze(critic, axis=-1)
