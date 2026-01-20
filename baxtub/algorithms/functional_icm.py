from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct

from baxtub.algorithms.datatypes import Transition
from baxtub.networks.dynamics import DynamicsEncoder, DynamicsForward, DynamicsInverse


@struct.dataclass
class ICMState:
    icm_encoder: nnx.Module
    icm_encoder_optim: nnx.Optimizer
    icm_inverse: nnx.Module
    icm_inverse_optim: nnx.Optimizer
    icm_forward: nnx.Module
    icm_forward_optim: nnx.Optimizer


@struct.dataclass
class ICMUpdateState(ICMState):
    batch: dict[str, Any]
    key: jax.random.PRNGKey


def init_icm(
    key: jax.random.PRNGKey,
    observation_space,
    action_space,
    config,
) -> ICMState:
    key, icm_encoder_key, icm_forward_key, icm_inverse_key = jax.random.split(key, 4)

    icm_encoder = DynamicsEncoder(
        din=observation_space.shape[0],
        layer_width=config["intrinsic"]["ICM"]["encoder"]["layer_width"],
        n_layers=config["intrinsic"]["ICM"]["encoder"]["n_layers"],
        dout=config["intrinsic"]["ICM"]["latent_dim"],
        rngs=nnx.Rngs(icm_encoder_key),
    )
    icm_encoder_optim = nnx.Optimizer(
        icm_encoder,
        optax.chain(
            optax.clip_by_global_norm(config["intrinsic"]["ICM"]["max_grad_norm"]),
            optax.adam(config["intrinsic"]["ICM"]["lr"], eps=1e-5),
        ),
    )

    icm_forward = DynamicsForward(
        din=config["intrinsic"]["ICM"]["latent_dim"],
        layer_width=config["intrinsic"]["ICM"]["forward"]["layer_width"],
        n_layers=config["intrinsic"]["ICM"]["forward"]["n_layers"],
        dout=config["intrinsic"]["ICM"]["latent_dim"],
        n_actions=action_space.n,
        rngs=nnx.Rngs(icm_forward_key),
    )
    icm_forward_optim = nnx.Optimizer(
        icm_forward,
        optax.chain(
            optax.clip_by_global_norm(config["intrinsic"]["ICM"]["max_grad_norm"]),
            optax.adam(config["intrinsic"]["ICM"]["lr"], eps=1e-5),
        ),
    )

    icm_inverse = DynamicsInverse(
        din=config["intrinsic"]["ICM"]["latent_dim"] * 2,
        layer_width=config["intrinsic"]["ICM"]["inverse"]["layer_width"],
        n_layers=config["intrinsic"]["ICM"]["inverse"]["n_layers"],
        n_actions=action_space.n,
        rngs=nnx.Rngs(icm_inverse_key),
    )
    icm_inverse_optim = nnx.Optimizer(
        icm_inverse,
        optax.chain(
            optax.clip_by_global_norm(config["intrinsic"]["ICM"]["max_grad_norm"]),
            optax.adam(config["intrinsic"]["ICM"]["lr"], eps=1e-5),
        ),
    )

    return ICMState(
        icm_encoder,
        icm_encoder_optim,
        icm_inverse,
        icm_inverse_optim,
        icm_forward,
        icm_forward_optim,
    )


def icm_step(
    icm_encoder: nnx.Module,
    icm_forward: nnx.Module,
    transition: Transition,
    #
    config: dict[str, Any],
) -> Transition:
    latent_obs = icm_encoder(transition.obs)
    latent_next_obs = icm_encoder(transition.next_obs)

    pred_latent_next_obs = icm_forward(latent_obs, transition.action)
    icm_reward = jnp.square(pred_latent_next_obs - latent_next_obs).mean(axis=-1)
    icm_reward = jnp.where(transition.done, 0.0, icm_reward)
    icm_reward *= config["intrinsic"]["ICM"]["reward_coef"]

    extra = transition.extra
    reward = transition.reward
    extra["reward_extrinsic"] = transition.extra.get("reward_extrinsic", transition.reward)  # fmt: skip
    extra["reward_intrinsic"] = transition.extra.get("reward_intrinsic", jnp.zeros_like(icm_reward)) + icm_reward  # fmt: skip
    extra["icm_reward"] = icm_reward
    reward += icm_reward

    return transition.replace(reward=reward, extra=extra)


def icm_batch_step(
    icm_update_state: ICMUpdateState,
    metric_info: dict[str, float],
    #
    config: dict[str, Any],
    batch_size: int,
) -> tuple[ICMState, dict[str, float]]:
    icm_epoch_update_fn = partial(icm_epoch_update, config=config, batch_size=batch_size)

    icm_update_state, (icm_inverse_loss, icm_forward_loss) = nnx.scan(
        icm_epoch_update_fn,
        length=config["intrinsic"]["ICM"]["n_epochs"],
    )(icm_update_state, None)

    metric_info.update(
        {
            "icm_inverse_loss": icm_inverse_loss.mean(),
            "icm_forward_loss": icm_forward_loss.mean(),
            #
            "reward_extrinsic": icm_update_state.batch.extra["reward_extrinsic"].mean(),
            "reward_intrinsic": icm_update_state.batch.extra["reward_intrinsic"].mean(),
            "icm_reward": icm_update_state.batch.extra["icm_reward"].mean(),
        }
    )

    return icm_update_state, metric_info


def icm_epoch_update(
    icm_update_state: ICMUpdateState,
    _,
    #
    config,
    batch_size,
) -> tuple[ICMUpdateState, tuple[float, float]]:
    key, permutation_key = jax.random.split(icm_update_state.key, 2)

    flat_batch = jax.tree.map(  # shape: (batch_size := n_steps * n_envs, ...)
        lambda x: x.reshape((batch_size,) + x.shape[2:]),
        icm_update_state.batch,
    )
    permutation = jax.random.permutation(permutation_key, batch_size)
    shuffled_joint = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=0),
        flat_batch,
    )
    minibatches = jax.tree.map(  # shape: (config["training"]["n_minibatches"], minibatch_size, ...)
        lambda x: jnp.reshape(x, [config["training"]["n_minibatches"], -1] + list(x.shape[1:])),
        shuffled_joint,
    )

    icm_minibatch_update_fn = partial(icm_minibatch_update, config=config)

    _, icm_losses = nnx.scan(
        icm_minibatch_update_fn,
        length=config["intrinsic"]["ICM"]["n_minibatches"],
    )(
        ICMState(
            icm_update_state.icm_encoder,
            icm_update_state.icm_encoder_optim,
            icm_update_state.icm_inverse,
            icm_update_state.icm_inverse_optim,
            icm_update_state.icm_forward,
            icm_update_state.icm_forward_optim,
        ),
        minibatches,
    )

    icm_update_state = icm_update_state.replace(key=key)
    return icm_update_state, icm_losses


def icm_minibatch_update(
    icm_state: ICMState,
    minibatch,
    #
    config,
) -> tuple[ICMState, tuple[float, float]]:
    inverse_loss_fn = partial(icm_inverse_loss, config=config)
    forward_loss_fn = partial(icm_forward_loss, config=config)

    inverse_loss, (encoder_grads, inverse_grads) = nnx.value_and_grad(
        inverse_loss_fn,
        argnums=(0, 1),  # w.r.t. both icm_encoder and icm_inverse
    )(icm_state.icm_encoder, icm_state.icm_inverse, minibatch)

    forward_loss, forward_grads = nnx.value_and_grad(
        forward_loss_fn,
        argnums=1,  # only w.r.t. icm_forward
    )(icm_state.icm_encoder, icm_state.icm_forward, minibatch)

    icm_state.icm_encoder_optim.update(encoder_grads)
    icm_state.icm_inverse_optim.update(inverse_grads)
    icm_state.icm_forward_optim.update(forward_grads)

    return icm_state, (inverse_loss, forward_loss)


def icm_inverse_loss(
    icm_encoder,
    icm_inverse,
    transition,
    #
    config,
) -> float:
    latent_obs = icm_encoder(transition.obs)
    latent_next_obs = icm_encoder(transition.next_obs)

    pred_action_logits = icm_inverse(latent_obs, latent_next_obs)
    true_action = jax.nn.one_hot(
        transition.action,
        num_classes=pred_action_logits.shape[-1],
    )

    bce_loss = -jnp.mean(
        jnp.sum(
            pred_action_logits * true_action * (1 - transition.done[:, None]),
            axis=1,
        )
    )
    return bce_loss * config["intrinsic"]["ICM"]["inverse_loss_coef"]


def icm_forward_loss(
    icm_encoder,
    icm_forward,
    transition,
    #
    config,
) -> float:
    latent_obs = icm_encoder(transition.obs)
    latent_next_obs = icm_encoder(transition.next_obs)

    pred_latent_next_obs = icm_forward(latent_obs, transition.action)

    error = (latent_next_obs - pred_latent_next_obs) * (1 - transition.done[:, None])
    loss = jnp.square(error).mean() * config["intrinsic"]["ICM"]["forward_loss_coef"]
    return loss
