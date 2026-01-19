import argparse
import os
import tempfile
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
import yaml
from craftax.craftax_env import make_craftax_env_from_name
from flax import nnx, struct

import wandb
from baxtub.environments.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
from baxtub.utils.logging import batch_log, create_log_dict

# Import Neural Network here
from baxtub.networks.actorcritic import ActorCritic  # isort:skip

# Import extras here
from baxtub.algorithms.datatypes import Transition
from baxtub.algorithms.functional_icm import (
    ICMUpdateState,
    icm_batch_step,
    icm_step,
    init_icm,
)


@struct.dataclass
class RunState:
    obs: jnp.ndarray
    model: nnx.Module
    optim: nnx.Optimizer
    env_state: Any
    batch_idx: int
    key: jax.random.PRNGKey
    extra: dict


@struct.dataclass
class UpdateState:
    model: nnx.Module
    optim: nnx.Optimizer
    batch: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    key: jax.random.PRNGKey


@struct.dataclass
class ActorCriticTransition(Transition):
    value: jnp.ndarray
    log_prob: jnp.ndarray


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config",
        type=str,
        required=True,
        help="Select from `configs/*.yaml`",
    )
    args = args.parse_args()

    with open(Path(args.config)) as file:
        config = yaml.safe_load(file)
    config["training"]["n_steps"] = int(float(config["training"]["n_steps"]))
    config["training"]["lr"] = float(config["training"]["lr"])

    deterministic = config.get("deterministic", True)
    if deterministic:
        os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

    # init experiment run
    config["experiment_name"] = config.get(
        "experiment_name",
        f"""Crafter PPO {config["training"]["n_steps"] // 1e6}M""",
    )

    wandb.init(
        entity=config["entity"],
        project=config["project"],
        name=config["experiment"],
        config=config,
    )

    # start

    key = jax.random.PRNGKey(config["seed"])
    runs_keys = jax.random.split(key, config["n_runs"])

    n_batches = config["training"]["n_steps"] // config["n_envs"] // config["training"]["n_batch_steps"]
    batch_size = config["n_envs"] * config["training"]["n_batch_steps"]
    logging_threads = []

    def lr_schedule(batch_idx: int) -> float:
        return config["training"]["lr"] * (
            1 - (batch_idx // (config["training"]["n_minibatches"] * config["training"]["n_epochs"])) / n_batches
        )

    env = make_craftax_env_from_name(
        config["env"]["id"],
        not config["env"]["optimistic_resets"],
    )
    env_params = env.default_params
    env = LogWrapper(env)
    if config["env"]["optimistic_resets"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["n_envs"],
            reset_ratio=config["env"]["optimistic_resets"]["reset_ratio"],
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(
            env,
            num_envs=config["n_envs"],
        )

    run_fn = partial(
        run,
        config=config,
        n_batches=n_batches,
        batch_size=batch_size,
        logging_threads=logging_threads,
        env=env,
        env_params=env_params,
        lr_schedule=lr_schedule,
    )
    run_fn = nnx.vmap(run_fn)
    run_fn = nnx.jit(run_fn)

    run_state, _ = run_fn(runs_keys)

    print(f"Training completed. Waiting for {len(logging_threads)} logging threads to complete...")
    for thread in logging_threads:
        if thread.is_alive():
            thread.join()
    print("Logging threads complete.")


def run(
    rng: jax.random.PRNGKey,
    #
    config: dict[str, Any],
    n_batches: int,
    batch_size: int,
    logging_threads: list[threading.Thread],
    env: Any,
    env_params: Any,
    lr_schedule: Callable[[int], float],
) -> tuple[Any, Any]:
    key, model_key, env_key, run_key = jax.random.split(rng, 4)

    if "Symbolic" in config["env"]["id"]:
        model = ActorCritic(
            din=env.observation_space(env_params).shape[0],
            layer_width=config["agent"]["layer_width"],
            dout=env.action_space(env_params).n,
            rngs=nnx.Rngs(model_key),
        )
    else:
        raise NotImplementedError("NNX ActorCriticConv not implemented.")

    optim = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(config["training"]["max_grad_norm"]),
            optax.adam(
                learning_rate=lr_schedule if config["training"]["anneal_lr"] else config["training"]["lr"],
                eps=1e-5,
            ),
        ),
    )

    obs, env_state = env.reset(env_key, env_params)

    run_state = RunState(
        obs,
        model,
        optim,
        env_state,
        batch_idx := 0,
        run_key,
        extra := {},
    )

    if config.get("intrinsic", False) and config["intrinsic"].get("ICM", False):
        run_state.extra["icm_state"] = init_icm(
            key,
            env.observation_space(env_params),
            env.action_space(env_params),
            config,
        )

    batch_step_fn = partial(
        batch_step,
        n_batches=n_batches,
        batch_size=batch_size,
        config=config,
        logging_threads=logging_threads,
        env=env,
        env_params=env_params,
    )

    run_state, _ = nnx.scan(
        batch_step_fn,
        length=n_batches,
    )(run_state, None)

    return run_state, _


def batch_step(
    run_state,
    _,
    #
    n_batches: int,
    batch_size: int,
    config: dict[str, Any],
    logging_threads: list[threading.Thread],
    env,
    env_params,
):
    step_fn = partial(step, config=config, env=env, env_params=env_params)

    run_state, batch = nnx.scan(
        step_fn,
        length=config["training"]["n_batch_steps"],
    )(run_state, None)

    rollout_step_fn = partial(rollout_step, config=config)

    _, (returns, advantages) = nnx.scan(
        rollout_step_fn,
        reverse=True,
        unroll=16,
    )(
        (
            next_value := batch.value[-1],  # bootstrap the last value
            next_done := batch.done[-1],  # bootstrap the last done
            prev_advantage := jnp.zeros_like(batch.value[-1]),
        ),
        batch,
    )

    advantages = jnp.where(
        config["training"].get("norm_advantage", False),
        (advantages - advantages.mean()) / (advantages.std() + 1e-8),
        advantages,
    )

    metric_info = jax.tree.map(
        lambda x: (x * batch.info["returned_episode"]).sum() / batch.info["returned_episode"].sum(),
        batch.info,
    )

    epoch_update_fn = partial(epoch_update, config=config, batch_size=batch_size)

    update_state = UpdateState(
        run_state.model,
        run_state.optim,
        batch,
        advantages,
        returns,
        run_state.key,
    )
    update_state, (loss, (policy_loss, value_loss, entropy_loss)) = nnx.scan(
        epoch_update_fn,
        length=config["training"]["n_epochs"],
    )(update_state, None)

    run_state = run_state.replace(
        model=update_state.model,
        optim=update_state.optim,
        batch_idx=run_state.batch_idx + 1,
        key=update_state.key,
    )

    metric_info.update(
        {
            "loss": loss.mean(),
            "policy_loss": policy_loss.mean(),
            "value_loss": value_loss.mean(),
            "entropy_loss": entropy_loss.mean(),
        }
    )

    if config.get("intrinsic", False) and config["intrinsic"].get("ICM", False):
        icm_batch_step_fn = partial(icm_batch_step, config=config, batch_size=batch_size)

        icm_update_state = ICMUpdateState(
            **vars(run_state.extra["icm_state"]),
            batch=batch,
            key=run_state.key,
        )

        icm_update_state, metric_info = icm_batch_step_fn(icm_update_state, metric_info)

    # region logging

    def do_metrics() -> None:
        def metrics_callback(
            metric_info: dict[str, Any],
            batch_idx: int,
        ) -> None:
            # Add NUM_REPEATS for batch logging compatibility
            log_config = config.copy()
            log_config["NUM_REPEATS"] = config["n_runs"]
            log_config["DEBUG"] = True  # Add DEBUG flag for batch logging
            log_config["NUM_STEPS"] = config["training"]["n_batch_steps"]  # Steps per batch, not total steps
            log_config["NUM_ENVS"] = config["n_envs"]

            to_log = create_log_dict(metric_info, log_config)
            batch_log(batch_idx, to_log, log_config)

        jax.debug.callback(
            metrics_callback,
            metric_info,
            run_state.batch_idx,
        )

    def do_checkpoint():
        def save_checkpoint(batch_idx, model_state) -> None:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    checkpoint_path = Path(temp_dir) / f"checkpoint_{batch_idx}"
                    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                    checkpointer.save(checkpoint_path, model_state)

                    artifact = wandb.Artifact(
                        name=f"model_checkpoint_{batch_idx}",
                        type="model",
                        description=f"Model checkpoint at batch {batch_idx}",
                    )
                    artifact.add_dir(str(checkpoint_path))
                    wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Error saving checkpoint at batch {batch_idx}: {e}")

        def checkpoint_callback(batch_idx, model_state) -> None:
            checkpoint_thread = threading.Thread(
                target=save_checkpoint,
                args=(batch_idx, model_state),
                daemon=False,
            )
            logging_threads.append(checkpoint_thread)
            checkpoint_thread.start()

        _, model_state = nnx.split(run_state.model)
        jax.debug.callback(
            checkpoint_callback,
            run_state.batch_idx,
            model_state,
        )

    def do_snapshot():
        def save_snapshot(batch_idx, snapshot) -> None:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    snapshot_path = Path(temp_dir) / f"snapshot_{batch_idx}"
                    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                    checkpointer.save(snapshot_path, snapshot)

                    artifact = wandb.Artifact(
                        name=f"full_snapshot_{batch_idx}",
                        type="snapshot",
                        description=f"Complete training snapshot at batch {batch_idx}",
                    )
                    artifact.add_dir(str(snapshot_path))
                    wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Error saving snapshot at batch {batch_idx}: {e}")

        def snapshot_callback(batch_idx, snapshot) -> None:
            snapshot_thread = threading.Thread(
                target=save_snapshot,
                args=(batch_idx, snapshot),
                daemon=False,
            )
            logging_threads.append(snapshot_thread)
            snapshot_thread.start()

        _, model_state = nnx.split(run_state.model)
        optim_state = nnx.state(run_state.optim)
        snapshot = {
            "obs": run_state.obs,
            "model_state": model_state,
            "optim_state": optim_state,
            "env_state": run_state.env_state,
            "batch_idx": run_state.batch_idx,
            "run_key": run_state.key,
        }
        jax.debug.callback(
            snapshot_callback,
            run_state.batch_idx,
            snapshot,
        )

    jax.lax.cond(
        jnp.logical_or(
            run_state.batch_idx % config["logging"].get("metrics_every", 1) == 0,
            run_state.batch_idx == n_batches - 1,
        ),
        do_metrics,
        lambda: None,
    )

    jax.lax.cond(
        jnp.logical_or(
            run_state.batch_idx % config["logging"].get("checkpoint_every", jnp.inf) == 0,
            run_state.batch_idx == n_batches - 1,
        ),
        do_checkpoint,
        lambda: None,
    )

    jax.lax.cond(
        jnp.logical_or(
            run_state.batch_idx % config["logging"].get("snapshot_every", jnp.inf) == 0,
            run_state.batch_idx == n_batches - 1,
        ),
        do_snapshot,
        lambda: None,
    )

    # endregion

    return run_state, _


def step(
    run_state,
    _,
    #
    config: dict[str, Any],
    env,
    env_params,
) -> tuple[Any, ActorCriticTransition]:
    key, action_key, step_key = jax.random.split(run_state.key, 3)

    distribution, value = run_state.model(run_state.obs)
    action = distribution.sample(seed=action_key)
    log_prob = distribution.log_prob(action)

    next_obs, env_state, reward, done, info = env.step(
        step_key,
        run_state.env_state,
        action,
        env_params,
    )

    transition = ActorCriticTransition(
        obs=run_state.obs,
        action=action,
        next_obs=next_obs,
        reward=reward,
        done=done,
        info=info,
        extra={},
        value=value,
        log_prob=log_prob,
    )

    if config.get("intrinsic", False) and config["intrinsic"].get("ICM", False):
        icm_step_fn = partial(icm_step, config=config)

        transition = icm_step_fn(
            run_state.extra["icm_state"].icm_encoder,
            run_state.extra["icm_state"].icm_forward,
            transition,
        )

    run_state = run_state.replace(
        obs=next_obs,
        env_state=env_state,
        key=key,
    )

    return run_state, transition


def rollout_step(
    carry,
    transition,
    #
    config,
):
    next_value, next_done, prev_advantage = carry
    reward = transition.reward
    value = transition.value

    # gae advantage
    delta = reward + config["training"]["gamma"] * next_value * jnp.logical_not(next_done) - value
    advantage = (
        delta
        + config["training"]["gamma"] * config["training"]["gae_lambda"] * jnp.logical_not(next_done) * prev_advantage
    )

    return (value, transition.done, advantage), (
        advantage + value,
        advantage,
    )


def epoch_update(
    update_state: UpdateState,
    _,
    #
    config,
    batch_size: int,
) -> tuple[UpdateState, Any]:
    key, permutation_key = jax.random.split(update_state.key, 2)

    joint = (update_state.batch, update_state.advantages, update_state.returns)  # shape: (n_steps, n_envs, ...)
    flat_joint = jax.tree.map(  # shape: (batch_size := n_steps * n_envs, ...)
        lambda x: x.reshape((batch_size,) + x.shape[2:]),
        joint,
    )
    permutation = jax.random.permutation(permutation_key, batch_size)
    shuffled_joint = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=0),
        flat_joint,
    )
    minibatches = jax.tree.map(  # shape: (config["training"]["n_minibatches"], minibatch_size, ...)
        lambda x: jnp.reshape(x, [config["training"]["n_minibatches"], -1] + list(x.shape[1:])),
        shuffled_joint,
    )

    minibatch_update_fn = partial(minibatch_update, config=config)

    (model, optim), losses = nnx.scan(
        minibatch_update_fn,
        length=config["training"]["n_minibatches"],
    )((update_state.model, update_state.optim), minibatches)

    return update_state.replace(model=model, optim=optim, key=key), losses


def minibatch_update(
    model_optim,
    minibatch,
    #
    config,
):
    model, optim = model_optim
    batch, advantages, returns = minibatch

    loss_fn_partial = partial(loss_fn, config=config)

    losses, grads = nnx.value_and_grad(loss_fn_partial, has_aux=True)(model, batch, advantages, returns)
    optim.update(grads)

    return model_optim, losses


def loss_fn(
    model,
    transition,
    advantages,
    returns,
    #
    config,
):
    distribution, new_value = model(transition.obs)
    new_log_prob = distribution.log_prob(transition.action)
    entropy = distribution.entropy()

    ratio = jnp.exp(new_log_prob - transition.log_prob)

    policy_loss = jnp.maximum(
        -advantages * ratio,
        -advantages
        * jnp.clip(
            ratio,
            1 - config["training"]["clip_coef"],
            1 + config["training"]["clip_coef"],
        ),
    ).mean()

    value_loss = jnp.where(
        config["training"].get("clip_vloss", False),
        0.5
        * jnp.maximum(
            (new_value - returns) ** 2,
            (
                transition.value
                + jnp.clip(
                    new_value - transition.value,
                    -config["training"]["clip_coef"],
                    config["training"]["clip_coef"],
                )
                - returns
            )
            ** 2,
        ),
        0.5 * ((new_value - returns) ** 2),
    ).mean()

    entropy_loss = entropy.mean()

    loss = policy_loss + value_loss * config["training"]["vf_coef"] - entropy_loss * config["training"]["ent_coef"]
    return loss, (policy_loss, value_loss, entropy_loss)


if __name__ == "__main__":
    main()
