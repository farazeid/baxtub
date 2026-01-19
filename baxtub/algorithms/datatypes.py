from typing import Any

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    next_obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    info: jnp.ndarray
    #
    extra: dict[str, Any]
