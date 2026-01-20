from typing import Any

import jax
from flax import struct


@struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    next_obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: jax.Array
    #
    extra: dict[str, Any]
