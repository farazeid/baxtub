# Baxtub

Single-file composable end-to-end JAX Reinforcement Learning via Flax NNX.

Copy-paste desired components into algorithm script or import — refer to `baxtub/algorithms/ppo.py`, cut any un-wanted components.

<br>
<br>
<br>

# Run

```bash
git clone https://github.com/farazeid/baxtub
cd baxtub
uv sync
```

```bash
# uv run {algorithm} --config {config}
uv run baxtub/algorithms/ppo.py --config baxtub/configs/ppo.crafter.yaml
```
