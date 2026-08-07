# Baxtub

Single-file composable end-to-end JAX Reinforcement Learning via Flax NNX.

Copy-paste desired components into algorithm script or import — refer to `baxtub/algorithms/ppo.py`, cut any un-wanted components.

Derived from [PureJaxRL](https://github.com/luchris429/purejaxrl) and [Craftax](https://github.com/MichaelTMatthews/Craftax_Baselines), Baxtub is (1) a NNX implementation and (2) algorithm scripts such as `baxtub/algorithms/ppo.py` are written to be read top-to-bottom with sections of the algorithm decomposed into separate functions for ease of identifying their inputs and outputs.

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
