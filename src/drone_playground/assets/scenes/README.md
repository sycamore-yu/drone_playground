# Scene assets

This directory stores external scene data; it is not a task registry.

| Scene | Actual source |
|---|---|
| Navigation fixed layouts | `navigation/catalog.json` and `benchmarks/navigation.yaml` |
| Empty hover/tracking world | `environments/scenes/empty.py`; no separate map asset |
| Racing gates/course | `assets/scenes/racing/lsy_level0.toml`; loader in `environments/scenes/racing.py`, source identity in `racing/SOURCE.md`; upstream mesh assets remain with the vendored task code |
| Random primitive worlds | `environments/scenes/primitives.py`; generated from explicit ranges and seeds |
| Generated navigation bank | `environments/scenes/generated_navigation.py`; generated from a declared seed |

Task defines the requested behavior. Scene defines this project's geometry and obstacle motion.
An asset folder is only necessary when that scene consumes stored data.
Isaac Lab's InteractiveScene uses a broader collection of simulation entities (including robots and sensors); this project keeps those components separately configurable.

Source: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html
