# Scene assets

This directory stores external scene data; it is not a task registry.

| Scene | Actual source |
|---|---|
| Navigation fixed layouts | `navigation/catalog.xml`, per-scene MJCF files, and `benchmarks/navigation.yaml` |
| Empty hover/tracking world | `environments/scenes/empty.py`; no separate map asset |
| Racing gates/course | `racing/lsy_level0.xml`; non-geometric settings in `assets/scenes/racing/lsy_native.yaml`; source identity in `racing/SOURCE.md` |
| Random primitive worlds | `environments/scenes/primitives.py`; generated from explicit ranges and seeds |
| Generated navigation bank | `environments/scenes/generated_navigation.py`; generated from a declared seed |

Task defines the requested behavior. Scene defines this project's geometry and obstacle motion.
An asset folder is only necessary when that scene consumes stored data.
Isaac Lab's InteractiveScene uses a broader collection of simulation entities (including robots and sensors); this project keeps those components separately configurable.

Runtime assets live inside the installable Python package so `importlib.resources` works from a wheel.
