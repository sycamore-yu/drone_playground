# Code and technical writing

Follow the [MuJoCo Style Guide](https://github.com/google-deepmind/mujoco/blob/main/STYLEGUIDE.md)
and [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
with interfaces aligned to [Crazyflow](https://github.com/learnsyslab/crazyflow).

Python targets 3.12. Ruff enforces the configured rules, a 100-column line limit,
and Google-style docstrings. Name units and contracts explicitly. Test public
behavior and real physical execution; cover JAX shapes, compilation, batching,
and gradients where the change depends on them. Pixi defines project commands.

MJCF uses standard MuJoCo elements and a single source for physical parameters.
C++ follows the worker's clang-format configuration and upstream conventions.
Generate Protobuf bindings from the checked-in service contract.

Write public documentation and comments in English. `GLOSSARY.md` defines terms;
ADRs record decisions; the specification defines behavior and acceptance; operating
guides explain commands. State capabilities and evidence directly. Put each
material limitation next to the relevant method or measurement. Remove repeated
defensive wording and host-specific development narration while retaining
upstream attribution and reproducibility information.
