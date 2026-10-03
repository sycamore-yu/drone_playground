# LSY Level0 course

Source: learnsyslab/lsy_drone_racing, pinned commit `b1f5b36adb8e08e8e2adea85de790bd0e0a1d118`.

The TOML was relocated unchanged from the vendored LSY task package. It retains the upstream course, simulation and deployment configuration; `RacingScene.config()` supplies the selected dynamics and disables training randomization for the nominal scene.

SHA-256: `2f9c8e60c2ca7606d7c37f14d0602291a2de38568d2407ddbb005834ef398630`.

License: MIT, retained in `src/drone_playground/environments/tasks/lsy_upstream/LICENSE` and shipped beside this asset for installed distributions.

The current physical course is `lsy_level0.xml`, converted from the pinned source TOML and gate/obstacle assets. Gate order, poses and physical sizes are retained. Nongeometric simulator parameters are in `assets/scenes/racing/lsy_native.yaml`.
