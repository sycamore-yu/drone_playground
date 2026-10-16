# Experiments

Hydra recipes under `src/drone_playground/configs/` are the configuration source.
Use the [Command Builder](command-builder.html) for supported release combinations.
The public recipes are `tracking`, `racing`, `oa_static_depth`, `oa_static_lidar`,
`oa_dynamic_depth`, and `oa_dynamic_lidar`, with `learning=bptt`, `shac`, or `ppo`.

Each experiment records its source identity, resolved configuration, random seed,
interaction budget, checkpoint, and complete evaluation outcomes. Start from
random weights for independent-seed comparisons. Explicit resumption retains the
original state and counters; it is not a new independent seed.

Task defines goals, rewards, and terminal conditions. Method defines Actor and
action units. Controller defines trajectory tracking. Learning defines the update
algorithm, auxiliary objective, and optional backward dynamics. Existing
paper-specific reward and observation recipes remain explicit experimental
controls; their upstream names identify published methods, not release claims.

Keep candidate-search cohorts separate from final test cohorts. Freeze the
quality criterion before running a convergence experiment. Report all attempts,
including failures and incomplete runs, with their original protocol. Stored
historical results cannot establish convergence of a new source revision.
