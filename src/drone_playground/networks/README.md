# Networks

`pointnet_gru.py` implements a point-wise PointNet encoder with additive state fusion and GRU memory. `cnn_gru.py` implements a depth CNN with GRU memory and the recorded acceleration/velocity output decoding. Named Flax parameter layers and dimensions are retained.

`perception.py` supplies shared sensor fusion; `environments/observations/range.py` is deterministic polar-range feature extraction, not a learned network. `policies.py` constructs the selected Brax policy/value networks and loads frozen policies. `actions/decoders.py` decodes explicit geometric outputs.

Configuration names describe architecture and initialization: `pointnet_gru`, `depth_cnn_gru`, `mlp_32`, `mlp_64`, `gaussian_mlp_64`, and sensor-fusion variants. Paper citations and experiment-specific hyperparameters belong to experiment presets and provenance documentation.
