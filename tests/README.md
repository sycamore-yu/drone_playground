# Tests

Tests are grouped by scope:

- `unit/`: isolated components and numerical behavior.
- `integration/`: configuration, environments, learning, evaluation and runtime interfaces.
- `regression/`: fixed numerical baselines.
- `native/`: C++/gRPC interoperability fixtures.
- `fixtures/` and `helpers/`: shared test data and utilities.

From the repository root, after installing the dependencies:

```bash
JAX_PLATFORMS=cpu pixi run test
pixi run lint
```

Run a smaller set during development:

```bash
JAX_PLATFORMS=cpu pixi run test tests/unit
JAX_PLATFORMS=cpu pixi run test tests/integration
JAX_PLATFORMS=cpu pixi run test tests/regression
```
