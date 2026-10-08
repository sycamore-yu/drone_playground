# Native protocol interoperability fixture

This executable is a synthetic C++ peer, not a planner or a deployment SDK. It
implements the gRPC Algorithm::Service generated from the package's one protocol.
Tests cover physical values, reset, time, invalid requests and late outputs.
Docker is not needed.

From the repository root:

```bash
pixi install --manifest-path tests/native/pixi.toml
prefix="$PWD/tests/native/.pixi/envs/default"
export PATH="$prefix/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S tests/native -B tmp/native-interop -G Ninja \
  -DCMAKE_PREFIX_PATH="$prefix" \
  -DCMAKE_CXX_COMPILER="$prefix/bin/x86_64-conda-linux-gnu-c++"
cmake --build tmp/native-interop -j 2
DRONE_NATIVE_TEST_BINARY="$PWD/tmp/native-interop/interop_server" \
  pixi run python -m pytest tests/integration/runtime/test_native_typed_service.py -q
```

There is no installable SDK library or second custom Algorithm base class. A real
solver implements the same generated service and supplies its actual conversions.
