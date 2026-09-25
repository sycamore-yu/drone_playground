"""Compatibility at the native rscope UI and remote/local filesystem interfaces."""

import importlib.util
from pathlib import Path


def client_module():
    path = Path(__file__).parents[1] / "scripts/rscope_client.py"
    spec = importlib.util.spec_from_file_location("rscope_client_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sftp_remote_names_remain_posix_and_local_paths_are_preserved():
    client = client_module()

    class Transport:
        def listdir(self, path):
            return [path]

        def get(self, remote, local, **kwargs):
            return remote, local

    proxy = client.RemoteSFTP(Transport(), "/tmp/rscope/active_run")
    local = r"C:\Users\user\AppData\Local\Temp\rscope\case.mj_unroll"
    assert proxy.listdir(r"C:\Temp\rscope") == ["/tmp/rscope/active_run"]
    assert proxy.get(local, local) == ("/tmp/rscope/active_run/case.mj_unroll", local)
    assert (
        proxy.get("/local/cache/rscope_meta.pkl", "/local/cache/rscope_meta.pkl")[0]
        == "/tmp/rscope/active_run/rscope_meta.pkl"
    )


def test_ui_lock_fix_is_narrow_version_checked_and_preserves_the_package():
    import importlib
    import inspect

    original = importlib.import_module("rscope.main")

    client = client_module()
    source_before = inspect.getsource(original.main)
    main, details = client.compatible_main(show_metrics=True)
    assert callable(main)
    assert details == {"ui_locks_removed": 1, "state_write_locks_added": 1}
    assert inspect.getsource(original.main) == source_before
    state = main.__globals__["ViewerState"]()
    assert state.show_metrics
