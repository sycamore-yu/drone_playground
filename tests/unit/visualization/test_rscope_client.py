"""Compatibility at the native rscope UI and remote/local filesystem interfaces."""

from types import SimpleNamespace

import numpy as np


def client_module():
    from drone_playground.visualization import rscope_client as module

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


def test_openssh_alias_resolution_preserves_user_host_port_and_existing_identity(
    monkeypatch, tmp_path
):
    client = client_module()
    identity = tmp_path / "id_ed25519"
    identity.write_text("placeholder")
    output = "\n".join(
        [
            "user tong",
            "hostname 100.92.244.71",
            "port 22",
            f"identityfile {identity}",
        ]
    )
    monkeypatch.setattr(client.shutil, "which", lambda name: "ssh.exe" if name == "ssh" else None)
    monkeypatch.setattr(
        client.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    user, host, port, identities = client.resolve_ssh_target("lab-gpu")

    assert (user, host, port) == ("tong", "100.92.244.71", 22)
    assert identities == [identity]


def test_ui_lock_fix_is_narrow_version_checked_and_preserves_the_package():
    import importlib
    import inspect

    original = importlib.import_module("rscope.main")

    client = client_module()
    source_before = inspect.getsource(original.main)
    main, details = client.compatible_main(show_metrics=True, show_reference=True)
    assert callable(main)
    assert details == {
        "ui_locks_removed": 1,
        "state_write_locks_added": 1,
        "reference_updates_added": 1,
    }
    assert inspect.getsource(original.main) == source_before
    state = main.__globals__["ViewerState"]()
    assert state.show_metrics
    assert "_update_reference_route" in main.__code__.co_names

    hidden_main, hidden_details = client.compatible_main(show_reference=False)
    assert hidden_details["reference_updates_added"] == 1
    assert hidden_main.__globals__["_update_reference_route"](None, None) == 0


def test_reference_points_are_reconstructed_from_tracking_observation():
    client = client_module()
    obs = np.zeros((4, 43), dtype=np.float32)
    positions = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.1], [0.2, 0.0, 1.2], [0.3, 0.0, 1.3]],
        dtype=np.float32,
    )
    references = np.array(
        [[1.0, 0.0, 1.0], [0.8, 0.0, 1.2], [0.2, 0.0, 1.5], [-0.4, 0.0, 1.2]],
        dtype=np.float32,
    )
    obs[:, :3] = positions
    obs[:, 13:16] = references - positions

    recovered = client.reference_points_from_rollout(
        type("Rollout", (), {"obs": obs, "metrics": {}})()
    )

    np.testing.assert_allclose(recovered, references, atol=1e-7)


def test_reference_route_populates_user_scene_with_red_line_segments():
    import mujoco

    client = client_module()
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    scene = mujoco.MjvScene(model, maxgeom=32)
    viewer = type("Viewer", (), {"user_scn": scene})()
    obs = np.zeros((3, 43), dtype=np.float32)
    obs[:, :3] = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]
    obs[:, 13:16] = [[0.0, 0.0, 0.0], [0.9, 0.0, 0.2], [1.8, 0.0, 0.0]]
    rollout = type("Rollout", (), {"obs": obs, "metrics": {}})()

    count = client.update_reference_route(viewer, rollout)

    assert count == 2
    assert scene.ngeom == 2
    assert scene.geoms[0].type == mujoco.mjtGeom.mjGEOM_LINE
    np.testing.assert_allclose(scene.geoms[0].rgba, [1.0, 0.0, 0.0, 1.0])


def test_reference_route_uses_most_complete_matching_case_across_checkpoints():
    import mujoco

    client = client_module()
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    scene = mujoco.MjvScene(model, maxgeom=32)
    viewer = type("Viewer", (), {"user_scn": scene})()

    partial = np.zeros((5, 43), dtype=np.float32)
    partial[:, :3] = np.array([[0, 0, 1], [0.1, 0, 1], [0.2, 0, 1], [0.2, 0, 1], [0.2, 0, 1]])
    current = type("Rollout", (), {"obs": partial, "metrics": {}})()

    complete = np.zeros((5, 2, 43), dtype=np.float32)
    references = np.array(
        [[0, 0, 1], [0.5, 0, 1.4], [1, 0, 1], [0.5, 0, 0.6], [0, 0, 1]],
        dtype=np.float32,
    )
    complete[:, 0, :3] = references
    complete[:, 1, :3] = references + np.array([0, 1, 0], dtype=np.float32)
    full_rollout = type("FullRollout", (), {"obs": complete, "metrics": {}})()

    count = client.update_reference_route(
        viewer,
        current,
        env_index=0,
        candidate_rollouts=[full_rollout],
    )

    assert count == 4
    assert scene.ngeom == 4
