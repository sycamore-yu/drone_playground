#!/usr/bin/env python3
"""Standalone rscope 0.0.8 client for MuJoCo 3.14, including Windows SSH viewing.

Install only rscope==0.0.8 and mujoco==3.14.0 on the viewing machine.
The native viewer, plots, keyboard handling, polling and file transfer are reused.
The installed packages remain unchanged. See docs/verification/p1-observation-checks.md.
"""

from __future__ import annotations

import argparse
import ast
import getpass
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np


class RemoteSFTP:
    """Separate the server's POSIX path from the client's local cache path."""

    def __init__(self, client, remote_root: str):
        self.client = client
        self.remote_root = str(PurePosixPath(remote_root))

    def listdir(self, _path="."):
        return self.client.listdir(self.remote_root)

    def get(self, remote, local, *args, **kwargs):
        name = str(remote).replace("\\", "/").rsplit("/", 1)[-1]
        return self.client.get(str(PurePosixPath(self.remote_root) / name), local, *args, **kwargs)

    def close(self):
        return self.client.close()


def _parse_ssh_g(output: str) -> dict[str, list[str]]:
    """Parse ``ssh -G`` output while preserving repeated configuration keys."""
    values: dict[str, list[str]] = {}
    for raw in output.splitlines():
        raw = raw.strip()
        if not raw or " " not in raw:
            continue
        key, value = raw.split(None, 1)
        values.setdefault(key.lower(), []).append(value.strip())
    return values


def resolve_ssh_target(spec: str) -> tuple[str, str, int, list[Path]]:
    """Resolve an OpenSSH alias or direct target with the system OpenSSH client."""
    ssh = shutil.which("ssh")
    if ssh:
        result = subprocess.run(
            [ssh, "-G", spec],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            cfg = _parse_ssh_g(result.stdout)
            host = cfg.get("hostname", [None])[0]
            user = cfg.get("user", [None])[0]
            port_text = cfg.get("port", ["22"])[0]
            if host and user:
                keys = []
                for raw in cfg.get("identityfile", []):
                    expanded = os.path.expandvars(os.path.expanduser(raw.strip('"')))
                    candidate = Path(expanded)
                    if candidate.exists():
                        keys.append(candidate)
                return user, host, int(port_text), keys

    if "@" not in spec:
        raise ValueError(
            "SSH alias resolution needs the system OpenSSH client (`ssh -G`). "
            "Use username@host[:port] or put ssh.exe on PATH."
        )
    user, host_port = spec.split("@", 1)
    port = 22
    host = host_port
    if ":" in host_port:
        host, port_text = host_port.rsplit(":", 1)
        port = int(port_text)
    return user, host, port, []


def load_private_key(paramiko, path: Path):
    """Load an OpenSSH private key and prompt once when it is encrypted."""
    key_classes = [paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey]
    password = None
    prompted = False
    errors = []
    for key_class in key_classes:
        try:
            return key_class.from_private_key_file(str(path), password=password)
        except paramiko.PasswordRequiredException:
            if not prompted:
                password = getpass.getpass(f"SSH key passphrase for {path}: ")
                prompted = True
                try:
                    return key_class.from_private_key_file(str(path), password=password)
                except paramiko.SSHException as exc:
                    errors.append(exc)
                    continue
            raise
        except paramiko.SSHException as exc:
            errors.append(exc)
    raise paramiko.SSHException(
        f"Could not load SSH private key {path}: " + "; ".join(str(error) for error in errors[-2:])
    )


def reference_points_from_rollout(rollout, env_index: int | None = None):
    """Recover the world-frame reference path from a selected or batched rollout."""
    metrics = getattr(rollout, "metrics", {})
    metric_names = ("reference_x", "reference_y", "reference_z")
    if all(name in metrics for name in metric_names):
        values = []
        for name in metric_names:
            value = np.asarray(metrics[name])
            if value.ndim == 2:
                if env_index is None or not 0 <= env_index < value.shape[1]:
                    return None
                value = value[:, env_index]
            values.append(value)
        points = np.stack(values, axis=-1)
        if points.ndim == 2 and points.shape[-1] == 3:
            return points

    obs = getattr(rollout, "obs", None)
    if isinstance(obs, dict):
        return None
    obs = np.asarray(obs)
    if obs.ndim == 3:
        if env_index is None or not 0 <= env_index < obs.shape[1]:
            return None
        obs = obs[:, env_index]
    if obs.ndim != 2 or obs.shape[-1] < 16:
        return None
    return obs[:, :3] + obs[:, 13:16]


def _reference_route_score(points: np.ndarray | None) -> int:
    if points is None or len(points) < 2:
        return -1
    points = np.asarray(points)
    finite = np.isfinite(points).all(axis=1)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    valid = finite[:-1] & finite[1:] & (segments > 1e-12)
    return int(valid.sum())


def update_reference_route(
    viewer,
    rollout,
    env_index: int | None = None,
    candidate_rollouts=None,
) -> int:
    """Draw the most complete matching reference trajectory in MuJoCo's user scene."""
    import mujoco

    scene = viewer.user_scn
    if scene is None:
        return 0
    scene.ngeom = 0
    points = reference_points_from_rollout(rollout)
    best_score = _reference_route_score(points)
    if candidate_rollouts is not None and env_index is not None:
        for candidate in candidate_rollouts:
            candidate_points = reference_points_from_rollout(candidate, env_index=env_index)
            score = _reference_route_score(candidate_points)
            if score > best_score:
                points, best_score = candidate_points, score
    if points is None or len(points) < 2:
        return 0

    rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    size = np.zeros(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    count = 0
    for start, end in zip(points[:-1], points[1:]):
        if count >= len(scene.geoms):
            break
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        if np.linalg.norm(end - start) <= 1e-12:
            continue
        geom = scene.geoms[count]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size, pos, mat, rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3.0, start, end)
        count += 1
    scene.ngeom = count
    return count


def compatible_main(show_metrics: bool = False, show_reference: bool = True):
    """Adapt native rscope for MuJoCo 3.14 and optionally draw task references."""
    import mujoco

    original = importlib.import_module("rscope.main")

    if importlib.metadata.version("rscope") != "0.0.8" or not mujoco.__version__.startswith(
        "3.14."
    ):
        raise RuntimeError("This verified client requires rscope 0.0.8 and MuJoCo 3.14.x")
    tree = ast.parse(inspect.getsource(original.main))

    class LockScope(ast.NodeTransformer):
        removed = 0
        added = 0
        reference_added = 0

        def visit_With(self, node):
            self.generic_visit(node)
            for item in node.items:
                call = item.context_expr
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "viewer"
                    and call.func.attr == "lock"
                ):
                    self.removed += 1
                    return node.body
            return node

        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "advance_rollout"
            ):
                self.added += 1
                lock = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="viewer", ctx=ast.Load()), attr="lock", ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[],
                )
                return ast.copy_location(
                    ast.With(
                        items=[ast.withitem(context_expr=lock)], body=[node], type_comment=None
                    ),
                    node,
                )
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "replay_len"
                and ast.unparse(node.value) == "cur_rollout.qpos.shape[0]"
            ):
                self.reference_added += 1
                lock = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="viewer", ctx=ast.Load()), attr="lock", ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[],
                )
                update = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="_update_reference_route", ctx=ast.Load()),
                        args=[
                            ast.Name(id="viewer", ctx=ast.Load()),
                            ast.Name(id="cur_rollout", ctx=ast.Load()),
                            ast.Attribute(
                                value=ast.Name(id="viewer_state", ctx=ast.Load()),
                                attr="cur_env",
                                ctx=ast.Load(),
                            ),
                            ast.Attribute(
                                value=ast.Name(id="rollout", ctx=ast.Load()),
                                attr="rollouts",
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    )
                )
                return [
                    node,
                    ast.copy_location(
                        ast.With(
                            items=[ast.withitem(context_expr=lock)],
                            body=[update],
                            type_comment=None,
                        ),
                        node,
                    ),
                ]
            return node

    transformer = LockScope()
    tree = transformer.visit(tree)
    if (transformer.removed, transformer.added, transformer.reference_added) != (1, 1, 1):
        raise RuntimeError("Unexpected rscope source: refusing to apply an unverified lock change")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__dict__)
    state_class = original.ViewerState

    def state_factory():
        state = state_class()
        state.show_metrics = show_metrics
        return state

    namespace["ViewerState"] = state_factory
    namespace["_update_reference_route"] = (
        update_reference_route if show_reference else lambda *_args, **_kwargs: 0
    )
    exec(compile(tree, "<rscope-0.0.8-ui-lock-compatible>", "exec"), namespace)
    return namespace["main"], {
        "ui_locks_removed": 1,
        "state_write_locks_added": 1,
        "reference_updates_added": 1,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="rscope 原生查看器：安全缓存与 MuJoCo 3.14 界面锁兼容"
    )
    parser.add_argument("--directory", type=Path, help="本地导出目录")
    parser.add_argument("--ssh_to", help="OpenSSH Host alias or username@host[:port]")
    parser.add_argument(
        "--ssh_key", type=Path, help="已有 SSH 私钥文件；省略时尝试现有代理/默认密钥"
    )
    parser.add_argument("--known_hosts", type=Path, help="已信任的 SSH 主机公钥清单")
    parser.add_argument("--remote_directory", default="/tmp/rscope/active_run")
    parser.add_argument("--cache_dir", type=Path, help="新建或空的本地缓存目录")
    parser.add_argument("--polling_interval", type=int, default=5)
    parser.add_argument("--show-metrics", action="store_true")
    parser.add_argument(
        "--hide-reference",
        action="store_true",
        help="隐藏 tracking rollout 的 3D 参考轨迹；默认显示",
    )
    args = parser.parse_args(argv)
    if args.ssh_to and args.directory:
        parser.error("Choose a local directory or an SSH source")
    if args.polling_interval < 1:
        parser.error("polling_interval must be positive")

    import rscope.config as config
    import rscope.model_loader as model_loader
    import rscope.ssh_utils as ssh_utils

    if args.ssh_to:
        import paramiko

        if not PurePosixPath(args.remote_directory).is_absolute():
            parser.error("remote_directory must be an absolute Linux path")
        try:
            username, host, port, configured_keys = resolve_ssh_target(args.ssh_to)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            parser.error(str(exc))
        if not 1 <= port <= 65535:
            parser.error("Invalid SSH port")

        key_path = args.ssh_key.expanduser() if args.ssh_key else None
        if key_path is None and configured_keys:
            key_path = configured_keys[0]
        private_key = load_private_key(paramiko, key_path) if key_path else None
        known_hosts = (
            args.known_hosts.expanduser()
            if args.known_hosts
            else Path.home() / ".ssh" / "known_hosts"
        )

        def connect(ssh):
            ssh.load_system_host_keys()
            if known_hosts.is_file():
                ssh.load_host_keys(str(known_hosts))
            ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
            ssh.connect(
                host,
                port=port,
                username=username,
                pkey=private_key,
                allow_agent=True,
                look_for_keys=private_key is None,
                timeout=10,
                auth_timeout=10,
                banner_timeout=10,
            )
            native_open = ssh.open_sftp
            ssh.open_sftp = lambda: RemoteSFTP(native_open(), args.remote_directory)

        # Fail before starting background polling when credentials are unavailable.
        with paramiko.SSHClient() as preflight:
            connect(preflight)
            sftp = preflight.open_sftp()
            sftp.listdir(args.remote_directory)
            sftp.close()

        cache = (
            args.cache_dir.expanduser().resolve()
            if args.cache_dir
            else Path(tempfile.mkdtemp(prefix="rscope-client-"))
        )
        if cache.exists() and any(cache.iterdir()):
            parser.error("Use an empty cache directory; existing files are preserved")
        cache.mkdir(parents=True, exist_ok=True)
        config.BASE_PATH = cache
        config.TEMP_PATH = cache.parent / (cache.name + "-transfer")
        config.META_PATH = cache / "rscope_meta.pkl"
        ssh_utils.ssh_connect = connect
        model_loader.ssh_connect = connect
    else:
        directory = (args.directory or Path("/tmp/rscope/active_run")).expanduser().resolve()
        if not (directory / "rscope_meta.pkl").is_file():
            parser.error(f"No rscope metadata in {directory}")
        config.BASE_PATH = directory
        config.TEMP_PATH = Path(tempfile.mkdtemp(prefix="rscope-local-transfer-"))
        config.META_PATH = directory / "rscope_meta.pkl"

    viewer_main, details = compatible_main(
        args.show_metrics, show_reference=not args.hide_reference
    )
    print(
        json.dumps(
            {
                "local_cache": str(config.BASE_PATH),
                "ssh": bool(args.ssh_to),
                "remote_directory": args.remote_directory if args.ssh_to else None,
                "compatibility": details,
            }
        ),
        flush=True,
    )
    viewer_main(ssh_enabled=bool(args.ssh_to), polling_interval=args.polling_interval)


if __name__ == "__main__":
    main()
