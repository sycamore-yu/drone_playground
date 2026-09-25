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
import tempfile
from pathlib import Path, PurePosixPath


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


def compatible_main(show_metrics: bool = False):
    """Move native UI methods outside the lock; lock only direct MjData writes."""
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

    transformer = LockScope()
    tree = transformer.visit(tree)
    if (transformer.removed, transformer.added) != (1, 1):
        raise RuntimeError("Unexpected rscope source: refusing to apply an unverified lock change")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__dict__)
    state_class = original.ViewerState

    def state_factory():
        state = state_class()
        state.show_metrics = show_metrics
        return state

    namespace["ViewerState"] = state_factory
    exec(compile(tree, "<rscope-0.0.8-ui-lock-compatible>", "exec"), namespace)
    return namespace["main"], {"ui_locks_removed": 1, "state_write_locks_added": 1}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="rscope 原生查看器：安全缓存与 MuJoCo 3.14 界面锁兼容"
    )
    parser.add_argument("--directory", type=Path, help="本地导出目录")
    parser.add_argument("--ssh_to", help="username@host[:port]")
    parser.add_argument(
        "--ssh_key", type=Path, help="已有 SSH 私钥文件；省略时尝试现有代理/默认密钥"
    )
    parser.add_argument("--known_hosts", type=Path, help="已信任的 SSH 主机公钥清单")
    parser.add_argument("--remote_directory", default="/tmp/rscope/active_run")
    parser.add_argument("--cache_dir", type=Path, help="新建或空的本地缓存目录")
    parser.add_argument("--polling_interval", type=int, default=5)
    parser.add_argument("--show-metrics", action="store_true")
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
        if "@" not in args.ssh_to:
            parser.error("Use username@host[:port]")
        username, host = args.ssh_to.split("@", 1)
        port = 22
        if ":" in host:
            host, port_text = host.rsplit(":", 1)
            port = int(port_text)
        if not 1 <= port <= 65535:
            parser.error("Invalid SSH port")
        passphrase = None

        def connect(ssh):
            ssh.load_system_host_keys()
            if args.known_hosts:
                ssh.load_host_keys(str(args.known_hosts.expanduser()))
            ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
            ssh.connect(
                host,
                port=port,
                username=username,
                key_filename=str(args.ssh_key.expanduser()) if args.ssh_key else None,
                passphrase=passphrase,
                timeout=10,
                auth_timeout=10,
                banner_timeout=10,
            )
            native_open = ssh.open_sftp
            ssh.open_sftp = lambda: RemoteSFTP(native_open(), args.remote_directory)

        # Fail before starting background polling when credentials are unavailable.
        try:
            with paramiko.SSHClient() as preflight:
                connect(preflight)
                sftp = preflight.open_sftp()
                sftp.listdir(args.remote_directory)
                sftp.close()
        except paramiko.PasswordRequiredException:
            passphrase = getpass.getpass("SSH key passphrase: ")
            with paramiko.SSHClient() as preflight:
                connect(preflight)

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

    viewer_main, details = compatible_main(args.show_metrics)
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
