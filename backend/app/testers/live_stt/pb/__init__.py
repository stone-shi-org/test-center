"""Protobuf and gRPC stub loader and build-time codegen for livestt.v1.asr.

The vendored schema lives at ``proto/livestt/v1/asr.proto``.
Python stubs (``asr_pb2.py``, ``asr_pb2_grpc.py``, ``asr_pb2.pyi``) are
generated in the build via ``scripts/gen_proto.sh`` (invoked by ``Dockerfile``
and ``backend/setup.py``). If imported in a clean working tree before the build
script has been run, ``ensure_proto_stubs()`` compiles the vendored ``.proto``
on the fly via ``grpc_tools.protoc``.

Because ``grpc_tools.protoc`` emits ``from livestt.v1 import asr_pb2`` inside
``asr_pb2_grpc.py``, this directory inserts itself onto ``sys.path`` on import.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType

_PB_DIR = Path(__file__).resolve().parent
if str(_PB_DIR) not in sys.path:
    sys.path.insert(0, str(_PB_DIR))


def find_proto_root() -> Path:
    """Locate the directory containing vendored ``livestt/v1/asr.proto``."""
    candidates = [
        _PB_DIR.parents[4] / "proto",  # <repo_root>/proto
        _PB_DIR.parents[3] / "proto",  # <backend_root>/proto
        Path.cwd() / "proto",
        Path.cwd().parent / "proto",
    ]
    for candidate in candidates:
        proto_file = candidate / "livestt" / "v1" / "asr.proto"
        if proto_file.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Vendored proto/livestt/v1/asr.proto not found; searched: "
        + ", ".join(str(c) for c in candidates)
    )


def generate_proto_stubs(
    proto_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Compile ``proto/livestt/v1/asr.proto`` into Python protobuf and gRPC stubs."""
    import grpc_tools
    from grpc_tools import protoc

    resolved_proto_root = (proto_root or find_proto_root()).resolve()
    resolved_out = (output_dir or _PB_DIR).resolve()
    proto_rel = Path("livestt") / "v1" / "asr.proto"
    proto_file = resolved_proto_root / proto_rel
    if not proto_file.is_file():
        raise FileNotFoundError(f"Proto file does not exist: {proto_file}")

    resolved_out.mkdir(parents=True, exist_ok=True)
    grpc_tools_proto_include = str((Path(grpc_tools.__file__).resolve().parent / "_proto"))

    args = [
        "grpc_tools.protoc",
        f"-I{resolved_proto_root}",
        f"-I{grpc_tools_proto_include}",
        f"--python_out={resolved_out}",
        f"--grpc_python_out={resolved_out}",
        f"--pyi_out={resolved_out}",
        str(proto_file),
    ]
    rc = protoc.main(args)
    if rc != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {rc} for {proto_file}")

    # Create package __init__.py markers for mirrored directories
    livestt_dir = resolved_out / "livestt"
    v1_dir = livestt_dir / "v1"
    for pkg_dir in (livestt_dir, v1_dir):
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Generated package namespace."""\n', encoding="utf-8")

    importlib.invalidate_caches()
    return v1_dir


def ensure_proto_stubs(output_dir: Path | None = None) -> Path:
    """Ensure ``livestt/v1/asr_pb2.py`` and ``asr_pb2_grpc.py`` exist and are up to date."""
    resolved_out = (output_dir or _PB_DIR).resolve()
    v1_dir = resolved_out / "livestt" / "v1"
    pb2_file = v1_dir / "asr_pb2.py"
    pb2_grpc_file = v1_dir / "asr_pb2_grpc.py"

    needs_codegen = not (pb2_file.is_file() and pb2_grpc_file.is_file())
    if not needs_codegen:
        try:
            proto_file = find_proto_root() / "livestt" / "v1" / "asr.proto"
            if proto_file.stat().st_mtime > min(pb2_file.stat().st_mtime, pb2_grpc_file.stat().st_mtime):
                needs_codegen = True
        except FileNotFoundError:
            pass

    if needs_codegen:
        generate_proto_stubs(output_dir=resolved_out)

    if str(resolved_out) not in sys.path:
        sys.path.insert(0, str(resolved_out))

    return v1_dir


def load_asr_stubs() -> tuple[ModuleType, ModuleType]:
    """Ensure stubs exist and return ``(asr_pb2, asr_pb2_grpc)``."""
    ensure_proto_stubs()
    from livestt.v1 import asr_pb2 as _asr_pb2
    from livestt.v1 import asr_pb2_grpc as _asr_pb2_grpc

    return _asr_pb2, _asr_pb2_grpc


os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")
asr_pb2, asr_pb2_grpc = load_asr_stubs()

__all__ = [
    "asr_pb2",
    "asr_pb2_grpc",
    "ensure_proto_stubs",
    "find_proto_root",
    "generate_proto_stubs",
    "load_asr_stubs",
]
