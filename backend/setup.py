"""Setuptools build hook to generate Python gRPC stubs from vendored proto/livestt/v1/asr.proto."""

from __future__ import annotations

from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class BuildPyWithProto(_build_py):
    """Run grpc_tools.protoc stub codegen before building Python modules."""

    def run(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        proto_root = repo_root / "proto"
        output_dir = Path(__file__).resolve().parent / "app" / "testers" / "live_stt" / "pb"
        if (proto_root / "livestt" / "v1" / "asr.proto").is_file():
            try:
                import grpc_tools
                from grpc_tools import protoc

                output_dir.mkdir(parents=True, exist_ok=True)
                grpc_include = str(Path(grpc_tools.__file__).resolve().parent / "_proto")
                rc = protoc.main(
                    [
                        "grpc_tools.protoc",
                        f"-I{proto_root}",
                        f"-I{grpc_include}",
                        f"--python_out={output_dir}",
                        f"--grpc_python_out={output_dir}",
                        f"--pyi_out={output_dir}",
                        str(proto_root / "livestt" / "v1" / "asr.proto"),
                    ]
                )
                if rc != 0:
                    raise RuntimeError(f"grpc_tools.protoc exited with status {rc}")
                for pkg_dir in (output_dir / "livestt", output_dir / "livestt" / "v1"):
                    init_file = pkg_dir / "__init__.py"
                    if not init_file.exists():
                        init_file.write_text('"""Generated package namespace."""\n', encoding="utf-8")
            except ImportError:
                pass
        super().run()


setup(cmdclass={"build_py": BuildPyWithProto})
