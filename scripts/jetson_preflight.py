"""Read-only portable handoff checks; no installs, camera starts or training."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sqlite3
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def under(root, relative):
    candidate = (root / relative).resolve()
    candidate.relative_to(root.resolve())
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=Path(__file__).resolve().parent,
                        help="External handoff folder containing transfer_manifest.json.")
    parser.add_argument("--files-only", action="store_true",
                        help="Verify files without importing optional ML/runtime packages.")
    args = parser.parse_args(argv)
    folder = args.kit.resolve()
    ssd = folder.parent
    manifest = json.loads((folder / "transfer_manifest.json").read_text(encoding="utf-8"))
    errors = []
    print("SSD:", ssd)
    print("Source:", manifest["source_commit"])
    for artifact in manifest["handoff_artifacts"]:
        path = under(folder, artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            errors.append("Artifact missing/hash mismatch: " + artifact["path"])
        else:
            print("HASH OK:", artifact["path"])
    for recording in manifest["recordings"]:
        path = under(ssd, recording["path"])
        if not path.is_file() or path.stat().st_size != recording["bytes"]:
            errors.append("Recording missing/size mismatch: " + recording["path"])
            continue
        try:
            con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
            try:
                con.execute("PRAGMA query_only=ON")
                topics = con.execute("SELECT id,name FROM topics").fetchall()
                for stream in ("Color_0/image/data", "Depth_0/image/data"):
                    ids = [tid for tid, name in topics if name.endswith(stream)]
                    if len(ids) != 1 or con.execute(
                        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1",
                        (ids[0],),
                    ).fetchone() is None:
                        raise ValueError("missing/empty " + stream)
            finally:
                con.close()
            print("SIZE/TOPICS OK:", recording["path"])
        except Exception as exc:
            errors.append(recording["path"] + ": " + str(exc))
    print("\nRaw hashes not computed: size/topic checks do not prove full-file integrity.")
    if not args.files_only:
        print("HOST:", platform.system(), platform.machine(), platform.python_version())
        for name in ("/proc/device-tree/model", "/etc/nv_tegra_release"):
            path = Path(name)
            if path.is_file():
                print(name + ":", path.read_text(errors="replace").replace("\x00", "").strip())
        if sys.version_info[:2] != (3, 10):
            print("SETUP NEEDED: source requires Python >=3.10,<3.11; do not replace system Python blindly.")
        for module, distribution in (
            ("numpy", "numpy"), ("cv2", "opencv-python"), ("sklearn", "scikit-learn"),
            ("rtmlib", "rtmlib"), ("pyrealsense2", "pyrealsense2"),
            ("onnxruntime", "onnxruntime"), ("typer", "typer"),
            ("pydantic", "pydantic"), ("yaml", "PyYAML"), ("zstandard", "zstandard"),
        ):
            try:
                loaded = importlib.import_module(module)
                try:
                    version = importlib.metadata.version(distribution)
                except importlib.metadata.PackageNotFoundError:
                    version = getattr(loaded, "__version__", "system/vendor package")
                print("IMPORT OK:", module, version)
                if module == "onnxruntime":
                    providers = loaded.get_available_providers()
                    print("ORT providers:", providers)
                    if "CUDAExecutionProvider" not in providers:
                        print("SETUP NEEDED: CUDAExecutionProvider unavailable; do not claim GPU acceleration.")
            except Exception as exc:
                print("SETUP NEEDED:", module, type(exc).__name__, str(exc))
    print("\nGPU inference, .db3 playback and alignment still require an actual Jetson smoke test.")
    print("No data changed; no training or collection started.")
    if errors:
        print("\nHANDOFF CHECK FAILED:")
        for error in errors:
            print(" -", error)
        return 1
    print("HANDOFF FILE CHECKS PASSED (not runtime or clinical validation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
