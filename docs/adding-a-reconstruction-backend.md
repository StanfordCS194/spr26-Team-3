# Adding a reconstruction backend

The reconstruction stage uses a plugin registry. Adding a new technique is
one new file plus a one-line import — no changes to routes, the UI, or
the database.

## The interface

`backend/src/features/reconstruction/backends/base.py`:

```python
class ReconstructionBackend(ABC):
    name: str                  # registry key
    requires_gpu: bool          # UI hint
    implemented: bool = True    # set False for stubs

    @abstractmethod
    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput: ...
```

`ReconstructionInput` gives you a `frames_dir` (with N sampled JPEGs) and
optional intrinsics. `ReconstructionOutput` asks for a `mesh_path` and
optional point cloud + camera poses + arbitrary backend metadata.

## Recipe

### 1. Create the file

```bash
touch backend/src/features/reconstruction/backends/my_new_backend.py
```

### 2. Write the backend

```python
# backend/src/features/reconstruction/backends/my_new_backend.py
from collections.abc import Callable
from pathlib import Path

from src.features.reconstruction.backends import register
from src.features.reconstruction.backends.base import (
    ReconstructionBackend,
    ReconstructionInput,
    ReconstructionOutput,
)


@register
class MyNewBackend(ReconstructionBackend):
    name = "my_new_backend"
    requires_gpu = True

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        out_dir.mkdir(parents=True, exist_ok=True)
        progress_cb(0.05, "starting")

        # ... your work here ...
        # frames = sorted(inp.frames_dir.glob("*.jpg"))
        # ... produce a mesh ...

        mesh_path = out_dir / "mesh.ply"
        mesh.export(str(mesh_path))

        progress_cb(0.98, "done")
        return ReconstructionOutput(
            mesh_path=mesh_path,
            backend_meta={"actual_backend": self.name},
        )
```

### 3. Register the import side-effect

Add your module to `backends/__init__.py` alongside the others:

```python
from src.features.reconstruction.backends import (  # noqa: E402, F401
    colmap,
    demo_fixture,
    depth_fusion,
    my_new_backend,    # ← new
    splat,
    vggt,
)
```

### 4. That's it

- The backend appears in `GET /api/reconstruction/backends` with
  `implemented: true`.
- The Reconstruct screen surfaces it in the picker grid automatically.
- The Inngest function and in-process thread worker dispatch to it via
  the registry — no router code touches your file.
- The `/validate` catalog runs the same six checks on your output mesh.

## Gracefully degrading when a dep is missing

If your backend needs a heavy dependency (CUDA, a large model weight),
fail loudly with a clear hint pointing at `demo_fixture` so contributors
without a GPU still get the rest of the pipeline:

```python
try:
    import vggt
except ImportError as e:
    raise RuntimeError(
        f"VGGT not installed: {e}. Pick the `demo_fixture` backend to "
        "demo the pipeline without GPU."
    ) from e
```

## Testing your backend

Add a unit test under `backend/tests/test_<your_backend>.py`. The
existing `test_e2e.py` covers the full pipeline against `demo_fixture`;
you can copy its shape and just swap the backend name to verify yours
works end-to-end (assumes your test environment has whatever deps the
backend needs).

## Naming

Plan future capability changes for each new backend so the work is
proposable in OpenSpec:

| Backend | Change ID |
|---|---|
| Gaussian Splatting | `worldscan-v2.1-splat` |
| COLMAP + MVS | `worldscan-v2.2-colmap` |
| Depth + TSDF | `worldscan-v2.3-depth-fusion` |
| Cloud-hosted execution | `worldscan-v2.4-cloud-reconstruction` |
| Stream / live | `worldscan-v2.5-stream-reconstruction` |
