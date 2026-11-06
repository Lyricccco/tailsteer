"""Read training exports and complete, native-layout TailSteer repositories."""

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path


def checkpoint_manifest(root):
    root = Path(root)
    release = root / "tailrefine_release.json"
    if not release.is_file():
        if not (root / "COMPLETE").is_file():
            raise FileNotFoundError(f"Incomplete inference export: {root}")
        return json.loads((root / "manifest.json").read_text())
    metadata = json.loads(release.read_text())
    model = metadata["model"]
    formats = {
        "flux2_klein_9b": "tailsteer_flux2_transformer",
        "qwen_image_21": "tailsteer_qwen_image_21_transformer",
        "sensenova_u1_5_8b_mot": "tailsteer_sensenova_delta",
    }
    sensenova = model == "sensenova_u1_5_8b_mot"
    weights = (
        "model.safetensors"
        if sensenova
        else "transformer/diffusion_pytorch_model.safetensors"
    )
    if (root / (weights + ".index.json")).is_file():
        weights += ".index.json"
    if not (root / weights).is_file():
        raise FileNotFoundError(root / weights)
    return {
        **metadata,
        "schema_version": 3 if sensenova else 1,
        "format": formats[model],
        "weights": weights,
        "weight_dtype": "bfloat16",
        "weight_prefix": "model." if sensenova else "",
        "variant": "sensenova-u1.5-8b-mot" if sensenova else model,
        "trainable": {
            "policy": metadata["trainable_policy"],
            "master_dtype": "model",
            "conditioning_gradients": sensenova,
            "reference_vision_model": False,
            "prefix_kv_adapter": {"enabled": False},
        },
    }


@contextmanager
def open_weights(path, *, prefix=""):
    """Expose the same tensor reader for single files and native shards."""
    from safetensors import safe_open

    path = Path(path)
    with ExitStack() as stack:
        if path.name.endswith(".index.json"):
            mapping = json.loads(path.read_text())["weight_map"]
        else:
            handle = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            mapping = dict.fromkeys(handle.keys(), path.name)
        handles = {} if path.name.endswith(".index.json") else {path.name: handle}

        class Reader:
            def keys(self):
                return [prefix + key for key in mapping]

            def get_tensor(self, name):
                if not name.startswith(prefix):
                    raise KeyError(name)
                key = name[len(prefix) :]
                shard = mapping[key]
                if shard not in handles:
                    handles[shard] = stack.enter_context(
                        safe_open(path.parent / shard, framework="pt", device="cpu")
                    )
                return handles[shard].get_tensor(key)

        yield Reader()
