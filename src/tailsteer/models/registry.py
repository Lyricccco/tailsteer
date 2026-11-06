"""Construct one of the four supported backbones."""

from importlib import import_module

MODELS = {
    "flux1_kontext_dev": ("flux1_kontext.model", "KontextModel"),
    "flux2_klein_9b": ("flux2.model", "Flux2Model"),
    "qwen_image_21": ("qwen_image_21.model", "QwenImage21Model"),
    "sensenova_u1_5_8b_mot": ("sensenova.model", "SenseNovaModel"),
}


def create_model(config, **load_options):
    name = config["name"]
    if name not in MODELS:
        raise ValueError(f"Unsupported backbone: {name}")
    module, cls = MODELS[name]
    return getattr(import_module(f"tailsteer.models.{module}"), cls).load(
        config, **load_options
    )
