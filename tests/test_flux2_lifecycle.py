from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tailsteer.models.flux2.model import Flux2Model  # noqa: E402


def test_flux2_unload_drops_all_gpu_module_references_without_cpu_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ModuleThatMustNotMove:
        def to(self, *_args, **_kwargs):
            raise AssertionError("unload must not copy FLUX.2 weights to CPU")

    transformer = ModuleThatMustNotMove()
    text_encoder = ModuleThatMustNotMove()
    vae = ModuleThatMustNotMove()
    pipe = SimpleNamespace(
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
    )
    model = Flux2Model.__new__(Flux2Model)
    model.pipe = pipe
    model.refined_transformer = transformer
    model.pretrained_transformer = transformer
    cache_clears = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_clears.append(True))

    model.unload()

    assert model.pipe is None
    assert model.refined_transformer is None
    assert model.pretrained_transformer is None
    assert pipe.transformer is None
    assert pipe.text_encoder is None
    assert pipe.vae is None
    assert cache_clears == [True]

    model.unload()
    assert cache_clears == [True, True]
