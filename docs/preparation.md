# Preparation

## Environment and upstream code

Start in a clean Python 3.10 virtual environment, install `PyYAML`, and run:

```bash
python scripts/setup.py flux2-klein-base-9b
```

Replace the backbone with `flux1-kontext-dev`, `qwen-image-2.1` or `sensenova-u1.5-8b-mot`. Qwen uses a separate environment. Dependency versions and source commits are recorded in `requirements/` and `upstream.lock`. The setup script installs the current repository in editable mode; keep the checkout alongside the installation. CUDA/PyTorch wheels must match your driver. Flash Attention is optional unless required by the chosen upstream runtime.

Kontext uses the pinned Black Forest Labs implementation. FLUX.2 and Qwen use their pinned Diffusers pipelines. SenseNova imports its pinned official source. Upstream code is downloaded to the ignored `upstream/` directory, rather than copied into TailSteer's source tree.

## Model weights

```bash
hf auth login
tailsteer prepare model --backbone flux2-klein-base-9b
```

The command downloads the full official model repository. To use existing weights, pass `--model-path /path/to/model` to cache, train and infer. For Kontext or SenseNova, `--model-code /path/to/official/source` selects an existing source checkout. Use `--revision` on download commands to pin a particular snapshot.

The default roots can be moved without editing configurations:

```bash
export TAILSTEER_MODEL_ROOT=/storage/models
export TAILSTEER_DATA_ROOT=/storage/data
export TAILSTEER_CACHE_ROOT=/storage/cache
export TAILSTEER_RUN_ROOT=/storage/runs
```

## TailSteer source data

[**TailSteer-Source**](https://huggingface.co/datasets/Lyricccco/TailSteer-Source) contains the source images and instruction manifest. **Preservation-Aware Trajectory Cache** refers to the outputs generated from them by a particular base model; these are not interchangeable datasets.

```bash
tailsteer prepare training-source
```

The downloaded layout is:

```text
data/training/
  sources.jsonl
  images/
    example.jpg
```

One JSONL record defines one independent editing trajectory:

```json
{"id":"example-local-1","source_id":"example","image":"images/example.jpg","prompt":"Make the small mug blue. Keep everything else unchanged.","task":"local","seed":0}
{"id":"example-identity","source_id":"example","image":"images/example.jpg","prompt":"Keep the image exactly unchanged.","task":"identity","seed":0}
```

`id` must be unique; `source_id` groups instructions derived from one image so the train/validation split does not leak that image across partitions. Paths are relative to the source dataset root. Each local edit is a separate rollout from the source; these records are not successive editing turns. A source image can have multiple local-edit instructions and an identity instruction. You can provide another manifest with `--sources` and `--image-root`.

The cache captures the model's final result, selected intermediate states, conditioning, schedule and keep masks. Numeric filtering is model specific. Local-edit and identity examples are stored together; the paper configuration uses the `dilate5` mask and splits by source identity.

## Add & Preserve dataset

```bash
tailsteer prepare add-preserve
```

Expected layout:

```text
data/add-preserve/
  tasks.json
  manifest.json
  images/...
  masks/...
```

The dataset contains the frozen source image with its white interior/red boundary, ten English editing instructions, seeds and the exact outside-box mask. The manifest also stores object labels and asset hashes. Keep geometry and filenames unchanged when downloading. See [dataset formats](dataset-formats.md) for the publication contract.

The default dataset revisions are fixed in the CLI. `--dataset-repo OWNER/DATASET` selects a custom repository (its `main` revision unless overridden), and `--revision REVISION` selects an explicit snapshot. Publication formats are documented in [dataset formats](dataset-formats.md).
