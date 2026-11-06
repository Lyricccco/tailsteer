# Dataset publication formats

Downloadable datasets contain relative asset paths and UTF-8 JSON. The CLI defaults to verified snapshots of `Lyricccco/TailSteer-Source` and `Lyricccco/Add-and-Preserve`. Custom IDs and revisions remain supported.

## TailSteer-Source

Recommended repository name: `TailSteer-Source`. Include a dataset card, `sources.jsonl` and the referenced image files. Each row has:

| Field | Meaning |
|---|---|
| `id` | Unique trajectory request ID |
| `source_id` | Stable original image ID, shared by all its instructions |
| `image` | Relative source-image path |
| `prompt` | Editing instruction |
| `task` | `local` or `identity` |
| `seed` | Integer generation seed |

Describe source provenance and image licenses in the dataset card. The cache command consumes this manifest without requiring target edited images or hand-drawn training masks.

Cache outputs are model dependent. If released separately, name them `TailSteer-Cache-<backbone>` and preserve their manifests, schedules and grouped tensor shards. The source-image dataset should not be named as if it already contains generated trajectory states.

## Add & Preserve

Recommended dataset title: **Add & Preserve**. The command-line identifier is `add-preserve`. Include `tasks.json`, `manifest.json`, all decorated source images, and outside-box masks.

`tasks.json` contains:

```json
{"tasks":[{"id":"example","image":"images/example.png","chain_rounds":true,"rounds":[{"prompt":"Add a small apple only inside the red rectangle. Keep everything outside unchanged.","seed":42}]}]}
```

The released benchmark has ten fixed English instructions per image. The snippet only illustrates the format; do not substitute it for the frozen benchmark prompts.

`manifest.json` contains `schema_version`, `name`, `prompt_language`, `rounds`, `source_decoration` and `items`. Each item requires:

```json
{
  "id":"example",
  "category":"photo",
  "image":"images/example.png",
  "keep_mask":"masks/example.png",
  "image_sha256":"sha256-of-image-file",
  "keep_mask_sha256":"sha256-of-mask-file",
  "size":[1024,768],
  "bbox":[256,192,768,576],
  "rounds":[{"object":"a small apple","prompt":"the full frozen instruction","seed":42}]
}
```

Coordinates are `[left, top, right, bottom]` in the released image's pixel space, with exclusive right/bottom bounds. Mask pixels outside the allowed rectangle are nonzero; interior pixels are zero. `source_decoration.width_pixels` specifies the inward red-border width, used to crop the COA judge input. The `image`, prompt and seed must agree exactly between the manifest and tasks file. Item rounds additionally contain the object label for cumulative judging.

Remove machine-specific selection paths and private notes from publication metadata. Do not change image bytes, masks, bounding boxes, prompts or seeds when making the metadata portable: doing so would define a different benchmark. Keep image provenance and licensing in the dataset card rather than absolute filesystem paths.

## Publishing a compatible dataset

Use `scripts/export_dataset.py training-source --input sources.jsonl --provenance provenance.jsonl --output release/` to package source images and instructions. Relative input paths are resolved against `--image-root`. Provenance rows are keyed by `source_id` and include the image category, source, attribution and reported upstream license. The exporter preserves request IDs, source identities, instruction order and seeds.

For an existing frozen benchmark use `scripts/export_dataset.py add-preserve --input benchmark-directory --provenance provenance.jsonl --output release/`. Benchmark provenance rows are keyed by task `id`. The exporter preserves image bytes, masks and the tasks file, and removes machine-specific selection metadata. The `original/` images are retained for reference when present.

Run `scripts/export_dataset.py validate release/` after packaging or downloading. The generated `checksums.sha256` covers the release files, and `release.json` records counts and the instruction contract. The bundled dataset-card templates describe the paper releases; update their counts, names and protocol descriptions when publishing a different dataset.
