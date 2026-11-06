# Inference suites

## Policies

- `base`: original pretrained model for every denoising step.
- `tailsteer`: trained model for every denoising step; the default method in the paper.
- `transition`: original model before the steering onset, trained model afterwards; optional and off by default.

Default comparisons use `base tailsteer`. Each branch uses its own preceding output image. FLUX and Qwen decode and re-encode at each turn; SenseNova directly uses image-space feedback. Input dimensions remain non-square, aligned as required by each backbone. Qwen editing benchmarks additionally cap image area at approximately 2048² pixels without changing aspect ratio.

```bash
tailsteer infer --run-dir runs/klein-main --step 2000 \
  --inference-suite cafe-20turn --output-dir outputs/cafe --gpus 8 --resume
```

## Released checkpoints

`--checkpoint` accepts a Hub model ID, a local complete TailSteer repository, or an inference export produced by training:

```bash
tailsteer infer --backbone qwen-image-2.1 \
  --checkpoint Lyricccco/Qwen-Image-2.1-TailSteer \
  --policies tailsteer --inference-suite cafe-20turn --gpus 1
```

For TailSteer-only inference, a complete released repository supplies all model components. To compare `base tailsteer` or include `transition`, prepare the original base weights as well. Keep the backbone's official environment and source dependencies installed. `--checkpoint-revision COMMIT` pins the downloaded model snapshot.

The released models also work directly with their official pipelines by replacing the model ID or path. The suite interface adds task sequencing, image feedback, comparisons and evaluation; it does not introduce a different model architecture.

## Included demos

| Suite | Input | Turns |
|---|---|---:|
| `cafe-20turn` | One source image | 20 |
| `lace-dress-dance` | Person plus pose references | 10 |
| `yard-object-composition` | Background plus objects | 5 |
| `people-composition` | Background plus people | 4 |

`--inference-suite demos` runs all supported demos. Inputs, prompts and reference results live in `demo-testset/`. Multi-reference demos are supported by FLUX.2, Qwen and SenseNova; Kontext skips them because its native interface accepts one reference image.

Outputs include `tasks/<id>/base/round-XX.png`, `tasks/<id>/tailsteer/round-XX.png`, per-task metadata and comparison panels. A resume checks task/configuration identity before reusing images, to avoid silently mixing different prompts, seeds or checkpoints.

## Custom suites

A suite points to a task file and an image root. Copy a demo YAML and adjust those two paths:

```yaml
schema_version: 1
name: my-edits
num_steps: 50
seed: 42
tasks: /absolute/path/to/tasks.json
image_root: /absolute/path/to/images
policies: [base, tailsteer]
backends:
  flux2:
    inference:
      feedback: {mode: image}
      decode_intermediate: true
      decode_after_chain: none
```

```json
{"tasks":[{"id":"example","image":"source.png","chain_rounds":true,"rounds":[{"prompt":"Make the mug blue.","seed":42},{"prompt":"Add a small spoon next to it.","seed":43}]}]}
```

```bash
tailsteer infer --run-dir runs/klein-main \
  --inference-suite configs/inference/suites/my_edits.yaml --gpus 8
```

The planner uses native denoising-step counts for each backbone (30/50/40/50). The bundled multi-reference task files demonstrate `condition_images`, `primary_source` and `reference_roles`; use them as templates so reference ordering and feedback semantics remain explicit. Multi-reference suites set `inference.multi_condition.max_static_references` inside each supported backend section; the bundled suites set this to 1, 4, or 3 as needed. A single-image parser rejects extra reference images instead of dropping them.

`full-multi-align` and `vary-test` are optional suite templates for externally supplied task files under `data/`. Their complete datasets are not bundled; only the specified café example is included.

## Benchmarks

```bash
tailsteer infer --run-dir runs/klein-main \
  --inference-suite t2i --inference-suite single-turn \
  --inference-suite multi-turn --inference-suite add-preserve --gpus 8
```

| Suite | Benchmarks |
|---|---|
| `t2i` | Qwen-Image-Bench → BizGenEval |
| `single-turn` | GEdit → RISE → ImgEdit-basic |
| `multi-turn` | MSE-Bench → EdiVal |
| `add-preserve` | Add & Preserve, objective metrics and COA |
| `paper-benchmarks` | All seven external benchmarks, then Add & Preserve |

Each external benchmark can also be selected directly: `qwenimagebench`, `bizgeneval`, `gedit`, `rise`, `imgedit-basic`, `mse`, `edival`. Use `--stage generate` or `--stage score` to separate the two stages. `--limit N` selects an isolated smoke subset of an external benchmark; it does not shorten demo turns. Native suites generate images and their configured metrics together.

External benchmark results are saved under `--eval-root/results/`; demo and Add & Preserve results are under `--output-dir`. Prepare benchmark assets and judges before launching; see [evaluation](evaluation.md).

Evaluation groups can also be stored as YAML (`kind: evaluation`), for example
`--inference-suite configs/evaluation/suites/editing-five.yaml`. Native task suites and evaluation profiles use the same public flag, including after training. CLI options override profile defaults.
