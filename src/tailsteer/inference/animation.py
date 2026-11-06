"""Render saved edit chains without interpolating or modifying generated frames."""

import html
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from tailsteer.evaluation.common import atomic_text, contained, read_json, write_json

DEFAULT_FPS = 2


def _frame(paths, names, round_index, size):
    width, height = size
    canvas = Image.new("RGB", (width * len(paths), height + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (path, name) in enumerate(zip(paths, names, strict=True)):
        with Image.open(path) as image:
            tile = ImageOps.contain(image.convert("RGB"), size)
        x = column * width + (width - tile.width) // 2
        canvas.paste(tile, (x, 32 + (height - tile.height) // 2))
        draw.text(
            (column * width + 8, 8), f"{name} | frame {round_index:02d}", fill="black"
        )
    return canvas


def _write_mp4(frames, path, fps):
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path),
        frames[0].size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=2,
        output_params=["-crf", "18"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(frame.tobytes())
    finally:
        writer.close()


def render_task(
    task_path: Path,
    generation: Path,
    output: Path,
    *,
    fps=DEFAULT_FPS,
    tile_size=512,
    mp4=True,
    include_source=True,
):
    task = read_json(task_path)
    records = task["results"]
    names = list(dict.fromkeys(r["policy"] for r in records))
    if not names or not task.get("chain_rounds"):
        raise ValueError(f"Expected a chained policy comparison: {task_path}")
    rounds = sorted({int(r["round"]) for r in records})
    if rounds != list(range(1, len(rounds) + 1)):
        raise ValueError("Round coverage must be contiguous from one")
    by_key = {(r["policy"], int(r["round"])): r for r in records}
    if len(by_key) != len(records) or len(records) != len(names) * len(rounds):
        raise ValueError("Duplicate or incomplete policy/round coverage")
    expected = task.get("input", {}).get("task", {}).get("rounds")
    if expected is not None and len(expected) != len(rounds):
        raise ValueError("Not all requested rounds were generated")
    source = contained(generation, task["source"])
    with Image.open(source) as img:
        size = ImageOps.contain(img, (tile_size, tile_size)).size
    size = tuple(v + v % 2 for v in size)
    paths = [[source] * len(names)]
    prompts = ["Source image"]
    has_references = any(r.get("condition_images") for r in records)
    # Preserve the existing single-guide column. Variable multi-reference sets
    # belong in the per-round HTML gallery, not one misleading guide thumbnail.
    guided = has_references and all(
        len(r.get("condition_images", [])) == 1 for r in records
    )
    reference_rows = [[]]
    guide_label = "guide (frame 0: source)" if include_source else "guide"
    display_names = [guide_label, *names] if guided else names
    if guided:
        paths[0].insert(0, source)
    if has_references:
        reference_root = Path(task["input"]["inference"]["image_root"])
    for index in rounds:
        row = [by_key[(name, index)] for name in names]
        if len({r["prompt"] for r in row}) != 1:
            raise ValueError("Policies must use identical prompts")
        frame_paths = [contained(generation, r["image"]) for r in row]
        reference_paths = []
        if has_references:
            references = {tuple(r.get("condition_images", [])) for r in row}
            if len(references) != 1:
                raise ValueError(
                    "Guide animation requires one identical guide per policy/round"
                    if guided
                    else "Animation requires identical references per policy/round"
                )
            reference_paths = [
                contained(reference_root, path) for path in next(iter(references))
            ]
            for path in reference_paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
        if guided:
            frame_paths.insert(0, reference_paths[0])
        reference_rows.append(reference_paths)
        paths.append(frame_paths)
        prompts.append(row[0]["prompt"])
    output.mkdir(parents=True, exist_ok=True)
    frames = [_frame(row, display_names, i, size) for i, row in enumerate(paths)]
    playback = frames if include_source else frames[1:]
    # One global palette prevents per-frame GIF palette changes causing flicker.
    palette_board = Image.new("RGB", (128 * len(playback), 128), "white")
    for i, frame in enumerate(playback):
        palette_board.paste(ImageOps.contain(frame, (128, 128)), (128 * i, 0))
    palette = palette_board.quantize(colors=256)
    gif_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in playback
    ]
    gif_frames[0].save(
        output / "comparison.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
        disposal=2,
    )
    if mp4:
        _write_mp4(playback, output / "comparison.mp4", fps)
    indices = sorted({0, *range(5, len(frames), 5), len(frames) - 1})
    # Rows are time checkpoints; columns are policies (readable, not a 20-wide strip).
    thumbnails = [ImageOps.contain(frames[i], (768, 320)) for i in indices]
    sheet = Image.new(
        "RGB",
        (max(i.width for i in thumbnails), sum(i.height for i in thumbnails)),
        "white",
    )
    y = 0
    for tile in thumbnails:
        sheet.paste(tile, (0, y))
        y += tile.height
    sheet.save(output / "keyframes.jpg", quality=95)
    # Link original PNGs; animations never become inputs for another round.
    source_manifest = os.path.relpath(task_path, output)
    items = [
        f"<h1>{html.escape(task['id'])}</h1>",
        "<p>Actual sequential outputs; "
        "no interpolation, stabilization or frame selection. "
        "Replay wraps to the first displayed frame; that jump is not generated.</p>",
        '<video controls loop src="comparison.mp4"></video>'
        if mp4
        else '<img src="comparison.gif">',
        '<p><a href="comparison.gif">GIF</a> · <a href="keyframes.jpg">Keyframes</a> · '
        f'<a href="{html.escape(source_manifest, quote=True)}">'
        "Original manifest</a></p>",
    ]
    for i, (row, prompt) in enumerate(zip(paths, prompts, strict=True)):
        items.append(f"<h3>Frame {i:02d}</h3><p>{html.escape(prompt)}</p>")
        items.extend(
            f'<a href="{html.escape(os.path.relpath(p, output), quote=True)}">'
            '<img width="320" '
            f'src="{html.escape(os.path.relpath(p, output), quote=True)}" '
            f'alt="{html.escape(name)}"></a>'
            for p, name in zip(row, display_names, strict=True)
        )
        if not guided and reference_rows[i]:
            items.append("<h4>Input references (image 1 is the current scene)</h4>")
            items.extend(
                f'<figure style="display:inline-block;margin:8px">'
                '<img width="160" '
                f'src="{html.escape(os.path.relpath(p, output), quote=True)}">'
                f"<figcaption>Image {j}</figcaption></figure>"
                for j, p in enumerate(reference_rows[i], 2)
            )
    atomic_text(output / "index.html", _page("\n".join(items)))
    info = {
        "task": task["id"],
        "policies": names,
        "display_columns": display_names,
        "generated_rounds": len(rounds),
        "frames_including_source": len(frames),
        "animation_frames": len(playback),
        "includes_source_frame": include_source,
        "fps": fps,
        "task_manifest": str(task_path.resolve()),
    }
    write_json(output / "animation.json", info)
    return info


def _page(body):
    return (
        '<!doctype html><meta charset="utf-8"><title>Motion chains</title><style>'
        "body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px}"
        "video{max-width:100%} img{max-width:100%;vertical-align:top}</style>" + body
    )


def render_sequences(
    generation: Path,
    output: Path,
    *,
    fps=DEFAULT_FPS,
    tile_size=512,
    mp4=True,
    include_source=True,
):
    if fps <= 0 or tile_size < 2:
        raise ValueError("fps and tile size must be positive")
    results = []
    for path in sorted((generation / "tasks").glob("*/task.json")):
        if not (path.parent / "COMPLETE").is_file():
            continue
        results.append(
            render_task(
                path,
                generation,
                output / path.parent.name,
                fps=fps,
                tile_size=tile_size,
                mp4=mp4,
                include_source=include_source,
            )
        )
    body = [
        "<h1>Source-derived sequential editing animations</h1>",
        f"<p>{len(results)} completed source-image chains. "
        "Comparison of actual generated frames, not an official benchmark score.</p>",
    ]
    for result in results:
        name = result["task"]
        body.append(
            f'<h2><a href="{html.escape(name)}/index.html">{html.escape(name)}</a></h2>'
            f'<a href="{html.escape(name)}/index.html"><img width="480" '
            f'src="{html.escape(name)}/comparison.gif" loading="lazy"></a>'
        )
    atomic_text(output / "index.html", _page("\n".join(body)))
    write_json(
        output / "manifest.json",
        {"generation": str(generation.resolve()), "sequences": results},
    )
    return output / "index.html"
