/* Qualitative flow and feedback schematics; not measured latent trajectories. */
(() => {
  "use strict";
  const section = document.querySelector("#scene-drift");
  const svg = section.querySelector("svg");
  const viewport = section.querySelector(".drift-viewport");
  const $ = (id) => section.querySelector(`#${id}`);
  const ns = "http://www.w3.org/2000/svg";
  const gray = "#bac3ce", blue = "#315b9c", purple = "#692568", orange = "#cc7033";
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };
  const mix = (a, b, t) => a + (b - a) * t;
  const set = (el, attrs) => Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  const make = (tag, attrs = {}, text, parent = svg) => {
    const el = document.createElementNS(ns, tag);
    set(el, attrs);
    if (text !== undefined) el.textContent = text;
    parent.append(el);
    return el;
  };
  const group = () => make("g");
  const path = (parent, stroke, width = 2, attrs = {}) => make("path", {fill: "none", stroke, "stroke-width": width, ...attrs}, undefined, parent);
  const label = (parent, text, x, y, cls = "", attrs = {}) => make("text", {x, y, class: cls, ...attrs}, text, parent);
  const dot = (parent, fill, r, attrs = {}) => make("circle", {r, fill, ...attrs}, undefined, parent);
  const opacity = (el, value) => el.setAttribute("opacity", clamp(value));
  const draw = (fn, progress = 1) => {
    const count = Math.max(1, Math.ceil(progress * 100));
    return Array.from({length: count + 1}, (_, i) => {
      const p = fn(progress * i / count);
      return `${i ? "L" : "M"}${p.x.toFixed(2)},${p.y.toFixed(2)}`;
    }).join(" ");
  };
  const stages = [
    {start: 0, end: 11, title: "One imperfect update changes the next state.",
      body: "Prediction errors can propagate as every step uses the state reached by the previous one.",
      equation: "x<sub>k+1</sub> = x<sub>k</sub> + Δt · v<sub>θ</sub>(x<sub>k</sub>, t<sub>k</sub>, c)"},
    {start: 11, end: 24, title: "Trained on interpolations. Tested on its own rollouts.",
      body: "Imperfect rollouts can shift the input distribution. Near the tail, fewer updates remain to correct errors.",
      equation: "x<sub>t</sub><sup>train</sup> = (1 − t) ε + t x<sub>data</sub>"},
    {start: 24, end: 38, title: "The output becomes the next edit’s starting image.",
      body: "Each turn samples fresh noise, but conditions on an image that can already carry preservation error.",
      equation: "I<sub>r+1</sub> = Edit<sub>θ</sub>(I<sub>r</sub>, c<sub>r+1</sub>; ε<sub>r+1</sub>)"},
  ];
  const total = stages.at(-1).end;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  let time = 0, phase = -1, manualPanUntil = 0;
  const buttons = [...section.querySelectorAll("[data-drift-phase]")];
  const slider = $("drift-seek");

  const defs = make("defs");
  for (const [name, color] of [["gray", gray], ["purple", purple], ["orange", orange], ["blue", blue]]) {
    const marker = make("marker", {id: `drift-arrow-${name}`, viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse"}, undefined, defs);
    make("path", {d: "M1 1L7 4L1 7", fill: "none", stroke: color, "stroke-width": 1.5}, undefined, marker);
  }
  const gradient = make("linearGradient", {id: "drift-tail-wash", x1: 0, x2: 1}, undefined, defs);
  make("stop", {offset: 0, "stop-color": orange, "stop-opacity": 0}, undefined, gradient);
  make("stop", {offset: 1, "stop-color": orange, "stop-opacity": .08}, undefined, gradient);

  const training = group();
  label(training, "Training · interpolated noisy states", 65, 42, "drift-blue");
  label(training, "Noise ε", 65, 174, "drift-small");
  label(training, "Clean data", 1000, 158, "drift-small", {"text-anchor": "end"});
  const trainingDots = [];
  for (let i = -4; i <= 4; i++) {
    const y0 = 115 + i * 8, y1 = 115 + i * 3;
    path(training, "#b9cbea", 1.2, {d: `M65 ${y0}L1000 ${y1}`, opacity: .6});
    trainingDots.push({el: dot(training, blue, 3.5, {opacity: .4}), y0, y1});
  }
  const trainPoint = dot(training, "#fff", 6, {stroke: blue, "stroke-width": 2});
  const timeGuide = path(training, "#b0bbca", 1.1, {"stroke-dasharray": "3 7"});
  label(training, "Inference · model-generated states", 65, 228, "drift-purple");

  const main = group();
  const tailWash = make("rect", {x: 840, y: 160, width: 180, height: 216, rx: 20, fill: "url(#drift-tail-wash)"}, undefined, main);
  const reference = path(main, gray, 2.6, {"stroke-dasharray": "5 7"});
  const context = path(main, purple, 3.2, {opacity: .12});
  const trace = path(main, purple, 3.5);
  const steps = Array.from({length: 11}, () => dot(main, purple, 2.8));
  const start = dot(main, "#fff", 6, {stroke: blue, "stroke-width": 2});
  const intended = dot(main, "#fff", 6, {stroke: gray, "stroke-width": 2});
  const endHalo = dot(main, purple, 20, {opacity: .07});
  const end = dot(main, "#fff", 6, {stroke: purple, "stroke-width": 2.4});
  const gap = path(main, orange, 1.6, {"stroke-dasharray": "2 5"});
  const originalLabel = label(main, "Reference endpoint", 1000, 165, "", {"text-anchor": "end"});
  const actualLabel = label(main, "Actual endpoint", 1000, 312, "drift-purple", {"text-anchor": "end"});
  const startLabel = label(main, "Noise", 65, 263, "drift-small", {"text-anchor": "middle"});
  const condition = label(main, "Same source + instruction", 65, 70, "drift-small");
  const tailLabel = label(main, "Trajectory tail", 915, 405, "drift-small", {"text-anchor": "middle"});
  const cursorHalo = dot(main, purple, 14, {opacity: .08});
  const cursor = dot(main, purple, 6);

  const velocity = group();
  const referenceArrow = path(velocity, "#94a1b1", 2, {"marker-end": "url(#drift-arrow-gray)"});
  const predictedArrow = path(velocity, purple, 2.5, {"marker-end": "url(#drift-arrow-purple)"});
  const errorArrow = path(velocity, orange, 1.8, {"marker-end": "url(#drift-arrow-orange)"});
  const velocityLabel = label(velocity, "Prediction error", 400, 115, "drift-hand drift-orange");
  const velocityLeader = path(velocity, orange, 1.4, {"marker-end": "url(#drift-arrow-orange)"});
  const exposure = group();
  const exposureLabel = label(exposure, "Training ≠ inference", 470, 195, "drift-hand drift-orange");
  const exposureLeader = path(exposure, orange, 1.5, {"marker-end": "url(#drift-arrow-orange)"});

  const feedback = group();
  const turns = [0, 1, 2].map((i) => {
    const g = make("g", {}, undefined, feedback), x = 65 + 365 * i;
    label(g, `Turn ${i + 1}`, x, 72, "drift-hand drift-ink");
    make("rect", {x, y: 119, width: 208, height: 42, rx: 12, fill: "#ecf1f8", stroke: "#dde5f0"}, undefined, g);
    label(g, i ? `Previous output + edit ${i + 1}` : "Source image + edit 1", x + 104, 145, "drift-small", {"text-anchor": "middle"});
    path(g, "#a9b8cc", 1.5, {d: `M${x + 104} 163Q${x + 104} 196 ${x + 135} 230`, "marker-end": "url(#drift-arrow-gray)"});
    const ref = path(g, gray, 2, {"stroke-dasharray": "4 6"});
    const output = path(g, purple, 3.2);
    const noise = dot(g, "#fff", 5, {stroke: blue, "stroke-width": 1.8});
    const referenceEnd = dot(g, "#fff", 5, {stroke: gray, "stroke-width": 1.8});
    const residue = path(g, orange, 1.5, {"stroke-dasharray": "2 4"});
    const outputDot = dot(g, purple, 5);
    label(g, "Fresh noise", x, 307, "drift-small");
    const outLabel = label(g, `Output ${i + 1}`, x + 260, 330, "drift-small drift-purple", {"text-anchor": "end"});
    const connector = path(g, purple, 1.6, {"marker-end": "url(#drift-arrow-purple)"});
    const carry = dot(g, orange, 4);
    const carryLabel = label(g, "next source", x + 300, 200, "drift-small drift-purple", {"text-anchor": "middle"});
    return {g, ref, output, noise, referenceEnd, residue, outputDot, outLabel, connector, carry, carryLabel};
  });
  label(feedback, "Residual changes can travel from one edit to the next.", 560, 408, "drift-hand drift-purple drift-chain-summary", {"text-anchor": "middle"});

  const legend = group();
  path(legend, gray, 2.4, {d: "M65 450h28", "stroke-dasharray": "4 5"});
  label(legend, "Reference flow", 103, 454, "drift-small");
  path(legend, purple, 3, {d: "M250 450h28"});
  label(legend, "Model rollout", 288, 454, "drift-small drift-purple");
  path(legend, orange, 1.8, {d: "M435 450h28", "stroke-dasharray": "2 4"});
  label(legend, "Prediction error", 473, 454, "drift-small drift-orange");
  label(legend, "Illustrative", 1070, 454, "drift-small", {"text-anchor": "end"});

  function base(t, shift = 0) {
    return {x: 65 + 935 * t, y: 222 + 30 * Math.sin(Math.PI * t) - 32 * t + shift};
  }
  function residual(t) { return 9 * t + 64 * t * t + 6 * Math.sin(4 * Math.PI * t) * t * (1 - t); }
  function rollout(t, shift = 0) { const p = base(t, shift); return {x: p.x, y: p.y + residual(t)}; }
  function turn(t, i, error = true) {
    return {x: 65 + 365 * i + 260 * t, y: 265 + 10 * Math.sin(Math.PI * t) - 16 * t + (error ? (25 + 25 * i) * t * t : 0)};
  }
  function move(el, p) { set(el, {cx: p.x, cy: p.y}); }
  function progressFor(i, elapsed) { return i === 0 ? 1 : clamp((elapsed - (i === 1 ? 2.2 : 6.8)) / 3.6); }

  function render(value = time) {
    time = clamp(value, 0, total);
    const index = stages.findIndex((s) => time < s.end);
    const current = index < 0 ? 2 : index;
    const stage = stages[current], elapsed = time - stage.start;
    if (phase !== current) {
      phase = current;
      buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(i === phase)));
      $("drift-caption-title").textContent = stage.title;
      $("drift-caption-body").textContent = stage.body;
      $("drift-equation").innerHTML = stage.equation;
    }
    const transition = current === 2 ? ease(elapsed / 1.3) : 0;
    const shift = current === 0 ? 0 : current === 1 ? 68 * ease(elapsed / 1.1) : 68;
    const progress = current === 0 ? clamp((elapsed - .8) / 7.8) : current === 1 ? clamp((elapsed - 1.3) / 9.2) : 1;
    const referenceAt = (t) => {
      const a = base(t, shift), b = turn(t, 0, false);
      return {x: mix(a.x, b.x, transition), y: mix(a.y, b.y, transition)};
    };
    const modelAt = (t) => {
      const a = rollout(t, shift), b = turn(t, 0);
      return {x: mix(a.x, b.x, transition), y: mix(a.y, b.y, transition)};
    };
    set(reference, {d: draw(referenceAt)});
    set(context, {d: draw(modelAt)});
    set(trace, {d: draw(modelAt, progress)});
    const q = modelAt(progress), p0 = modelAt(0), p1 = modelAt(1), refEnd = referenceAt(1);
    move(start, p0); move(intended, refEnd); move(end, p1); move(endHalo, p1);
    move(cursor, q); move(cursorHalo, q);
    steps.forEach((d, i) => { move(d, modelAt(i / 10)); opacity(d, (i / 10 <= progress ? .7 : .1) * (1 - transition)); });
    opacity(end, ease((progress - .92) / .08));
    opacity(endHalo, .07 * ease((progress - .92) / .08) * (1 - transition));
    opacity(cursor, (current === 2 ? 1 - transition : 1));
    opacity(cursorHalo, .08 * (1 - transition));
    set(gap, {d: `M${refEnd.x} ${refEnd.y + 8}V${p1.y - 8}`});
    opacity(gap, ease((progress - .87) / .13));
    set(originalLabel, {x: refEnd.x, y: refEnd.y - 24});
    set(actualLabel, {x: p1.x, y: p1.y + 34});
    set(startLabel, {x: p0.x, y: p0.y + 33});
    for (const el of [originalLabel, actualLabel, startLabel, condition, tailLabel, tailWash]) opacity(el, 1 - transition);
    opacity(actualLabel, ease((progress - .84) / .16) * (1 - transition));
    opacity(condition, current === 0 ? 1 : 0);
    opacity(tailWash, current === 1 ? ease(elapsed / 2) : current === 0 ? .5 : 0);
    const trainingAlpha = current === 1 ? ease(elapsed / 1.1) : current === 2 ? 1 - transition : 0;
    opacity(training, trainingAlpha);
    trainingDots.forEach(({el, y0, y1}) => move(el, {x: q.x, y: mix(y0, y1, progress)}));
    move(trainPoint, {x: q.x, y: 115});
    set(timeGuide, {d: `M${q.x} 79V${q.y - 12}`});
    const derivative = (fn) => { const a = fn(Math.max(0, progress - .002)), b = fn(Math.min(1, progress + .002)); return (b.y - a.y) / (b.x - a.x); };
    const predicted = {x: q.x + 66, y: q.y + 66 * derivative(modelAt)};
    const expected = {x: q.x + 66, y: q.y + 66 * derivative(referenceAt) - residual(progress) * .2};
    set(predictedArrow, {d: `M${q.x} ${q.y}L${predicted.x} ${predicted.y}`});
    set(referenceArrow, {d: `M${q.x} ${q.y}L${expected.x} ${expected.y}`});
    set(errorArrow, {d: `M${expected.x + 5} ${expected.y}L${predicted.x + 5} ${predicted.y}`});
    const calloutX = clamp(q.x - 90, 170, 680), calloutY = q.y - 135;
    set(velocityLabel, {x: calloutX, y: calloutY});
    set(velocityLeader, {d: `M${calloutX + 90} ${calloutY + 12}Q${predicted.x + 60} ${calloutY + 30} ${predicted.x + 18} ${(predicted.y + expected.y) / 2}`});
    opacity(velocity, current === 0 ? ease((progress - .14) / .1) * (1 - ease((progress - .87) / .1)) : 0);
    const exposureX = clamp(q.x - 130, 260, 720);
    set(exposureLabel, {x: exposureX, y: 195});
    set(exposureLeader, {d: `M${exposureX + 130} 205Q${q.x + 55} 235 ${q.x + 10} ${q.y - 18}`});
    opacity(exposure, current === 1 ? ease((progress - .25) / .12) : 0);

    opacity(feedback, transition);
    turns.forEach((item, i) => {
      const p = progressFor(i, elapsed), visible = i === 0 ? 1 : ease((elapsed - (i === 1 ? 1.3 : 5.9)) / .6);
      opacity(item.g, current === 2 ? visible : 0);
      const first = turn(0, i), target = turn(1, i), intendedEnd = turn(1, i, false);
      set(item.ref, {d: draw((t) => turn(t, i, false))});
      set(item.output, {d: draw((t) => turn(t, i), p)});
      move(item.noise, first); move(item.referenceEnd, intendedEnd); move(item.outputDot, turn(p, i));
      set(item.residue, {d: `M${target.x} ${intendedEnd.y + 7}V${target.y - 7}`});
      opacity(item.residue, ease((p - .9) / .1));
      set(item.outLabel, {y: target.y + 30}); opacity(item.outLabel, ease((p - .9) / .1));
      if (i === 0) { opacity(item.ref, 0); opacity(item.output, 0); opacity(item.noise, 0); opacity(item.referenceEnd, 0); opacity(item.residue, 0); }
      const connectorProgress = i === 0 ? ease((elapsed - 1.1) / 1.1) : ease((elapsed - 5.7) / 1.1);
      const nextX = 65 + 365 * (i + 1);
      set(item.connector, {d: `M${target.x + 10} ${target.y}C${target.x + 65} ${target.y} ${nextX - 55} 140 ${nextX - 5} 140`});
      const length = item.connector.getTotalLength();
      set(item.connector, {"stroke-dasharray": length, "stroke-dashoffset": length * (1 - connectorProgress)});
      opacity(item.connector, i < 2 ? connectorProgress : 0);
      move(item.carry, item.connector.getPointAtLength(length * connectorProgress));
      opacity(item.carry, i < 2 && connectorProgress > 0 && connectorProgress < 1 ? 1 : 0);
      opacity(item.carryLabel, i < 2 ? connectorProgress : 0);
    });
    let focus = q.x;
    if (current === 2) {
      const i = elapsed < 2.2 ? 0 : elapsed < 6.8 ? 1 : 2;
      focus = turn(progressFor(i, elapsed), i).x;
    }
    if (viewport.scrollWidth > viewport.clientWidth + 5 && performance.now() > manualPanUntil) {
      const target = clamp(focus / 1120 * svg.clientWidth - viewport.clientWidth * .55, 0, viewport.scrollWidth - viewport.clientWidth);
      viewport.scrollLeft += (target - viewport.scrollLeft) * (reduced.matches ? 1 : .08);
    }
    slider.value = time;
    $("drift-progress").textContent = current < 2 ? `Denoising step ${Math.round(progress * 50)} / 50` : `Edit turn ${elapsed < 2.2 ? 1 : elapsed < 6.8 ? 2 : 3} / 3`;
  }
  function seek(value, play) {
    manualPanUntil = 0;
    section.dispatchEvent(new CustomEvent("driftseek", {
      bubbles: true, detail: {time: value, play},
    }));
  }
  buttons.forEach((button, i) => button.addEventListener("click", () => {
    seek(reduced.matches ? stages[i].end - .1 : stages[i].start, !reduced.matches);
  }));
  slider.addEventListener("input", () => seek(Number(slider.value), false));
  viewport.addEventListener("pointerdown", () => { manualPanUntil = performance.now() + 6000; }, {passive: true});
  viewport.addEventListener("wheel", () => { manualPanUntil = performance.now() + 6000; }, {passive: true});
  new ResizeObserver(() => { if (!section.hidden) render(); }).observe(viewport);
  window.TailSteerDrift = {duration: total, render};
  render();
})();
