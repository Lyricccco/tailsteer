(() => {
  "use strict";
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const make = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const cards = new Set();
  const images = window.TailSteerImages;
  const idleOwner = {},
    evidenceOwner = {};
  let resultsNear = false,
    warmingModel = null;
  const initialIndex = (data) =>
    Math.max(0, data.turns.indexOf(data.defaultTurn ?? data.turns.at(-1)));
  function modelLabel(tag, text) {
    const el = make(tag, "", text);
    if (text.startsWith("After TailSteer"))
      el.replaceChildren(
        document.createTextNode("After "),
        make("span", "tailsteer-accent", "TailSteer"),
        document.createTextNode(text.slice("After TailSteer".length)),
      );
    return el;
  }
  const quiet = (error) => {
    if (error.name !== "AbortError") console.error(error);
  };
  let gallery,
    evidence,
    t2i,
    currentModel = "flux2",
    currentView = "visuals";
  let globalPlaying = false;
  let dialogCard = null,
    galleryCards = [];
  const dialog = $("#result-comparison-dialog");
  const cardObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const state = entry.target.resultState;
        if (!state) continue;
        state.visible = entry.isIntersecting;
        if (state.visible) {
          state.nextAt = performance.now() + 3500;
          hydrateGallery();
        }
      }
    },
    { threshold: 0.15 },
  );

  function frameColumns(state) {
    return [...state.columns, ...(state.guide ? [state.guide] : [])];
  }
  function scheduleWarmup() {
    if (
      !resultsNear ||
      currentView !== "visuals" ||
      document.hidden ||
      warmingModel === currentModel
    )
      return;
    const shown = galleryCards.filter((s) => s.element.offsetWidth > 0);
    if (!shown.length || shown.some((s) => !s.ready || s.loading)) return;
    warmingModel = currentModel;
    // Round-robin across groups: nearby turns first, one idle download at a time.
    const length = Math.max(...shown.map((s) => s.data.turns.length));
    for (let offset = 1; offset < length; offset++)
      for (const state of shown) {
        if (offset >= state.data.turns.length) continue;
        const index = (state.index + offset) % state.data.turns.length;
        for (const col of frameColumns(state))
          images.warm(col.frames[index], idleOwner);
      }
  }
  function resetWarmup() {
    images.cancel(idleOwner);
    warmingModel = null;
  }
  function hydrateGallery() {
    if (currentView !== "visuals") return;
    for (const state of galleryCards) {
      if (!state.hydrated && state.element.offsetWidth > 0) {
        state.hydrated = true;
        setTurn(state, state.index, 1);
        for (const ref of state.references)
          images
            .assign(ref.image, ref.url, 1, state.referenceOwner)
            .catch(quiet);
      }
    }
    scheduleWarmup();
  }
  function updatePlayback(state) {
    state.element.dataset.playing = String(state.playing);
  }
  function updateGlobalPlayback() {
    const button = $("#results-play-toggle");
    button.replaceChildren(
      make("span", "", globalPlaying ? "Ⅱ" : "▶"),
      document.createTextNode(
        globalPlaying ? " Pause all comparisons" : " Play all comparisons",
      ),
    );
    button.firstChild.setAttribute("aria-hidden", "true");
    button.setAttribute("aria-pressed", String(globalPlaying));
    button.setAttribute(
      "aria-label",
      `${globalPlaying ? "Pause" : "Play"} all comparisons`,
    );
    for (const state of galleryCards) {
      state.playing = globalPlaying;
      if (!globalPlaying) pauseCard(state);
      state.nextAt = performance.now() + 1100;
      if (globalPlaying && state.visible)
        setTurn(
          state,
          state.ready
            ? (state.index + 1) % state.data.turns.length
            : state.index,
        );
      updatePlayback(state);
    }
  }
  $("#results-play-toggle").addEventListener("click", () => {
    globalPlaying = !globalPlaying;
    updateGlobalPlayback();
  });
  updateGlobalPlayback();
  function updateTurnLabel(state) {
    const turn = state.data.turns[state.index];
    state.output.replaceChildren(
      make("b", "", `${state.expanded ? "Turn " : ""}${turn}`),
      document.createTextNode(` / ${state.data.turns.at(-1)}`),
    );
    state.prompt.textContent =
      (state.expanded ? state.data.prompts : state.data.shortPrompts)?.[
        state.index
      ] || "";
    state.prompt.title = state.data.prompts[state.index] || "";
    state.range.setAttribute(
      "aria-valuetext",
      `Turn ${turn} of ${state.data.turns.at(-1)}`,
    );
    state.range.style.setProperty(
      "--turn-progress",
      `${(state.index / Math.max(1, state.data.turns.length - 1)) * 100}%`,
    );
    state.element.dataset.turn = turn;
  }
  async function setTurn(state, index, priority = 0) {
    index = Math.max(0, Math.min(state.data.turns.length - 1, index));
    if (state.loading && state.pendingIndex === index) {
      for (const col of frameColumns(state))
        images.load(col.frames[index], priority, state).catch(quiet);
      return;
    }
    const request = ++state.request;
    images.cancel(state);
    images.cancel(state.lookAheadOwner);
    state.pendingIndex = index;
    state.loading = true;
    state.element.setAttribute("aria-busy", "true");
    state.range.value = index;
    state.output.textContent = `Loading ${state.data.turns[index]}…`;
    try {
      const loaded = await Promise.all(
        frameColumns(state).map((col) =>
          images.load(col.frames[index], priority, state),
        ),
      );
      if (!cards.has(state) || request !== state.request) return;
      state.index = index;
      state.ready = true;
      state.images.forEach((image, i) => {
        if (!image) return;
        image.dataset.src = state.columns[i].frames[index];
        image.src = loaded[i].src;
        image.alt = `${state.columns[i].label}, ${state.data.title}, turn ${state.data.turns[index]}`;
      });
      for (const image of state.guideImages) {
        image.dataset.src = state.guide.frames[index];
        image.src = loaded.at(-1).src;
      }
      updateTurnLabel(state);
    } catch (error) {
      if (request !== state.request || error.name === "AbortError") return;
      state.playing = false;
      updatePlayback(state);
      state.range.value = state.index;
      state.output.textContent = "Could not load";
    } finally {
      if (request === state.request) {
        state.loading = false;
        state.element.removeAttribute("aria-busy");
        state.nextAt =
          performance.now() +
          (state.index === state.data.turns.length - 1 ? 3500 : 1100);
        if (state.playing && state.ready) {
          const next = (state.index + 1) % state.data.turns.length;
          for (const col of frameColumns(state))
            images.load(col.frames[next], 2, state.lookAheadOwner).catch(quiet);
        }
        scheduleWarmup();
      }
    }
  }
  function pauseCard(state) {
    state.playing = false;
    images.cancel(state.lookAheadOwner);
    if (state.loading && state.ready) {
      state.request++;
      images.cancel(state);
      state.loading = false;
      state.element.removeAttribute("aria-busy");
      state.range.value = state.index;
      updateTurnLabel(state);
    }
    updatePlayback(state);
  }
  function createCard(data, expanded = false, firstIndex = initialIndex(data)) {
    const element = make(
      "article",
      `result-card${data.featured ? " featured" : ""}`,
    );
    element.dataset.kind = data.kind;
    element.setAttribute("aria-label", data.title);
    const state = {
      data,
      element,
      index: firstIndex,
      expanded,
      playing: !expanded && globalPlaying,
      visible: expanded,
      loading: false,
      request: 0,
      nextAt: performance.now() + 3500,
      images: [],
      guideImages: [],
      ready: false,
      references: [],
      referenceOwner: {},
      lookAheadOwner: {},
      columns: data.columns.filter((col) =>
        data.kind === "single"
          ? col.role !== "input"
          : ["skeleton", "people", "objects"].includes(data.kind)
            ? col.role !== "reference"
            : true,
      ),
      guide:
        data.kind === "skeleton"
          ? data.columns.find((col) => col.role === "reference")
          : null,
    };
    if (!state.columns.length) state.columns = data.columns;
    cards.add(state);
    element.resultState = state;
    const frames = make("div", "comparison-frames");
    frames.style.setProperty("--columns", state.columns.length);
    if (data.kind === "preservation")
      frames.style.setProperty(
        "--comparison-aspects",
        state.columns.map((c) => `${c.aspect}fr`).join(" "),
      );
    state.frames = frames;
    state.columns.forEach((col) => {
      const figure = make("figure", "comparison-frame");
      figure.dataset.role = col.role;
      if (col.role === "reference" && data.referenceDescription)
        figure.title = data.referenceDescription;
      const shortLabels = {
        input: "Input",
        api: "GPT-2.5-Sunburst (API)",
        base: "Base model",
        tailsteer: "After TailSteer",
        reference: data.kind === "skeleton" ? "Guide" : "All inputs",
      };
      figure.append(
        modelLabel(
          "figcaption",
          col.role === "api"
            ? "GPT-2.5-Sunburst (API)"
            : expanded || data.kind === "preservation"
              ? col.label
              : shortLabels[col.role] || col.label,
        ),
      );
      const button = make(expanded ? "div" : "button", "comparison-image");
      if (!expanded) {
        button.setAttribute(
          "aria-label",
          `Enlarge ${col.label}, ${data.title}`,
        );
        button.addEventListener("click", () => {
          pauseCard(state);
          openComparison(data, state.index);
        });
      }
      if (col.frames[firstIndex]) {
        const image = make("img", "result-photo");
        image.dataset.src = col.frames[firstIndex];
        image.loading = "eager";
        image.decoding = "async";
        image.alt = `${col.label}, ${data.title}, turn ${data.turns[firstIndex]}`;
        image.addEventListener("load", () => fitFrames(state));
        button.append(image);
        state.images.push(image);
      } else {
        button.append(make("span", "missing-frame", "Result unavailable"));
        state.images.push(null);
      }
      figure.append(button);
      if (state.guide) {
        const guide = make("button", "result-guide");
        guide.setAttribute(
          "aria-label",
          `Enlarge pose guide for ${data.title}`,
        );
        guide.title = "Pose guide";
        const image = make("img");
        image.dataset.src = state.guide.frames[firstIndex];
        image.alt = "Pose guide";
        guide.append(image);
        state.guideImages.push(image);
        guide.addEventListener("click", () =>
          openReference(state.guide.frames[state.index], "Pose guide"),
        );
        figure.append(guide);
      }
      frames.append(figure);
    });
    element.append(frames);
    const controls = make("div", "result-turnbar");
    if (data.referenceImages?.length) {
      const references = make("div", "result-references");
      references.setAttribute("aria-label", "Reference people and objects");
      for (const ref of data.referenceImages) {
        const button = make("button");
        button.setAttribute("aria-label", `Enlarge reference: ${ref.label}`);
        const image = make("img");
        image.dataset.src = ref.src;
        state.references.push({ image, url: ref.src });
        image.alt = ref.label;
        image.loading = "lazy";
        button.append(image);
        button.addEventListener("click", () =>
          openReference(ref.src, ref.label),
        );
        references.append(button);
      }
      element.append(references);
    }
    state.range = make("input");
    state.range.type = "range";
    state.range.min = 0;
    state.range.max = data.turns.length - 1;
    state.range.step = 1;
    state.range.value = firstIndex;
    state.range.setAttribute("aria-label", `Editing turn for ${data.title}`);
    state.range.addEventListener("pointerdown", () => pauseCard(state));
    state.range.addEventListener("input", () => {
      const index = Number(state.range.value);
      pauseCard(state);
      setTurn(state, index);
    });
    state.output = make("output");
    controls.append(state.range, state.output);
    if (data.turns.length > 1) element.append(controls);
    state.prompt = make("p", "result-prompt");
    if (data.kind === "t2i")
      state.prompt.dataset.label = data.promptLabel || "Prompt";
    if (data.kind === "single" && !expanded)
      element.insertBefore(state.prompt, controls);
    if (expanded && data.kind === "t2i") element.append(state.prompt);
    updatePlayback(state);
    updateTurnLabel(state);
    if (!expanded) cardObserver.observe(element);
    return state;
  }
  function openReference(src, title) {
    openComparison(
      {
        id: src,
        title,
        turns: [0],
        columns: [{ label: title, role: "reference", frames: [src] }],
        prompts: [],
        kind: "reference",
      },
      0,
    );
  }
  function removeCard(state) {
    cards.delete(state);
    cardObserver.unobserve(state.element);
    state.request++;
    images.cancel(state);
    images.cancel(state.referenceOwner);
    images.cancel(state.lookAheadOwner);
  }
  function openComparison(data, index) {
    if (dialogCard) removeCard(dialogCard);
    dialogCard = createCard(data, true, index);
    $("#comparison-dialog-content").replaceChildren(dialogCard.element);
    if (!dialog.open) dialog.showModal();
    setTurn(dialogCard, index, 0);
    for (const ref of dialogCard.references)
      images
        .assign(ref.image, ref.url, 0, dialogCard.referenceOwner)
        .catch(quiet);
    document.body.classList.add("comparison-open");
    $(".comparison-dialog-close").focus();
  }
  $(".comparison-dialog-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      const r = dialog.getBoundingClientRect();
      if (
        event.clientX < r.left ||
        event.clientX > r.right ||
        event.clientY < r.top ||
        event.clientY > r.bottom
      )
        dialog.close();
    }
  });
  dialog.addEventListener("close", () => {
    if (dialogCard) removeCard(dialogCard);
    dialogCard = null;
    document.body.classList.remove("comparison-open");
  });
  setInterval(() => {
    if (document.hidden) return;
    const now = performance.now();
    for (const state of cards)
      if (
        state.playing &&
        state.visible &&
        (currentView === "visuals" || state.expanded) &&
        !state.loading &&
        (!dialog.open || state.expanded) &&
        now >= state.nextAt
      )
        setTurn(state, (state.index + 1) % state.data.turns.length);
  }, 120);
  reduced.addEventListener("change", () => {
    if (reduced.matches) {
      globalPlaying = false;
      updateGlobalPlayback();
    }
  });
  document.addEventListener("visibilitychange", () => {
    for (const state of cards) state.nextAt = performance.now() + 3500;
    if (document.hidden) resetWarmup();
    else scheduleWarmup();
  });

  function fitFrames(state) {
    for (const image of state.images) {
      if (!image?.naturalWidth || !image.clientHeight) continue;
      const ratio =
        image.clientWidth /
        image.clientHeight /
        (image.naturalWidth / image.naturalHeight);
      const crop = 1 - Math.min(ratio, 1 / ratio);
      image.style.objectFit =
        state.expanded || crop > 0.08 ? "contain" : "cover";
    }
  }

  // Each group keeps its images' aspect ratios. A slicing layout packs groups
  // together; only a small, shared edge crop is allowed to close the canvas.
  function splitAspects(state) {
    return [
      state.columns.find((c) => c.role === "tailsteer"),
      ...state.columns.filter((c) => c.role !== "tailsteer"),
    ].map((c) => c.aspect || 1);
  }
  function mosaicTrees(states, promptHeight) {
    const gap = 7,
      memo = new Map();
    function trees(mask) {
      if (memo.has(mask)) return memo.get(mask);
      const ids = states.map((_, i) => i).filter((i) => mask & (1 << i));
      const result = [];
      if (ids.length === 1) {
        const state = states[ids[0]];
        const ratios = state.columns.map((c) => c.aspect || 1);
        const split = ratios.length === 3 ? splitAspects(state) : [];
        const chrome =
          20 +
          (state.data.kind === "single" ? promptHeight : 0) +
          (state.data.referenceImages?.length ? 35 : 0);
        for (const mode of ratios.length === 3
          ? ["row", "column", "split"]
          : ["row", "column"]) {
          const aspect =
            mode === "row"
              ? ratios.reduce((a, b) => a + b, 0)
              : mode === "column"
                ? 1 / ratios.reduce((a, b) => a + 1 / b, 0)
                : split[0] + 1 / (1 / split[1] + 1 / split[2]);
          result.push({ state, mode, a: 1 / aspect, b: chrome, c: 0, cuts: 0 });
        }
      } else {
        const first = mask & -mask;
        for (let left = (mask - 1) & mask; left; left = (left - 1) & mask) {
          const right = mask ^ left;
          if (!(left & first) || !right) continue;
          for (const p of trees(left))
            for (const q of trees(right)) {
              result.push({
                p,
                q,
                axis: "y",
                a: p.a + q.a,
                b: p.b + q.b + gap,
                c: p.c + q.c,
                cuts: p.cuts | q.cuts | 1,
              });
              const a = 1 / (1 / p.a + 1 / q.a);
              result.push({
                p,
                q,
                axis: "x",
                a,
                b: a * (p.b / p.a + q.b / q.a),
                c: a * (p.c / p.a + q.c / q.a - gap),
                cuts: p.cuts | q.cuts | 2,
              });
            }
        }
      }
      // Six groups have millions of possible slicing trees. Keep a bounded,
      // diverse set of shapes, preferring balanced groups within each shape.
      const candidates =
        states.length > 4 && result.length > 128
          ? compactMosaics(result, ids.length)
          : result;
      memo.set(mask, candidates);
      return candidates;
    }
    return trees((1 << states.length) - 1);
  }
  function compactMosaics(trees, count) {
    const shapes = new Map();
    for (const tree of trees) {
      const height = tree.a * 600 + tree.b + tree.c;
      const rects = mosaicRects(tree, 600, height, 1);
      const score = rects.reduce(
        (sum, r) =>
          sum +
          Math.abs(Math.log((r.width * r.height * count) / (600 * height))),
        0,
      );
      const key = `${Math.round(Math.log(tree.a) * 24)}:${Math.round(tree.b / 16)}:${tree.cuts}`;
      if (!shapes.has(key) || score < shapes.get(key).score)
        shapes.set(key, { tree, score });
    }
    const diverse = [...shapes.values()].sort((p, q) => p.tree.a - q.tree.a);
    const stride = Math.max(1, diverse.length / 128);
    return Array.from(
      { length: Math.min(128, diverse.length) },
      (_, i) => diverse[Math.floor(i * stride)].tree,
    );
  }
  function mosaicRects(tree, width, height, scale, x = 0, y = 0, result = []) {
    if (tree.state) result.push({ ...tree, x, y, width, height });
    else if (tree.axis === "y") {
      const h = scale * (tree.p.a * width + tree.p.c) + tree.p.b;
      mosaicRects(tree.p, width, h, scale, x, y, result);
      mosaicRects(tree.q, width, height - h - 7, scale, x, y + h + 7, result);
    } else {
      const w = (height - tree.p.b - scale * tree.p.c) / (scale * tree.p.a);
      mosaicRects(tree.p, w, height, scale, x, y, result);
      mosaicRects(tree.q, width - w - 7, height, scale, x + w + 7, y, result);
    }
    return result;
  }
  function packMosaic(list) {
    const width = list.clientWidth,
      height = list.clientHeight;
    if (!width || !height) return;
    const promptHeight = parseFloat(
      getComputedStyle(list).getPropertyValue("--prompt-height"),
    );
    if (list.mosaicPromptHeight !== promptHeight) {
      list.mosaicPromptHeight = promptHeight;
      list.mosaicTrees = mosaicTrees(
        [...list.children].map((el) => el.resultState),
        promptHeight,
      );
    }
    let best,
      bestScore = Infinity;
    for (const tree of list.mosaicTrees) {
      const scale = (height - tree.b) / (tree.a * width + tree.c);
      if (scale <= 0) continue;
      const rects = mosaicRects(tree, width, height, scale);
      let score = 24 * Math.abs(Math.log(scale)) + (tree.cuts === 3 ? 0 : 1);
      for (const r of rects) {
        const count = r.state.columns.length;
        const iw =
          r.width / (r.mode === "row" ? count : r.mode === "split" ? 3 : 1);
        const ih =
          (r.height - r.b) /
          (r.mode === "row" ? 1 : r.mode === "split" ? 2 : count);
        const area = (r.width * r.height) / (width * height);
        if (iw < 48 || ih < 40 || r.width < 72 || area < 0.09) {
          score = Infinity;
          break;
        }
        score += 2 * Math.abs(Math.log(area * list.children.length));
        score += Math.max(0, 80 - iw) / 30 + Math.max(0, 80 - ih) / 30;
        if (r.state.data.featured) score += 30 * Math.max(0, 0.34 - area);
      }
      if (score < bestScore) {
        bestScore = score;
        best = rects;
      }
    }
    if (!best) return;
    for (const r of best) {
      const { element, frames, columns } = r.state;
      Object.assign(element.style, {
        left: `${r.x}px`,
        top: `${r.y}px`,
        width: `${r.width}px`,
        height: `${r.height}px`,
      });
      element.dataset.arrangement = r.mode;
      const ratios = columns.map((c) => c.aspect || 1);
      if (r.mode === "split") {
        const [main, upper, lower] = splitAspects(r.state);
        frames.style.gridTemplateColumns = `${main}fr ${1 / (1 / upper + 1 / lower)}fr`;
        frames.style.gridTemplateRows = `${1 / upper}fr ${1 / lower}fr`;
      } else {
        frames.style.gridTemplateColumns =
          r.mode === "row" ? ratios.map((r) => `${r}fr`).join(" ") : "1fr";
        frames.style.gridTemplateRows =
          r.mode === "column"
            ? ratios.map((r) => `${1 / r}fr`).join(" ")
            : "1fr";
      }
    }
    requestAnimationFrame(() => {
      best.forEach((r) => fitFrames(r.state));
      hydrateGallery();
    });
  }

  const mosaicObserver = new ResizeObserver((entries) => {
    for (const { target } of entries) packMosaic(target);
  });
  function renderGallery() {
    resetWarmup();
    mosaicObserver.disconnect();
    for (const state of galleryCards) removeCard(state);
    galleryCards = [];
    const model = gallery.models.find((m) => m.id === currentModel);
    const lanes = [];
    for (const [kind, title, index] of [
      ["single", "Single-input with multi-turn editing", "01"],
      ["multi", "Multi-input with multi-turn editing", "02"],
    ]) {
      const lane = make("div", "result-lane");
      lane.dataset.kind = kind;
      const header = make("div", "result-lane-heading");
      header.append(make("span", "", index), make("h3", "", title));
      lane.append(header);
      const list = make("div", "result-case-list");
      for (const data of model[kind]) {
        const state = createCard(data);
        galleryCards.push(state);
        list.append(state.element);
      }
      lane.append(list);
      lanes.push(lane);
    }
    $("#results-gallery").replaceChildren(...lanes);
    $("#results-gallery").dataset.model = currentModel;
    for (const list of $$(".result-case-list")) {
      mosaicObserver.observe(list);
    }
  }
  function selectView(view) {
    currentView = view;
    resetWarmup();
    $("#results").dataset.view = view;
    $$("[data-results-pane]").forEach((panel) => {
      panel.hidden = panel.dataset.resultsPane !== view;
    });
    $$("#results-title [data-results-view]").forEach((button) => {
      const selected = button.dataset.resultsView === view;
      button.setAttribute("aria-pressed", String(selected));
    });
    for (const panel of $$("[data-results-pane]"))
      panel.getAnimations().forEach((animation) => animation.cancel());
    if (!reduced.matches)
      $(`[data-results-pane="${view}"]`).animate(
        [{ opacity: 0 }, { opacity: 1 }],
        { duration: 200, easing: "ease-out" },
      );
    for (const state of galleryCards) state.nextAt = performance.now() + 3500;
    requestAnimationFrame(hydrateEvidence);
    requestAnimationFrame(hydrateGallery);
  }
  $$("[data-results-view]").forEach((button) =>
    button.addEventListener("click", () => {
      const view = button.dataset.resultsView;
      selectView(currentView === view && view !== "visuals" ? "visuals" : view);
    }),
  );

  function initClaimGuidance() {
    const section = $("#results");
    const claims = $$("#results-title .result-claim");
    const hover = matchMedia("(hover: hover) and (pointer: fine)");
    if (reduced.matches || !hover.matches || currentView !== "visuals") return;
    let started = false;
    const timers = [];
    const stop = () => {
      timers.forEach(clearTimeout);
      claims.forEach((claim) => claim.classList.remove("is-invited"));
      observer.disconnect();
      section.removeEventListener("pointerdown", stop);
      section.removeEventListener("focusin", stop);
      claims.forEach((claim) =>
        claim.removeEventListener("pointerenter", stop),
      );
      document.removeEventListener("visibilitychange", onVisibility);
      reduced.removeEventListener("change", stop);
      hover.removeEventListener("change", stop);
    };
    const onVisibility = () => {
      if (document.hidden) stop();
    };
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.intersectionRatio < 0.55) {
          if (started) stop();
          return;
        }
        if (started || document.hidden) return;
        started = true;
        claims.forEach((claim, i) => {
          const delay = 450 + i * 1950;
          timers.push(
            setTimeout(() => claim.classList.add("is-invited"), delay),
            setTimeout(
              () => claim.classList.remove("is-invited"),
              delay + 1700,
            ),
          );
        });
        timers.push(setTimeout(stop, 450 + claims.length * 1950));
      },
      { threshold: 0.55 },
    );
    section.addEventListener("pointerdown", stop);
    section.addEventListener("focusin", stop);
    claims.forEach((claim) => claim.addEventListener("pointerenter", stop));
    document.addEventListener("visibilitychange", onVisibility);
    reduced.addEventListener("change", stop);
    hover.addEventListener("change", stop);
    observer.observe(section);
  }
  $$("[data-result-lane]").forEach((button) =>
    button.addEventListener("click", () => {
      $("#results").dataset.lane = button.dataset.resultLane;
      resetWarmup();
      requestAnimationFrame(hydrateGallery);
      $$("[data-result-lane]").forEach((tab) =>
        tab.setAttribute("aria-selected", String(tab === button)),
      );
    }),
  );
  const evidenceDialog = $("#results-evidence-dialog");
  $$("[data-evidence-details]").forEach((button) =>
    button.addEventListener("click", () => {
      const protocol = button.dataset.evidenceDetails === "protocol";
      $("#evidence-dialog-title").textContent = protocol
        ? "Protocol & edit success"
        : "General multi-turn editing";
      $("#preservation-protocol-details").hidden = !protocol;
      $("#general-scores").hidden = protocol;
      evidenceDialog.showModal();
      document.body.classList.add("comparison-open");
    }),
  );
  $(".evidence-dialog-close").addEventListener("click", () =>
    evidenceDialog.close(),
  );
  evidenceDialog.addEventListener("close", () =>
    document.body.classList.remove("comparison-open"),
  );
  function selectModel(id) {
    currentModel = id;
    $$("[data-result-model]").forEach((b) => {
      const active = b.dataset.resultModel === id;
      b.setAttribute("aria-selected", String(active));
      b.tabIndex = active ? 0 : -1;
    });
    renderGallery();
  }
  function stepModel(delta) {
    if (!gallery) return;
    const i = gallery.models.findIndex((m) => m.id === currentModel);
    selectModel(
      gallery.models[
        (i + delta + gallery.models.length) % gallery.models.length
      ].id,
    );
  }
  $("#result-model-prev").addEventListener("click", () => stepModel(-1));
  $("#result-model-next").addEventListener("click", () => stepModel(1));
  $$("[data-result-model]").forEach((button) => {
    button.addEventListener("click", () => {
      if (gallery) selectModel(button.dataset.resultModel);
    });
    button.addEventListener("keydown", (event) => {
      if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
        event.preventDefault();
        stepModel(event.key === "ArrowLeft" ? -1 : 1);
        $(`[data-result-model="${currentModel}"]`).focus();
      }
    });
  });
  const metricData = (id) => evidence.metrics.find((m) => m.id === id);
  const format = (value, definition) =>
    Number.isFinite(value) ? value.toFixed(definition.decimals) : "—";
  function scoreTable(metrics, values, includeApi = false) {
    const table = make("table", "score-table"),
      head = make("thead"),
      header = make("tr");
    [
      "Benchmark",
      "Base model",
      "After TailSteer",
      ...(includeApi ? ["Sunburst (API)"] : []),
    ].forEach((t) => header.append(modelLabel("th", t)));
    head.append(header);
    table.append(head);
    const body = make("tbody");
    for (const m of metrics) {
      const tr = make("tr");
      const refined = make(
        "td",
        "score-refined",
        format(values.tailsteer[m.id], m),
      );
      if (!includeApi && Number.isFinite(values.deltas[m.id])) {
        const delta = values.deltas[m.id];
        refined.append(
          make(
            "small",
            "score-delta",
            `${delta > 0 ? "+" : ""}${delta.toFixed(m.decimals)}`,
          ),
        );
      }
      tr.append(
        make("td", "", m.label + (m.unit === "%" ? " (%)" : "")),
        make("td", "", format(values.base[m.id], m)),
        refined,
      );
      if (includeApi)
        tr.append(make("td", "", format(evidence.api.preservation[m.id], m)));
      body.append(tr);
    }
    table.append(body);
    return table;
  }
  function retentionTable() {
    const table = make("table", "retention-table");
    const head = make("thead"),
      tr = make("tr");
    for (const text of ["Benchmark ↑", "Base model", "After TailSteer"])
      tr.append(modelLabel("th", text));
    head.append(tr);
    table.append(head);
    for (const model of evidence.models) {
      const body = make("tbody"),
        title = make("tr", "retention-model");
      body.dataset.model = model.id;
      const label = make("th", "", model.label);
      label.colSpan = 3;
      label.scope = "rowgroup";
      label.append(make("small", "", `kₛ = ${model.onset}`));
      title.append(label);
      body.append(title);
      for (const metric of evidence.metrics.filter((m) =>
        ["t2i", "editing"].includes(m.group),
      )) {
        const row = make("tr", metric.id === "gedit_en" ? "editing-start" : "");
        row.dataset.metric = metric.id;
        const name = make(
          "th",
          "",
          metric.label + (metric.unit === "%" ? " (%)" : ""),
        );
        name.scope = "row";
        const base = make(
          "td",
          "",
          format(model.single.base[metric.id], metric),
        );
        const refined = make("td", "score-refined");
        refined.append(
          make(
            "span",
            "score-value",
            format(model.single.tailsteer[metric.id], metric),
          ),
        );
        const delta = model.single.deltas[metric.id];
        refined.append(
          make(
            "small",
            "score-delta",
            `${delta > 0 ? "+" : ""}${delta.toFixed(metric.decimals)}`,
          ),
        );
        row.append(name, base, refined);
        body.append(row);
      }
      table.append(body);
    }
    return table;
  }
  function imagePair(data) {
    const grid = make("div", "evidence-pair comparison-frames");
    grid.style.setProperty("--columns", data.columns.length);
    if (data.kind === "preservation")
      grid.style.gridTemplateColumns = data.columns
        .map((c) => `${c.aspect}fr`)
        .join(" ");
    for (const col of data.columns) {
      const fig = make("figure", "comparison-frame");
      fig.dataset.role = col.role;
      const button = make("button", "comparison-image");
      button.type = "button";
      button.setAttribute("aria-label", `Enlarge ${data.title}: ${col.label}`);
      const image = make("img", "evidence-photo");
      image.alt = `${data.title}, ${col.label}`;
      image.dataset.src = col.frames[0];
      image.decoding = "async";
      button.append(image);
      button.addEventListener("click", () => openComparison(data, 0));
      fig.append(button, modelLabel("figcaption", col.label));
      grid.append(fig);
    }
    return grid;
  }
  function hydrateEvidence() {
    images.cancel(evidenceOwner);
    if (currentView === "visuals") return;
    for (const image of $$(".results-pane:not([hidden]) .evidence-photo"))
      if (image.offsetWidth)
        images.assign(image, image.dataset.src, 0, evidenceOwner).catch(quiet);
  }
  function renderEvidence() {
    $("#capability-scores").replaceChildren(retentionTable());
    const t2iRows = [],
      exampleRows = [],
      chartRows = [];
    for (const model of evidence.models) {
      const row = make("section", "evidence-model-row");
      row.dataset.model = model.id;
      row.append(make("h4", "", model.label));
      const pairs = make("div", "t2i-pairs");
      for (const example of t2i.models[model.id]) {
        pairs.append(
          imagePair({
            id: example.id,
            title: `${model.label} · Text-to-image`,
            kind: "t2i",
            turns: [1],
            prompts: [example.displayPrompt || example.prompt],
            promptLabel: example.promptLabel,
            columns: [
              {
                label: "Base model",
                role: "base",
                frames: [example.images.base],
              },
              {
                label: "After TailSteer",
                role: "tailsteer",
                frames: [example.images.tailsteer],
              },
            ],
          }),
        );
      }
      row.append(pairs);
      t2iRows.push(row);
      const ex = evidence.examples.find((e) => e.modelId === model.id);
      const exampleRow = make("section", "evidence-model-row");
      exampleRow.dataset.model = model.id;
      const label = make("h4", "", model.label);
      label.append(make("small", "", "Turn 10"));
      exampleRow.append(
        label,
        imagePair({
          id: `preservation-${model.id}-${ex.id}`,
          title: ex.label,
          kind: "preservation",
          turns: [10],
          prompts: [ex.prompt],
          columns: [
            {
              label: "GPT-2.5-Sunburst (API)",
              role: "api",
              frames: [ex.api],
              aspect: ex.aspects.api,
            },
            {
              label: "Base model",
              role: "base",
              frames: [ex.models[model.id].base],
              aspect: ex.aspects.base,
            },
            {
              label: "After TailSteer (ours)",
              role: "tailsteer",
              frames: [ex.models[model.id].tailsteer],
              aspect: ex.aspects.tailsteer,
            },
          ],
        }),
      );
      exampleRows.push(exampleRow);
      const chartRow = make("section", "curve-model");
      chartRow.dataset.model = model.id;
      const chartLabel = make("h4", "", model.label);
      chartLabel.append(make("small", "", `kₛ = ${model.onset}`));
      const charts = make("div", "curve-row");
      for (const metric of ["psnr", "ssim", "lpips"])
        charts.append(curveChart(model, metric));
      chartRow.append(chartLabel, charts);
      chartRows.push(chartRow);
    }
    $("#capability-examples").replaceChildren(...t2iRows);
    $("#preservation-examples").replaceChildren(...exampleRows);
    $("#preservation-charts").replaceChildren(...chartRows);
    const legend = $("#preservation-legend");
    legend.replaceChildren();
    for (const [label, color] of [
      ["Base model", "#9ca9ba"],
      ["GPT-2.5-Sunburst (API)", "#bd8c63"],
      ["After TailSteer", "#692568"],
    ]) {
      const item = make("span"),
        swatch = make("i");
      swatch.style.background = color;
      item.append(swatch, modelLabel("span", label));
      legend.append(item);
    }
    const general = $("#general-scores"),
      details = $("#preservation-protocol-details");
    general.replaceChildren(
      make(
        "p",
        "",
        evidence.protocol.general.mse + " " + evidence.protocol.general.edival,
      ),
    );
    details.replaceChildren(make("p", "", evidence.protocol.coa.caption));
    for (const model of evidence.models) {
      general.append(
        make("h4", "", model.label),
        scoreTable(
          evidence.metrics.filter((m) => m.group === "general"),
          model.general,
        ),
      );
      details.append(
        make("h4", "", model.label),
        scoreTable([metricData("coa")], model.preservation, true),
      );
    }
    for (const text of [
      evidence.protocol.curves.caption,
      evidence.protocol.coa.caveat,
      evidence.protocol.reference + " " + evidence.protocol.mask,
      evidence.protocol.table.caption,
      evidence.protocol.api.caption,
      evidence.protocol.api.caveat,
    ])
      details.append(make("p", "results-source-note", text));
  }
  function curveChart(model, metric) {
    const definition = metricData(metric);
    const series = [
      {
        id: "base",
        name: "Base model",
        color: "#9ca9ba",
        points: model.preservation.curves[metric].base,
      },
      {
        id: "api",
        name: "GPT-2.5",
        color: "#bd8c63",
        points: evidence.api.curves[metric],
      },
      {
        id: "tailsteer",
        name: "After TailSteer",
        color: "#692568",
        points: model.preservation.curves[metric].tailsteer,
      },
    ];
    const tile = make("div", "curve-tile");
    tile.dataset.metric = metric;
    tile.append(
      make(
        "h5",
        "",
        `${definition.label}${metric === "psnr" ? " (dB)" : ""} ${definition.direction === "up" ? "↑" : "↓"}`,
      ),
    );
    const values = series.flatMap((s) => s.points.map((p) => p.value));
    const min = Math.min(...values),
      max = Math.max(...values),
      pad = (max - min) * 0.1 || 0.1;
    const lo = Math.max(0, min - pad),
      hi = max + pad;
    const W = 240,
      H = 160,
      L = 34,
      R = 10,
      T = 10,
      B = 30;
    const x = (t) => L + ((t - 1) / 9) * (W - L - R),
      y = (v) => T + ((hi - v) / (hi - lo)) * (H - T - B);
    function node(tag, attrs, text) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
      if (text !== undefined) el.textContent = text;
      return el;
    }
    const svg = node("svg", {
      viewBox: `0 0 ${W} ${H}`,
      role: "img",
      tabindex: "0",
      "aria-label": `${model.label}, ${definition.label} over ten editing turns; base, GPT-2.5 and TailSteer onset ${model.onset}.`,
    });
    for (let i = 0; i < 4; i++) {
      const value = lo + ((hi - lo) * i) / 3;
      svg.append(
        node("line", {
          x1: L,
          x2: W - R,
          y1: y(value),
          y2: y(value),
          class: "chart-grid",
        }),
        node(
          "text",
          {
            x: L - 6,
            y: y(value) + 4,
            "text-anchor": "end",
            class: "chart-axis-label",
          },
          value.toFixed(metric === "psnr" ? 0 : 2),
        ),
      );
    }
    for (const turn of [1, 5, 10])
      svg.append(
        node(
          "text",
          {
            x: x(turn),
            y: H - 14,
            "text-anchor": "middle",
            class: "chart-axis-label",
          },
          turn,
        ),
      );
    svg.append(
      node(
        "text",
        {
          x: (L + W - R) / 2,
          y: H - 1,
          "text-anchor": "middle",
          class: "chart-axis-label chart-axis-title",
        },
        "Editing turn",
      ),
    );
    for (const s of series) {
      const path = s.points
        .map(
          (p, i) =>
            `${i ? "L" : "M"}${x(p.turn).toFixed(2)},${y(p.value).toFixed(2)}`,
        )
        .join(" ");
      svg.append(
        node("path", {
          d: path,
          fill: "none",
          stroke: s.color,
          "stroke-width": s.id === "tailsteer" ? 2.6 : 2,
          "stroke-linecap": "round",
          "stroke-linejoin": "round",
          "data-series": s.id,
        }),
      );
    }
    const guide = node("line", {
      y1: T,
      y2: H - B,
      stroke: "#bdcadd",
      "stroke-dasharray": "3 4",
      opacity: 0,
    });
    svg.append(guide);
    const tooltip = make("div", "curve-tooltip");
    function inspect(turn) {
      guide.setAttribute("x1", x(turn));
      guide.setAttribute("x2", x(turn));
      guide.setAttribute("opacity", 1);
      tooltip.replaceChildren(make("b", "", `Turn ${turn}`));
      for (const s of series) {
        const line = make(
          "span",
          "",
          `${s.name}  ${format(s.points[turn - 1].value, definition)}`,
        );
        line.style.color = s.color;
        tooltip.append(line);
      }
      tile.dataset.inspect = "true";
    }
    svg.addEventListener("pointermove", (event) => {
      const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(
        svg.getScreenCTM().inverse(),
      );
      inspect(
        Math.max(
          1,
          Math.min(10, Math.round(1 + ((p.x - L) / (W - L - R)) * 9)),
        ),
      );
    });
    svg.addEventListener("focus", () => inspect(10));
    for (const event of ["pointerleave", "blur"])
      svg.addEventListener(event, () => {
        delete tile.dataset.inspect;
        guide.setAttribute("opacity", 0);
      });
    tile.append(svg, tooltip);
    return tile;
  }
  $$("[data-evidence-panel]").forEach((button) =>
    button.addEventListener("click", () => {
      const pane = button.closest(".results-pane");
      $(".evidence-split", pane).dataset.panel = button.dataset.evidencePanel;
      $$("[data-evidence-panel]", pane).forEach((b) =>
        b.setAttribute("aria-pressed", String(b === button)),
      );
      requestAnimationFrame(hydrateEvidence);
    }),
  );
  async function start() {
    try {
      [gallery, evidence, t2i] = await Promise.all(
        ["gallery", "evidence", "t2i"].map(async (name) => {
          const response = await fetch(`assets/results/${name}.json`, {
            cache: "no-cache",
          });
          if (!response.ok) throw Error(`Cannot load ${name}`);
          return response.json();
        }),
      );
      renderEvidence();
      selectModel(currentModel);
      const initialView = {
        "#results-capabilities": "capabilities",
        "#results-preservation": "preservation",
      }[location.hash];
      if (initialView) {
        selectView(initialView);
        requestAnimationFrame(() =>
          $("#results").scrollIntoView({ behavior: "instant", block: "start" }),
        );
      }
      initClaimGuidance();
    } catch (error) {
      $("#results-gallery").replaceChildren(
        make(
          "p",
          "results-loading",
          "The comparisons could not be loaded. Please refresh the page.",
        ),
      );
      console.error(error);
    }
  }
  new ResizeObserver(() => {
    if (evidence && currentView !== "visuals") hydrateEvidence();
  }).observe($("#results-stage"));
  new IntersectionObserver(
    ([entry]) => {
      resultsNear = entry.isIntersecting;
      if (resultsNear) scheduleWarmup();
      else resetWarmup();
    },
    { rootMargin: "150px 0px" },
  ).observe($("#results"));
  start();
})();
