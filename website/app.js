/* Interactive trajectory schematics and measured editing results. */
(() => {
  "use strict";
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
  const smooth = (v) => {
    v = clamp(v);
    return v * v * (3 - 2 * v);
  };
  const svgNS = "http://www.w3.org/2000/svg";
  const node = (tag, attrs, text) => {
    const n = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const attr = (el, attrs) => {
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  };
  const show = (selector, visible) => {
    $(selector).style.opacity = visible ? "1" : "0";
  };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  let paused = reduced.matches;
  let activeChapter = "home";
  let visible = !document.hidden;
  let heroTime = 0;

  // One native scrolling document. The floating touchbar follows the reading position.
  const chapters = $$(".chapter");
  const touchbar = $(".chapters");
  const chapterLinks = $$(".chapters a, #mobile-navigation a");
  // Reserve the actual card height for narrow screens and short viewports.
  const identityCard = $(".identity-card");
  new ResizeObserver(() => {
    document.documentElement.style.setProperty(
      "--identity-height",
      `${identityCard.getBoundingClientRect().height}px`,
    );
  }).observe(identityCard);
  let scrollQueued = false;
  function updateChapter() {
    const readingLine = Math.min(innerHeight * 0.32, 280);
    const atEnd =
      scrollY + innerHeight >= document.documentElement.scrollHeight - 2;
    // Keep the last chapter selected below its content, including in the footer.
    const current = atEnd
      ? chapters[chapters.length - 1]
      : [...chapters]
          .reverse()
          .find(
            (section) => section.getBoundingClientRect().top <= readingLine,
          ) || chapters[0];
    activeChapter = current.id;
    chapterLinks.forEach((a) => {
      const active = a.hash === "#" + activeChapter;
      a.classList.toggle("active", active);
      if (active) a.setAttribute("aria-current", "location");
      else a.removeAttribute("aria-current");
    });
    const activeLink = $("a.active", touchbar);
    touchbar.style.setProperty("--active-y", `${activeLink.offsetTop}px`);
    touchbar.style.setProperty(
      "--active-height",
      `${activeLink.offsetHeight}px`,
    );
    const bounds = current.getBoundingClientRect();
    touchbar.style.setProperty(
      "--chapter-progress",
      atEnd ? 1 : clamp((readingLine - bounds.top) / bounds.height),
    );
    scrollQueued = false;
  }
  function queueChapterUpdate() {
    if (!scrollQueued) {
      scrollQueued = true;
      requestAnimationFrame(updateChapter);
    }
  }
  addEventListener("scroll", queueChapterUpdate, { passive: true });
  addEventListener("resize", queueChapterUpdate);
  new ResizeObserver(queueChapterUpdate).observe($("#main"));
  document.fonts.ready.then(queueChapterUpdate);
  document.addEventListener("visibilitychange", () => {
    visible = !document.hidden;
  });
  const revealObserver = new IntersectionObserver(
    (entries) =>
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("visible");
          revealObserver.unobserve(e.target);
        }
      }),
    { threshold: 0.08 },
  );
  $$(".reveal").forEach((el) => revealObserver.observe(el));
  const menu = $(".mobile-menu");
  menu.addEventListener("click", () => {
    const open = menu.getAttribute("aria-expanded") !== "true";
    menu.setAttribute("aria-expanded", String(open));
    menu.setAttribute(
      "aria-label",
      open ? "Close chapter navigation" : "Open chapter navigation",
    );
    $("#mobile-navigation").hidden = !open;
  });
  $$("#mobile-navigation a").forEach((a) =>
    a.addEventListener("click", () => {
      menu.setAttribute("aria-expanded", "false");
      $("#mobile-navigation").hidden = true;
    }),
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      menu.setAttribute("aria-expanded", "false");
      $("#mobile-navigation").hidden = true;
    }
  });
  function updateMotionState() {
    document.body.classList.toggle("motion-paused", paused);
  }
  reduced.addEventListener("change", (e) => {
    paused = e.matches;
    if (paused) renderStatic();
    updateMotionState();
    carouselButton();
    trainingButton();
  });
  updateMotionState();

  // A hoverable model list with click, touch, and keyboard disclosure.
  const models = $(".resource-models");
  const modelsTrigger = $("#models-trigger");
  const modelLinks = $("#model-links");
  let modelsPinned = false;
  let modelCloseTimer;
  function positionModels() {
    models.classList.toggle(
      "opens-below",
      models.getBoundingClientRect().right + modelLinks.offsetWidth + 28 >
        innerWidth,
    );
  }
  function showModels(open) {
    clearTimeout(modelCloseTimer);
    if (open) positionModels();
    models.classList.toggle("is-open", open);
    modelsTrigger.setAttribute("aria-expanded", String(open));
    modelLinks.inert = !open;
  }
  models.addEventListener("pointerenter", (e) => {
    if (e.pointerType === "mouse") showModels(true);
  });
  models.addEventListener("pointerleave", () => {
    if (!modelsPinned && !models.contains(document.activeElement))
      modelCloseTimer = setTimeout(() => showModels(false), 160);
  });
  models.addEventListener("focusin", () => showModels(true));
  models.addEventListener("focusout", (e) => {
    if (!models.contains(e.relatedTarget)) {
      modelsPinned = false;
      showModels(false);
    }
  });
  modelsTrigger.addEventListener("click", () => {
    modelsPinned = !modelsPinned;
    showModels(modelsPinned);
  });
  models.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      modelsPinned = false;
      modelsTrigger.focus();
      showModels(false);
    } else if (e.key === "ArrowDown" && e.target === modelsTrigger) {
      e.preventDefault();
      showModels(true);
      $("a", modelLinks).focus();
    }
  });
  document.addEventListener("pointerdown", (e) => {
    if (!models.contains(e.target)) {
      modelsPinned = false;
      showModels(false);
    }
  });
  addEventListener("resize", () => {
    if (models.classList.contains("is-open")) positionModels();
  });

  // The original curve stays fixed. Split it at the onset so every steered
  // prefix is exactly a portion of the original trajectory, even while dragging.
  const heroStage = $(".hero-stage");
  const heroDiagram = $(".hero-diagram");
  const heroBase = $("#hero-base");
  const heroPath = $("#hero-path");
  const heroOnset = $("#hero-onset");
  const heroTooltip = $("#hero-trajectory-tooltip");
  const heroRegionLabels = $$("[data-hero-region]", heroDiagram);
  const heroSegments = [
    [
      { x: -20, y: 475 },
      { x: 210, y: 562 },
      { x: 420, y: 505 },
      { x: 670, y: 437 },
    ],
    [
      { x: 670, y: 437 },
      { x: 826, y: 395 },
      { x: 964, y: 263 },
      { x: 1150, y: 160 },
    ],
  ];
  const heroEchoes = [];
  for (let i = -4; i <= 4; i++) {
    if (!i) continue;
    const echo = node("path", {});
    $("#hero-echoes").append(echo);
    heroEchoes.push({ element: echo, offset: i });
  }
  let heroLength = heroPath.getTotalLength();
  let heroDisplayedTime = 0;
  const heroOnsetMinX = 610;
  // Leave the last 1/20 of the original curve unavailable to the onset.
  // Arc length makes the limit follow the curve rather than the viewport width.
  const heroOnsetMaxX = heroBase.getPointAtLength(
    heroBase.getTotalLength() * 0.95,
  ).x;
  const heroOnsetSpan = heroOnsetMaxX - heroOnsetMinX;
  const heroDefaultOnset = (100 * (670 - heroOnsetMinX)) / heroOnsetSpan;
  let heroOnsetValue = heroDefaultOnset;
  let heroDrag = null;
  let heroFocusedCurve = null;
  let heroFocusedRegion = null;
  let heroSplitX = 670;
  const heroMix = (a, b, t) => ({
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
  });
  function splitHeroCurve(curve, t) {
    const a = heroMix(curve[0], curve[1], t);
    const b = heroMix(curve[1], curve[2], t);
    const c = heroMix(curve[2], curve[3], t);
    const d = heroMix(a, b, t);
    const e = heroMix(b, c, t);
    return [curve[0], a, d, heroMix(d, e, t)];
  }
  function heroAnchorAt(x) {
    const index = x <= 670 ? 0 : 1;
    const curve = heroSegments[index];
    let lo = 0,
      hi = 1;
    for (let i = 0; i < 28; i++) {
      const t = (lo + hi) / 2;
      if (splitHeroCurve(curve, t)[3].x < x) lo = t;
      else hi = t;
    }
    const left = splitHeroCurve(curve, x === 670 ? 1 : (lo + hi) / 2);
    return { left, index, point: left[3] };
  }
  const heroXY = (p) => `${p.x.toFixed(3)} ${p.y.toFixed(3)}`;
  const heroCubic = (c) => `C ${c.slice(1).map(heroXY).join(" ")}`;
  function sampleHeroCurve(curve) {
    const length = curve.getTotalLength();
    const count = Math.ceil(length / 5);
    return Array.from({ length: count + 1 }, (_, i) =>
      curve.getPointAtLength((length * i) / count),
    );
  }
  const heroSamples = { base: sampleHeroCurve(heroBase), steered: [] };
  function highlightHeroRegion(region) {
    if (region) heroStage.dataset.region = region;
    else delete heroStage.dataset.region;
  }
  function positionHeroRegionLabels() {
    for (const label of heroRegionLabels) {
      const text = $("text", label);
      const halfWidth = text.getComputedTextLength() / 2 + 12;
      const center =
        label.dataset.heroRegion === "prior"
          ? (heroSplitX - 20) / 2
          : (heroSplitX + 1150) / 2;
      const x = clamp(center, -20 + halfWidth, 1150 - halfWidth);
      text.setAttribute("x", x);
      attr($("rect", label), { x: x - halfWidth, width: halfWidth * 2 });
    }
  }
  function animateHero(t) {
    heroDisplayedTime = t;
    const progress = (t / 9) % 1;
    const p = heroPath.getPointAtLength(heroLength * progress);
    attr($("#hero-particle"), {
      cx: p.x,
      cy: p.y,
      opacity: smooth(progress * 12) * smooth((1 - progress) * 12),
    });
  }
  function setHeroOnset(value) {
    heroOnsetValue = clamp(value, 0, 100);
    const {
      left,
      index,
      point: q,
    } = heroAnchorAt(heroOnsetMinX + (heroOnsetSpan * heroOnsetValue) / 100);
    const tangent = { x: q.x - left[2].x, y: q.y - left[2].y };
    // Shorten both tail handles as the onset advances. This keeps x monotone,
    // preserves the original tangent, and prevents a late tail from looping back.
    const tailScale = Math.min(1, (1150 - q.x) / (1150 - 670));
    const handle = 170 * tailScale;
    const control = {
      x: q.x + handle,
      y: q.y + (handle * tangent.y) / tangent.x,
    };
    const landingControl = { x: 1150 - 165 * tailScale, y: 515 };
    const prefix = index
      ? `${heroCubic(heroSegments[0])} ${heroCubic(left)}`
      : heroCubic(left);
    // Region geometry shares the exact onset with the trajectory; no fixed
    // viewport split, so dragging and responsive scaling keep them aligned.
    heroSplitX = q.x;
    attr($("#hero-prior-mask"), { width: q.x + 60 });
    attr($("#hero-transition-mask"), { x: q.x, width: 1240 - q.x });
    $("#hero-prior-wash").setAttribute(
      "d",
      `M -20 475 ${prefix} L ${q.x} 640 L -20 640 Z`,
    );
    const baseTail = heroSamples.base.filter((p) => p.x > q.x);
    $("#hero-transition-wash").setAttribute(
      "d",
      `M ${heroXY(q)} ${baseTail.map((p) => `L ${heroXY(p)}`).join(" ")} L 1150 640 L ${q.x} 640 Z`,
    );
    positionHeroRegionLabels();
    heroPath.setAttribute(
      "d",
      `M -20 475 ${prefix} C ${heroXY(control)} ${heroXY(landingControl)} 1150 570`,
    );
    heroLength = heroPath.getTotalLength();
    heroSamples.steered = sampleHeroCurve(heroPath);
    for (const { element, offset: i } of heroEchoes) {
      const offsetCurve = (curve, offsets) =>
        curve.map((p, j) => ({ x: p.x, y: p.y + i * offsets[j] }));
      const echoPrefix = index
        ? `${heroCubic(offsetCurve(heroSegments[0], [13, 10, 9, 7]))} ${heroCubic(offsetCurve(left, [7, 7, 7, 7]))}`
        : heroCubic(offsetCurve(left, [13, 10, 9, 7]));
      element.setAttribute(
        "d",
        `M -20 ${475 + i * 13} ${echoPrefix} C ${control.x} ${control.y + i * 15} ${landingControl.x} ${landingControl.y + i * 16} 1150 ${570 + i * 15}`,
      );
    }
    attr(heroOnset, {
      transform: `translate(${q.x} ${q.y})`,
      "aria-valuenow": heroOnsetValue.toFixed(1),
      "aria-valuetext":
        Math.abs(heroOnsetValue - heroDefaultOnset) < 0.5
          ? "Default steering onset"
          : heroOnsetValue === 100
            ? "Latest steering onset; final five percent reserved"
            : heroOnsetValue < heroDefaultOnset
              ? "Earlier steering onset"
              : "Later steering onset",
    });
    // Keep both labels inside the drawing when the marker nears its right edge.
    $$("text", heroOnset).forEach((text) =>
      text.setAttribute("x", Math.min(0, 1060 - q.x)),
    );
    animateHero(heroDisplayedTime);
  }
  const heroRange = Array.from({ length: 65 }, (_, i) => {
    const p = heroAnchorAt(heroOnsetMinX + (heroOnsetSpan * i) / 64).point;
    return `${i ? "L" : "M"} ${p.x} ${p.y - 24}`;
  }).join(" ");
  $("#hero-onset-range").setAttribute("d", heroRange);
  setHeroOnset(heroDefaultOnset);

  const heroPointer = (event) =>
    new DOMPoint(event.clientX, event.clientY).matrixTransform(
      heroDiagram.getScreenCTM().inverse(),
    );
  function heroDistance(points, p) {
    let nearest = Infinity;
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1],
        b = points[i];
      const dx = b.x - a.x,
        dy = b.y - a.y;
      const t = clamp(
        ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy),
      );
      nearest = Math.min(
        nearest,
        Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy),
      );
    }
    return nearest;
  }
  function highlightHero(kinds, clientX = 0, clientY = 0) {
    for (const [kind, curve] of [
      ["base", heroBase],
      ["steered", heroPath],
    ]) {
      curve.classList.toggle("is-highlighted", kinds.includes(kind));
      $(`[data-trajectory="${kind}"]`, heroTooltip).hidden =
        !kinds.includes(kind);
      if (kinds.includes(kind))
        curve.setAttribute("aria-describedby", heroTooltip.id);
      else curve.removeAttribute("aria-describedby");
    }
    $(".hero-tooltip-separator", heroTooltip).hidden = kinds.length !== 2;
    heroTooltip.hidden = kinds.length === 0;
    if (!kinds.length) return;
    const bounds = heroStage.getBoundingClientRect();
    const width = heroTooltip.offsetWidth,
      height = heroTooltip.offsetHeight;
    heroTooltip.style.left = `${clamp(clientX - bounds.left + 16, 0, Math.max(0, bounds.width - width))}px`;
    heroTooltip.style.top = `${Math.max(6, clientY - bounds.top - height - 18)}px`;
  }
  function focusHeroCurve(kind) {
    const curve = kind === "base" ? heroBase : heroPath;
    const local = curve.getPointAtLength(curve.getTotalLength() * 0.8);
    const p = new DOMPoint(local.x, local.y).matrixTransform(
      curve.getScreenCTM(),
    );
    highlightHero([kind], p.x, p.y);
    highlightHeroRegion(local.x <= heroSplitX ? "prior" : "transition");
  }
  function hoverHero(event) {
    if (heroDrag || heroOnset.contains(event.target)) {
      highlightHero([]);
      highlightHeroRegion("transition");
      return;
    }
    const p = heroPointer(event);
    const scale = Math.hypot(
      heroDiagram.getScreenCTM().a,
      heroDiagram.getScreenCTM().b,
    );
    // The hover zone follows the upper curve and extends to the region labels,
    // leaving the title, authors, buttons, and surrounding whitespace alone.
    const label = event.target.closest("[data-hero-region]");
    const right = heroSamples.base.findIndex((point) => point.x >= p.x);
    let region = label?.dataset.heroRegion || null;
    if (!region && right > 0 && p.y <= 684) {
      const a = heroSamples.base[right - 1],
        b = heroSamples.base[right];
      const y = a.y + ((b.y - a.y) * (p.x - a.x)) / (b.x - a.x);
      if (p.y >= y - 18 / scale)
        region = p.x <= heroSplitX ? "prior" : "transition";
    }
    highlightHeroRegion(region || heroFocusedRegion);
    const base = heroDistance(heroSamples.base, p) * scale;
    const steered = heroDistance(heroSamples.steered, p) * scale;
    const near = Math.min(base, steered) <= 14;
    const kinds = !near
      ? []
      : Math.abs(base - steered) <= 3 && Math.max(base, steered) <= 14
        ? ["base", "steered"]
        : [base < steered ? "base" : "steered"];
    if (!kinds.length && heroFocusedCurve) focusHeroCurve(heroFocusedCurve);
    else highlightHero(kinds, event.clientX, event.clientY);
  }
  heroStage.addEventListener("pointermove", (e) => {
    if (e.pointerType === "mouse" || e.pointerType === "pen") hoverHero(e);
  });
  heroStage.addEventListener("pointerleave", (e) => {
    if (e.pointerType === "touch") return;
    if (heroFocusedCurve) focusHeroCurve(heroFocusedCurve);
    else {
      highlightHero([]);
      highlightHeroRegion(heroFocusedRegion);
    }
  });
  heroStage.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "touch") hoverHero(e);
  });
  document.addEventListener("pointerdown", (e) => {
    if (!heroStage.contains(e.target)) {
      highlightHero([]);
      highlightHeroRegion(null);
    }
  });
  for (const label of heroRegionLabels) {
    label.addEventListener("focus", () => {
      if (!label.matches(":focus-visible")) return;
      heroFocusedRegion = label.dataset.heroRegion;
      highlightHero([]);
      highlightHeroRegion(heroFocusedRegion);
    });
    label.addEventListener("blur", () => {
      heroFocusedRegion = null;
      highlightHeroRegion(null);
    });
    label.addEventListener("keydown", (e) => {
      if (e.key === "Escape") label.blur();
    });
  }
  for (const [kind, curve] of [
    ["base", heroBase],
    ["steered", heroPath],
  ]) {
    curve.addEventListener("focus", () => {
      if (curve.matches(":focus-visible")) {
        heroFocusedCurve = kind;
        focusHeroCurve(kind);
      }
    });
    curve.addEventListener("blur", () => {
      heroFocusedCurve = null;
      highlightHero([]);
      highlightHeroRegion(null);
    });
    curve.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        heroFocusedCurve = null;
        highlightHero([]);
        highlightHeroRegion(null);
      }
    });
  }
  heroOnset.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || heroDrag) return;
    e.preventDefault();
    heroOnset.focus({ preventScroll: true });
    heroDrag = { id: e.pointerId, x: heroPointer(e).x, value: heroOnsetValue };
    heroOnset.setPointerCapture(e.pointerId);
    heroOnset.classList.add("is-dragging");
    highlightHero([]);
    highlightHeroRegion("transition");
  });
  heroOnset.addEventListener("pointermove", (e) => {
    if (!heroDrag || e.pointerId !== heroDrag.id) return;
    setHeroOnset(
      heroDrag.value + (100 * (heroPointer(e).x - heroDrag.x)) / heroOnsetSpan,
    );
  });
  function endHeroDrag(e) {
    if (!heroDrag || e.pointerId !== heroDrag.id) return;
    heroDrag = null;
    heroOnset.classList.remove("is-dragging");
    if (heroOnset.hasPointerCapture(e.pointerId))
      heroOnset.releasePointerCapture(e.pointerId);
  }
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"])
    heroOnset.addEventListener(event, endHeroDrag);
  heroOnset.addEventListener("dblclick", () => setHeroOnset(heroDefaultOnset));
  heroOnset.addEventListener("keydown", (e) => {
    const delta = e.shiftKey ? 10 : 2;
    const values = {
      ArrowLeft: heroOnsetValue - delta,
      ArrowDown: heroOnsetValue - delta,
      ArrowRight: heroOnsetValue + delta,
      ArrowUp: heroOnsetValue + delta,
      Home: 0,
      End: 100,
      Escape: heroDefaultOnset,
    };
    if (Object.hasOwn(values, e.key)) {
      e.preventDefault();
      setHeroOnset(values[e.key]);
    }
  });
  // A filled circle gives reliable hit testing; keep its radius at 40 screen pixels.
  function sizeHeroOnsetHitArea() {
    const matrix = heroDiagram.getScreenCTM();
    $(".hero-onset-hit", heroOnset).setAttribute(
      "r",
      40 / Math.hypot(matrix.a, matrix.b),
    );
  }
  sizeHeroOnsetHitArea();
  addEventListener("resize", () => {
    highlightHero([]);
    highlightHeroRegion(null);
    positionHeroRegionLabels();
    sizeHeroOnsetHitArea();
  });
  document.fonts.ready.then(positionHeroRegionLabels);

  // Motivation carousel. Timers advance only while this chapter is in view.
  let scene = 0,
    sceneTime = 0,
    carouselPaused = false,
    frequencyManual = null;
  let motivationInView = false;
  new IntersectionObserver(([entry]) => {
    motivationInView = entry.isIntersecting && entry.intersectionRatio >= 0.2;
  }, { threshold: 0.2 }).observe($(".motivation-stage"));
  const durations = [window.TailSteerDrift.duration + 3, 18, 16, 22];
  const panels = $$(".scene"),
    tabs = $$(".scene-tabs button");
  function carouselButton() {
    const stopped = paused || carouselPaused;
    $("#carousel-play").textContent = stopped ? "▷" : "Ⅱ";
    $("#carousel-play").setAttribute(
      "aria-label",
      stopped ? "Play motivation autoplay" : "Pause motivation autoplay",
    );
    $("#carousel-play").setAttribute("aria-pressed", String(stopped));
  }
  function selectScene(index, focus = false) {
    clearCurationInspect();
    scene = (index + panels.length) % panels.length;
    sceneTime = 0;
    frequencyManual = null;
    panels.forEach((p, i) => {
      p.hidden = i !== scene;
      p.classList.toggle("active", i === scene);
    });
    tabs.forEach((t, i) => {
      t.classList.toggle("selected", i === scene);
      t.setAttribute("aria-selected", String(i === scene));
      t.tabIndex = i === scene ? 0 : -1;
    });
    $("#scene-count").textContent = `0${scene + 1} / 0${panels.length}`;
    if (focus) tabs[scene].focus();
    if (scene === 2) curationGeometry = null;
    sceneTime = paused || carouselPaused ? durations[scene] * 0.85 : 0;
    renderScene(sceneTime);
  }
  tabs.forEach((t, i) => t.addEventListener("click", () => selectScene(i)));
  $(".scene-tabs").addEventListener("keydown", (e) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
      e.preventDefault();
      carouselPaused = true;
      carouselButton();
      selectScene(
        e.key === "Home"
          ? 0
          : e.key === "End"
            ? panels.length - 1
            : scene + (e.key === "ArrowRight" ? 1 : -1),
        true,
      );
    }
  });
  $("#scene-prev").addEventListener("click", () => selectScene(scene - 1));
  $("#scene-next").addEventListener("click", () => selectScene(scene + 1));
  $("#carousel-play").addEventListener("click", () => {
    if (paused) {
      paused = false;
      updateMotionState();
      carouselPaused = false;
      trainingButton();
    } else carouselPaused = !carouselPaused;
    frequencyManual = null;
    carouselButton();
  });
  $("#scene-drift").addEventListener("driftseek", ({ detail }) => {
    sceneTime = detail.time;
    carouselPaused = !detail.play || reduced.matches;
    if (detail.play && !reduced.matches) {
      paused = false;
      updateMotionState();
    }
    carouselButton();
    renderScene(sceneTime);
  });
  $$("[data-drift-next]").forEach((button) =>
    button.addEventListener("click", () => selectScene(1, true)),
  );
  let swipeX = null;
  $(".motivation-stage").addEventListener(
    "touchstart",
    (e) => {
      swipeX = e.target.closest("input,button,.drift-viewport") ? null : e.touches[0].clientX;
    },
    { passive: true },
  );
  $(".motivation-stage").addEventListener(
    "touchend",
    (e) => {
      if (swipeX !== null) {
        const dx = e.changedTouches[0].clientX - swipeX;
        if (Math.abs(dx) > 70) selectScene(scene + (dx < 0 ? 1 : -1));
      }
      swipeX = null;
    },
    { passive: true },
  );
  function renderScene(t) {
    if (scene === 0) window.TailSteerDrift.render(t);
    $("#carousel-progress").style.width =
      `${clamp(t / durations[scene]) * 100}%`;
    if (scene === 1)
      renderFrequency(
        frequencyManual === null
          ? (frequencyFrames[0]?.step ?? 0) +
              (50 - (frequencyFrames[0]?.step ?? 0)) * smooth(clamp(t / 13))
          : frequencyManual,
      );
    if (scene === 2) renderCuration(t);
    if (scene === 3) renderTerminal(t);
  }

  // Measured frequency data, with actual cached trajectory images.
  let frequencyFrames = [],
    frequencyReady = false;
  let msePoints = new Map(),
    mseMax = 0.16,
    mseTailMax = 0.01;
  const imageQueue = window.TailSteerImages;
  const frequencyOwner = {},
    frequencyIdleOwner = {};
  let frequencyNear = false,
    frequencyWarming = false;
  const loadImage = (src, priority = 1, owner = "illustrations") =>
    imageQueue.load(src, priority, owner);
  function warmFrequency() {
    if (!frequencyNear || frequencyWarming || !frequencyReady) return;
    frequencyWarming = true;
    for (const frame of frequencyFrames) {
      imageQueue.warm(frame.state, frequencyIdleOwner);
      imageQueue.warm(frame.error, frequencyIdleOwner);
    }
  }
  function assetFailure(error) {
    if (error.name !== "AbortError") console.error(error);
  }
  const stepX = (step) => 42 + (step / 50) * 344;
  const similarityY = (similarity) => 222 - similarity * 202;
  const mseY = (value) => 172 - (value / mseMax) * 162;
  const grid = $("#frequency-grid");
  for (let i = 0; i <= 4; i++) {
    const y = similarityY(i / 4);
    grid.append(
      node("line", {
        x1: 42,
        y1: y,
        x2: 386,
        y2: y,
        stroke: "#e5e9ee",
        "stroke-width": 1,
      }),
    );
    grid.append(
      node(
        "text",
        { x: 34, y: y + 4, "text-anchor": "end", class: "svg-small" },
        (i * 0.25).toFixed(i === 0 || i === 4 ? 0 : 2),
      ),
    );
  }
  for (let s = 0; s <= 50; s += 10)
    grid.append(
      node(
        "text",
        {
          x: stepX(s),
          y: 244,
          "text-anchor": "middle",
          class: "svg-small",
        },
        s,
      ),
    );
  Promise.all(
    ["frequency-sequence.json", "frequency-mse.json"].map(async (name) => {
      const response = await fetch(`assets/${name}`, { cache: "no-cache" });
      if (!response.ok) throw Error(`Frequency data unavailable: ${name}`);
      return response.json();
    }),
  )
    .then(([data, mse]) => {
      if (
        mse.sequence_data_sha256 !== data.data_sha256 ||
        mse.roi_sha256 !== data.measurement.roi_sha256 ||
        mse.values.length !== data.frames.length ||
        mse.values.some(
          (v, i) =>
            v.step !== data.frames[i].step ||
            !Number.isFinite(v.mse) ||
            v.mse < 0,
        )
      )
        throw Error("MSE must match the measured frequency trajectory");
      frequencyFrames = data.frames;
      prepareMSE(mse.values);
      positionFrequencyLink();
      const { colors } = data;
      for (const [band, values] of Object.entries(data.bands)) {
        const d = values
          .map(
            (v, i) =>
              `${i ? "L" : "M"}${stepX(v.step)},${similarityY(v.similarity)}`,
          )
          .join(" ");
        $("#frequency-curves").append(
          node("path", {
            d,
            stroke: colors[band],
            "stroke-width": 3,
            fill: "none",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          }),
        );
        for (const v of values)
          $("#frequency-curves").append(
            node("circle", {
              cx: stepX(v.step),
              cy: similarityY(v.similarity),
              r: 2,
              fill: colors[band],
            }),
          );
      }
      $("#frequency-progress").min = frequencyFrames[0].step;
      $("#frequency-progress").disabled = false;
      renderFrequency(frequencyManual ?? (paused || carouselPaused ? 50 : 0));
    })
    .catch(assetFailure);

  // Both MSE axes start at zero; the tail inset magnifies measured steps 48–50.
  function prepareMSE(values) {
    msePoints = new Map(values.map((v) => [v.step, v.mse]));
    mseMax = Math.ceil(Math.max(...msePoints.values()) * 100) / 100;
    const tail = values.filter((v) => v.step >= 48);
    mseTailMax = Math.ceil(Math.max(...tail.map((v) => v.mse)) * 1000) / 1000;
    const charts = [
      {
        id: "mse",
        values,
        x: (s) => stepX(s),
        y: (v) => mseY(v),
        max: mseMax,
        ticks: [0, 10, 20, 30, 40, 50],
        digits: 2,
      },
      {
        id: "mse-tail",
        values: tail,
        x: (s) => 40 + ((s - 48) / 2) * 140,
        y: (v) => 120 - (v / mseTailMax) * 80,
        max: mseTailMax,
        ticks: [48, 49, 50],
        digits: 3,
      },
    ];
    for (const chart of charts) {
      const { id, x, y, ticks, max, digits } = chart;
      const grid = $(`#${id}-grid`);
      for (const value of [0, max / 2, max]) {
        grid.append(
          node("line", {
            x1: x(ticks[0]),
            x2: x(ticks.at(-1)),
            y1: y(value),
            y2: y(value),
            stroke: value === 0 ? "#9ba7b5" : "#e5e9ee",
            "stroke-dasharray": value === 0 ? "3 4" : "none",
          }),
          node(
            "text",
            {
              x: x(ticks[0]) - 8,
              y: y(value) + 3,
              "text-anchor": "end",
              class: "svg-small",
            },
            value === 0 ? "0" : value.toFixed(digits),
          ),
        );
      }
      for (const step of ticks)
        grid.append(
          node(
            "text",
            {
              x: x(step),
              y: y(0) + 19,
              "text-anchor": "middle",
              class: "svg-small",
            },
            step,
          ),
        );
      attr($(`#${id}-curve`), {
        d: chart.values
          .map((v, i) => `${i ? "L" : "M"}${x(v.step)},${y(v.mse)}`)
          .join(" "),
      });
    }
    const endpoint = msePoints.get(50),
      y = 120 - (endpoint / mseTailMax) * 80;
    attr($("#mse-endpoint"), { cx: 180, cy: y });
    attr($("#mse-endpoint-halo"), { cx: 180, cy: y });
    attr($("#mse-endpoint-value"), { y: y - 16 });
    $("#mse-endpoint-value").textContent = `${endpoint.toFixed(5)} ≠ 0`;
    attr($("#residual-arrow"), { d: `M246 110 C235 140 205 138 187 ${y + 7}` });
    attr($("#residual-arrowhead"), {
      d: `M184 ${y + 15} L187 ${y + 7} L195 ${y + 11}`,
    });
  }
  function renderMSE(step) {
    const value = msePoints.get(step);
    if (value === undefined) return;
    const x = stepX(step);
    $("#frequency-mse-value").textContent = value.toFixed(5);
    attr($("#mse-current"), {
      cx: x,
      cy: mseY(value),
      opacity: 1,
    });
    attr($("#mse-cursor"), { x1: x, x2: x });
    attr($("#mse-clip"), { width: x - 42 + 1 });
    const tailX = 40 + clamp((step - 48) / 2) * 140;
    attr($("#mse-tail-clip"), { width: tailX - 40 + 1 });
    attr($("#mse-tail-current"), {
      cx: tailX,
      cy: 120 - (value / mseTailMax) * 80,
    });
    $("#frequency-residual").classList.toggle("is-tail", step >= 48);
    $("#frequency-residual").classList.toggle("at-endpoint", step === 50);
    $("#frequency-residual").setAttribute("aria-hidden", String(step < 48));
  }
  function positionFrequencyLink() {
    const layout = $(".frequency-layout").getBoundingClientRect();
    if (!layout.width || !layout.height) return;
    const anchor = $("#frequency-region-anchor").getBoundingClientRect();
    const image = $(".frequency-error-frame").getBoundingClientRect();
    const stacked = anchor.left < image.right;
    const box = $("#frequency-region-box").getBoundingClientRect();
    const sx = box.right - layout.left;
    const sy = box.top + box.height * 0.22 - layout.top;
    const ex = anchor.left + anchor.width / 2 - layout.left - 6;
    const ey = anchor.top + anchor.height / 2 - layout.top;
    const outerX = image.right - layout.left + 10;
    const underY = image.bottom - layout.top + 32;
    const d = stacked
      ? `M${sx} ${sy} C${outerX} ${sy} ${outerX} ${underY - 28} ${outerX} ${underY - 12}
         Q${outerX} ${underY} ${outerX - 18} ${underY}
         C${outerX - 80} ${underY + 16} ${ex - 30} ${ey - 26} ${ex} ${ey}`
      : `M${sx} ${sy} C${sx + 28} ${sy} ${ex - 30} ${ey} ${ex} ${ey}`;
    attr($("#frequency-region-link"), {
      viewBox: `0 0 ${layout.width} ${layout.height}`,
    });
    attr($("#frequency-region-connector"), { d });
  }
  const frequencyLayoutObserver = new ResizeObserver(positionFrequencyLink);
  [
    ".frequency-layout",
    ".frequency-error-frame",
    ".frequency-mse-chart",
  ].forEach((selector) => frequencyLayoutObserver.observe($(selector)));
  document.fonts.ready.then(positionFrequencyLink);

  let latestSnapshot = -1,
    displayedSnapshot = -1;
  function prepareFrequencyFrame(frame) {
    return Promise.all(
      [frame.state, frame.error].map((src) =>
        loadImage(src, frequencyManual === null ? 1 : 0, frequencyOwner),
      ),
    );
  }
  function frequencyCursor(progress) {
    const x = stepX(progress);
    attr($("#frequency-cursor"), { x1: x, x2: x });
    attr($("#frequency-clip"), { width: x - 42 });
  }
  function renderFrequency(progress) {
    if (!frequencyFrames.length || !frequencyNear) return;
    const frame = frequencyFrames.reduce((a, b) =>
      Math.abs(b.step - progress) < Math.abs(a.step - progress) ? b : a,
    );
    const snap = frame.step;
    if (snap === latestSnapshot) {
      if (displayedSnapshot === snap) frequencyCursor(progress);
      return;
    }
    latestSnapshot = snap;
    frequencyReady = false;
    imageQueue.cancel(frequencyOwner);
    prepareFrequencyFrame(frame)
      .then(([state, error]) => {
        if (latestSnapshot !== snap) return;
        displayedSnapshot = snap;
        $("#frequency-state").dataset.src = frame.state;
        $("#frequency-error").dataset.src = frame.error;
        $("#frequency-state").src = state.src;
        $("#frequency-error").src = error.src;
        frequencyReady = true;
        warmFrequency();
        $("#frequency-progress").value = snap;
        $("#frequency-progress-label").textContent = snap;
        $("#frequency-state").alt =
          `Actual decoded state after ${snap} denoising updates`;
        $("#frequency-step").textContent = snap;
        $("#frequency-state-note").textContent =
          `Actual decoded state · step ${snap}`;
        $("#frequency-error-note").textContent = "Full-image MAE · scale 0–0.5";
        $("#frequency-error").alt =
          `Full-image RGB MAE heatmap after ${snap} denoising updates; the square highlights an illustrative non-edit area above the flowerpot`;
        renderMSE(snap);
        frequencyCursor(progress);
      })
      .catch(assetFailure);
  }
  $("#frequency-progress").addEventListener("input", (e) => {
    frequencyManual = Number(e.target.value);
    carouselPaused = true;
    carouselButton();
    renderFrequency(frequencyManual);
  });

  // Preserve the selected source pixels exactly. Both masks are complementary.
  let editType = "local",
    compositeReady = false;
  const curationFlow = $("#curation-flow");
  const localCuration = {
    source: $("#curation-source").getAttribute("src"),
    generated: $("#curation-generated").getAttribute("src"),
    sourceAlt: $("#curation-source").alt,
    generatedAlt: $("#curation-generated").alt,
    prompt: $("#curation-prompt").textContent.trim(),
    aspectRatio:
      Number($("#curation-source").getAttribute("width")) /
        Number($("#curation-source").getAttribute("height")) || 1,
  };
  let curationGeometry = null,
    curationTime = 0,
    curationInspect = null,
    curationHover = null;
  const curationReferences = $$(".curation-reference");
  function inspectCuration() {
    const focused = document.activeElement.closest?.(".curation-reference");
    curationInspect = curationHover || focused?.dataset.curationPart || null;
    curationFlow.dataset.inspect = curationInspect || "";
    curationReferences.forEach((reference) =>
      reference.classList.toggle(
        "is-matched",
        reference.dataset.curationPart === curationInspect,
      ),
    );
    $$("[data-curation-image]").forEach((card) =>
      card.classList.toggle(
        "is-matched",
        card.dataset.curationImage === curationInspect,
      ),
    );
  }
  function clearCurationInspect() {
    curationHover = null;
    if (document.activeElement.matches(".curation-reference"))
      document.activeElement.blur();
    inspectCuration();
  }
  curationReferences.forEach((reference) => {
    reference.addEventListener("pointerenter", (event) => {
      if (event.pointerType === "touch" || reference.disabled) return;
      curationHover = reference.dataset.curationPart;
      inspectCuration();
    });
    reference.addEventListener("pointerleave", () => {
      curationHover = null;
      inspectCuration();
    });
    reference.addEventListener("focus", inspectCuration);
    reference.addEventListener("blur", () => queueMicrotask(inspectCuration));
    reference.addEventListener("click", () =>
      reference.focus({ preventScroll: true }),
    );
  });
  document.addEventListener("pointerdown", (event) => {
    if (curationInspect && !event.target.closest(".curation-reference"))
      clearCurationInspect();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && curationInspect) clearCurationInspect();
  });
  const canvasIds = [
    "source-mask-canvas",
    "edit-mask-canvas",
    "target-source-canvas",
    "target-edit-canvas",
  ];
  async function prepareComposite() {
    const [src, generated, mask] = await Promise.all(
      [
        localCuration.source,
        localCuration.generated,
        "assets/curation-keep_mask.webp?v=a26f0e6bb8e4",
      ].map((src) => loadImage(src)),
    );
    if (
      src.naturalWidth !== generated.naturalWidth ||
      src.naturalHeight !== generated.naturalHeight
    )
      throw new Error("Endpoint composition requires aligned source and result images");
    localCuration.aspectRatio = src.naturalWidth / src.naturalHeight;
    if (editType === "local")
      curationFlow.style.setProperty(
        "--curation-aspect-ratio", localCuration.aspectRatio,
      );
    const scale = Math.min(
        1, 768 / Math.max(src.naturalWidth, src.naturalHeight),
      ),
      width = Math.round(src.naturalWidth * scale),
      height = Math.round(src.naturalHeight * scale),
      c = document.createElement("canvas");
    c.width = width;
    c.height = height;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(src, 0, 0, width, height);
    const a = ctx.getImageData(0, 0, width, height);
    ctx.drawImage(generated, 0, 0, width, height);
    const b = ctx.getImageData(0, 0, width, height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(mask, 0, 0, width, height);
    const m = ctx.getImageData(0, 0, width, height);
    for (let i = 0; i < a.data.length; i += 4) {
      a.data[i + 3] = m.data[i];
      b.data[i + 3] = 255 - m.data[i];
    }
    canvasIds.forEach((id, i) => {
      const target = $("#" + id);
      target.width = width;
      target.height = height;
      target.getContext("2d").putImageData(i % 2 === 0 ? a : b, 0, 0);
    });
    curationGeometry = null;
    compositeReady = true;
    curationFlow.dataset.ready = "true";
    if (scene === 2) renderCuration(paused || carouselPaused ? 15 : sceneTime);
  }
  let compositePreparing = false;
  new IntersectionObserver(
    ([entry]) => {
      frequencyNear = entry.isIntersecting;
      if (frequencyNear) {
        if (!compositePreparing) {
          compositePreparing = true;
          prepareComposite().catch(assetFailure);
        }
        renderFrequency(frequencyManual ?? (paused || carouselPaused ? 50 : 0));
        warmFrequency();
      } else {
        imageQueue.cancel(frequencyIdleOwner);
        imageQueue.cancel(frequencyOwner);
        frequencyWarming = false;
        frequencyReady = false;
        latestSnapshot = -1;
      }
    },
    { rootMargin: "150px 0px" },
  ).observe($("#motivation"));
  function changeEditType(type) {
    clearCurationInspect();
    editType = type;
    curationFlow.dataset.editType = type;
    sceneTime = 0;
    const identity = type === "identity";
    curationFlow.style.setProperty(
      "--curation-aspect-ratio", identity ? 1 : localCuration.aspectRatio,
    );
    curationGeometry = null;
    $("#local-edit-button").classList.toggle("selected", !identity);
    $("#local-edit-button").setAttribute("aria-pressed", String(!identity));
    $("#identity-edit-button").classList.toggle("selected", identity);
    $("#identity-edit-button").setAttribute("aria-pressed", String(identity));
    $("#curation-source").src =
      identity ? "assets/identity-source.webp" : localCuration.source;
    $("#curation-source").alt = identity
      ? "Source image for an identity edit"
      : localCuration.sourceAlt;
    $("#curation-generated").src =
      identity ? "assets/identity-generated.webp" : localCuration.generated;
    $("#curation-generated").alt = identity
      ? "Frozen base model output for the instruction to keep the image unchanged"
      : localCuration.generatedAlt;
    $("#curation-prompt").textContent = identity
      ? "Keep the image exactly unchanged."
      : localCuration.prompt;
    $("#identity-target").hidden = !identity;
    canvasIds.forEach((id) => ($("#" + id).hidden = identity));
    $("#curation-target").setAttribute(
      "aria-label",
      identity
        ? "Identity endpoint: the entire source image, unchanged."
        : "Preservation-aware endpoint: source pixels outside the edit region, generated pixels inside it.",
    );
    $("#equation-source-mask").hidden = identity;
    $("#equation-plus").hidden = identity;
    $("#equation-edit-term").hidden = identity;
    $(".curation-symbol[data-curation-part='edit']").disabled = identity;
    $("#equation-source-meaning").textContent = identity
      ? "Entire source image"
      : "Preserved source pixels";
    $(".equation-term[data-curation-part='source']").setAttribute(
      "aria-label",
      identity
        ? "Source term: the entire source image"
        : "Source term: preserve the source pixels selected by M",
    );
    $("#curation-equation-note").innerHTML = identity
      ? "M = 1 everywhere. <br>The endpoint is the source."
      : "M selects what stays. <br>1 − M selects the edit.";
    $("#source-label").innerHTML = identity
      ? '<i class="legend-dot green"></i><span class="mask-symbol">M = 1</span> · Entire source'
      : '<i class="legend-dot green"></i><span class="mask-symbol">M</span> · Content to preserve';
    $("#base-label").innerHTML = identity
      ? "Not used in target"
      : '<i class="legend-dot blue"></i><span class="mask-symbol">1 − M</span> · The model’s own edit';
    $("#target-label").innerHTML = identity
      ? '<i class="legend-dot green"></i>Exactly the source image'
      : '<i class="legend-dot green"></i>Preserve <span>+</span> <i class="legend-dot blue"></i>Edit';
    renderCuration(paused || carouselPaused ? 15 : 0);
  }
  $("#local-edit-button").addEventListener("click", () =>
    changeEditType("local"),
  );
  $("#identity-edit-button").addEventListener("click", () =>
    changeEditType("identity"),
  );
  $("#curation-replay").addEventListener("click", () => {
    clearCurationInspect();
    sceneTime = 0;
    paused = carouselPaused = false;
    updateMotionState();
    carouselButton();
    trainingButton();
    renderCuration(0);
  });

  function measureCuration() {
    const origin = curationFlow.getBoundingClientRect();
    const boxes = [
      "#curation-source",
      "#curation-generated",
      "#curation-target",
    ].map((selector) => {
      const box = $(selector).getBoundingClientRect();
      return {
        x: box.left - origin.left,
        y: box.top - origin.top,
        width: box.width,
        height: box.height,
      };
    });
    if (boxes.some((box) => !box.width || !box.height)) return null;
    for (const piece of $$(".curation-pieces > *")) {
      piece.style.width = `${boxes[2].width}px`;
      piece.style.height = `${boxes[2].height}px`;
    }
    return boxes;
  }

  // Move the actual complementary cutouts from their donor images into one frame.
  // Geometry is measured on scene entry / resize, never on each animation frame.
  function renderCuration(t) {
    curationTime = t;
    if (!curationGeometry) curationGeometry = measureCuration();
    if (!curationGeometry) return;
    const identity = editType === "identity";
    const highlight = compositeReady ? smooth((t - 0.9) / 0.7) : 0;
    const settled = smooth((t - (identity ? 4.4 : 6.7)) / 0.7);
    const [source, generated, target] = curationGeometry;
    const place = (selector, from, start, duration) => {
      const progress = smooth((t - start) / duration);
      const lift = Math.sin(Math.PI * progress);
      const x = from.x + (target.x - from.x) * progress;
      const y =
        from.y +
        (target.y - from.y) * progress -
        lift * Math.min(32, from.width * 0.15);
      const sx =
        (from.width + (target.width - from.width) * progress) / target.width;
      const sy =
        (from.height + (target.height - from.height) * progress) /
        target.height;
      const piece = $(selector);
      piece.style.opacity = compositeReady && t >= start ? "1" : "0";
      piece.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${sx}, ${sy})`;
      piece.style.filter =
        lift > 0.01
          ? `drop-shadow(0 ${lift * 7}px ${lift * 8}px #26395030)`
          : "none";
    };
    place(
      identity ? "#identity-target" : "#target-source-canvas",
      source,
      2.2,
      2.2,
    );
    if (!identity) place("#target-edit-canvas", generated, 4.9, 1.8);
    $("#source-mask-canvas").style.opacity =
      highlight * (1 - smooth((t - 2.2) / 0.18));
    $("#edit-mask-canvas").style.opacity =
      highlight * smooth((t - 4.2) / 0.5) * (1 - smooth((t - 4.9) / 0.18));
    $("#curation-source").style.opacity =
      1 -
      0.78 * (1 - settled) * (identity ? smooth((t - 2.2) / 0.2) : highlight);
    $("#curation-generated").style.opacity =
      1 - 0.78 * (1 - settled) * highlight;
    curationFlow.classList.toggle(
      "composing",
      compositeReady && t >= (identity ? 4.4 : 6.7),
    );
    curationFlow.dataset.phase =
      t < 0.9
        ? "rollout"
        : t < 2.2
          ? "select"
          : t < 4.4
            ? "preserve"
            : !identity && t < 6.7
              ? "edit"
              : "complete";
    curationFlow.style.setProperty("--rollout-progress", clamp(t / 0.9));
    curationFlow.classList.toggle("rolling", t < 0.9);
  }
  new ResizeObserver(() => {
    curationGeometry = null;
    if (scene === 2) renderCuration(curationTime);
  }).observe(curationFlow);

  // Terminal-only experiment: images are taken directly from paper Figure 2.
  const terminalRows = $$(".terminal-row");
  fetch("assets/terminal-sequence.json", { cache: "no-cache" })
    .then((response) => {
      if (!response.ok) throw new Error("Unable to load Figure 2 results");
      return response.json();
    })
    .then(({ rows }) => {
      for (const { method, frames } of rows) {
        const row = terminalRows.find((el) => el.dataset.terminal === method);
        for (const { turn, src } of frames) {
          const cell = document.createElement("td");
          cell.dataset.turn = `Turn ${turn}`;
          const img = document.createElement("img");
          img.loading = "lazy";
          img.src = src;
          img.width = 347;
          img.height = 217;
          img.alt = `${$("strong", row).textContent}, ${$(".terminal-schedule", row).textContent}, after ${turn} edits`;
          cell.append(img);
          row.append(cell);
        }
      }
      $("#terminal-grid").dataset.ready = "true";
    })
    .catch((error) => console.error(error));

  // Schematic extension of the toy intervention, not a measured trajectory.
  const terminalX = (p) => 30 + 600 * p;
  const terminalBaseY = (p) => 45 + 18 * Math.sin(Math.PI * p) - 20 * p ** 3;
  const terminalPath = (y) =>
    Array.from({ length: 101 }, (_, i) => {
      const p = i / 100;
      return `${i ? "L" : "M"}${terminalX(p).toFixed(2)},${y(p).toFixed(2)}`;
    }).join(" ");
  attr($("#terminal-base-path"), { d: terminalPath(terminalBaseY) });
  const terminalTailSteps = $$("#terminal-tail-steps circle");
  function renderTerminal(t) {
    const progress = smooth(t / 2.6);
    attr($("#terminal-final-update"), {
      "stroke-dasharray": "1",
      "stroke-dashoffset": 1 - progress,
    });
    attr($("#terminal-train-dot"), {
      cx: 380 + 140 * progress,
      cy: 56 + 29 * progress,
      opacity: t < 2.6 ? 1 : 0,
    });
    const row = t < 3 || t >= 15 ? -1 : Math.floor((t - 3) / 4);
    terminalRows.forEach((el, i) =>
      el.classList.toggle("is-active", i === row),
    );
    $(".terminal-cause").classList.toggle("is-active", row === 2);

    const expansion = smooth((t - 15) / 3);
    const start = 0.98 - 0.33 * expansion;
    const y = (p) => terminalBaseY(p) + 50 * smooth((p - start) / (1 - start));
    attr($("#terminal-refined-path"), { d: terminalPath(y) });
    attr($("#terminal-prior-region"), { width: 600 * start });
    attr($("#terminal-tail-region"), {
      x: terminalX(start),
      width: 600 * (1 - start),
    });
    attr($("#terminal-onset"), {
      transform: `translate(${terminalX(start)} ${terminalBaseY(start)})`,
    });
    attr($("#terminal-onset-label"), { x: terminalX(start) });
    terminalTailSteps.forEach((el, i) => {
      const p = start + ((1 - start) * (i + 1)) / 5;
      attr(el, { cx: terminalX(p), cy: y(p), opacity: expansion });
    });
    const p = smooth((t - 18) / 3);
    attr($("#terminal-model-dot"), {
      cx: terminalX(p),
      cy: y(p),
      opacity: t >= 18 ? 1 : 0,
    });
    $("#scene-terminal").dataset.phase =
      t < 3 ? "supervise" : t < 15 ? ["base", "switch", "all"][row] : "extend";
  }
  $("#terminal-target-link").addEventListener("click", () =>
    selectScene(2, true),
  );
  $("#terminal-replay").addEventListener("click", () => {
    sceneTime = 0;
    paused = carouselPaused = false;
    updateMotionState();
    carouselButton();
    trainingButton();
    renderTerminal(0);
  });

  // One persistent diagram. Only the active teaching layer changes between phases.
  const phaseDurations = [3, 3.5, 5.5, 4, 7, 7, 6.5, 6];
  const phaseStarts = phaseDurations.map((_, i) =>
    phaseDurations.slice(0, i).reduce((sum, value) => sum + value, 0),
  );
  const totalTime = phaseDurations.reduce((sum, value) => sum + value, 0);
  let trainingTime = 0,
    trainingPhase = -1,
    trainingPaused = false,
    trainingDone = false;
  let trainingInView = false;
  new IntersectionObserver(
    ([entry]) => {
      trainingInView = entry.isIntersecting && entry.intersectionRatio >= 0.1;
    },
    { threshold: [0, 0.1] },
  ).observe($(".training-canvas"));
  let onset = 0.6,
    onsetLocked = 0.6,
    priorSample = 0.34,
    tailSample = 0.84;
  const X = (t) => 75 + 870 * t;
  const baseY = (t) => 210 + 29 * Math.sin(t * Math.PI) - 110 * t ** 3;
  const baseDY = (t) => 29 * Math.PI * Math.cos(t * Math.PI) - 330 * t ** 2;
  const terminalTime = 0.98;
  const targetY = 340;
  const blendWeight = (t) => clamp((terminalTime - t) / (terminalTime - onset));
  // Illustrative state-dependent teacher field. This is not experimental latent data.
  const baseVelocityY = (t, y) => baseDY(t) + 0.4 * (baseY(t) - y);
  const targetVelocityY = (t, y) => {
    const lambda = blendWeight(t);
    return (
      lambda * baseVelocityY(t, y) + ((1 - lambda) * (targetY - y)) / (1 - t)
    );
  };
  let guidedCurve = [],
    curveOnset = -1;
  function prepareGuidedCurve() {
    if (curveOnset === onset) return;
    curveOnset = onset;
    const h = (terminalTime - onset) / 256;
    let y = baseY(onset);
    guidedCurve = [{ y, dy: baseDY(onset) }];
    // Integrate the very same blended field drawn as v*. Its arrow is therefore
    // tangent to the guided curve, rather than an independently chosen direction.
    for (let i = 0; i < 256; i++) {
      const t = onset + i * h;
      const a = targetVelocityY(t, y);
      const b = targetVelocityY(t + h / 2, y + (a * h) / 2);
      const c = targetVelocityY(t + h / 2, y + (b * h) / 2);
      const d = targetVelocityY(t + h, y + c * h);
      y += (h * (a + 2 * b + 2 * c + d)) / 6;
      guidedCurve.push({ y, dy: targetVelocityY(t + h, y) });
    }
  }
  function steerY(t) {
    if (t <= onset) return baseY(t);
    if (t >= terminalTime)
      return (
        targetY +
        ((guidedCurve.at(-1).y - targetY) * (1 - t)) / (1 - terminalTime)
      );
    const position = ((t - onset) / (terminalTime - onset)) * 256;
    const i = Math.floor(position),
      u = position - i;
    const a = guidedCurve[i],
      b = guidedCurve[i + 1];
    const h = (terminalTime - onset) / 256;
    return (
      (2 * u ** 3 - 3 * u ** 2 + 1) * a.y +
      (u ** 3 - 2 * u ** 2 + u) * h * a.dy +
      (-2 * u ** 3 + 3 * u ** 2) * b.y +
      (u ** 3 - u ** 2) * h * b.dy
    );
  }
  const point = (t) => ({ x: X(t), y: steerY(t) });
  function path(from, to, type = "steer", morph = 1) {
    return Array.from({ length: 201 }, (_, i) => {
      const t = from + ((to - from) * i) / 200;
      const y =
        type === "base" ? baseY(t) : baseY(t) + morph * (steerY(t) - baseY(t));
      return `${i ? "L" : "M"}${X(t).toFixed(3)},${y.toFixed(3)}`;
    }).join(" ");
  }
  const phases = [
    [
      "01 / THE BASE TRAJECTORY",
      "Start from the learned prior.",
      "Initialize from the frozen base model.",
      "x₀ → x₁<sup>base</sup>",
      "The original denoising trajectory.",
    ],
    [
      "01 / THE PRESERVATION TARGET",
      "Keep the edit. Restore the rest.",
      "The endpoint constructed in Motivation becomes the training target.",
      "x₁<sup>tgt</sup> = M ⊙ x<sup>src</sup> + (1 − M) ⊙ x₁<sup>base</sup>",
      "Same early dynamics; a corrected tail.",
    ],
    [
      "02 / CHOOSE THE ONSET",
      "Choose where steering begins.",
      "The shared beginning stays fixed as the tail changes.",
      "k ≤ kₛ &nbsp; | &nbsp; k > kₛ",
      "Prior region → transition region.",
    ],
    [
      "03 / CACHE THE BASE STATES",
      "Keep anchors before the tail.",
      "Cache every ≤3 updates, including the onset. No restarts inside the tail.",
      "x<sub>anchor</sub> → x<sub>k</sub>",
      "Prior: nearest preceding cache. Tail: fixed onset.",
    ],
    [
      "04 / LEARN THE PRIOR",
      "Roll out. Match the base velocity.",
      "Start at the nearest preceding cache; train at the reached state.",
      "v* = v<sup>base</sup>(x<sub>k</sub>, k, c)",
      "Fit vθ to v*. Only this prediction receives gradients.",
    ],
    [
      "05 / FOLLOW YOUR OWN TAIL",
      "Always restart from the onset.",
      "Self-roll out to a sampled tail step. Recompute both velocities there.",
      "v<sup>pres</sup> = (x₁<sup>tgt</sup> − x<sub>k</sub>) / (1 − k)",
      "Both velocities use xₖ reached by the current model.",
    ],
    [
      "06 / BLEND THE SUPERVISION",
      "Shift from prior to preservation.",
      "Later in the tail, less base velocity and more endpoint correction.",
      "v* = λv<sup>base</sup> + (1 − λ)v<sup>pres</sup>",
      "MSE(vθ, stopgrad(v*)) · λ decreases from 1 to 0.",
    ],
    [
      "07 / INFERENCE",
      "One model. The whole trajectory.",
      "The learned correction is now part of the editor.",
      "x₀ → TailSteer → x₁",
      "Teacher, targets and blending are used only for training.",
    ],
  ];
  function trainingButton() {
    const stopped = paused || trainingPaused || trainingDone;
    $("#training-play").textContent = stopped ? "▷" : "Ⅱ";
    $("#training-play").setAttribute(
      "aria-label",
      stopped ? "Play training animation" : "Pause training animation",
    );
    $("#training-play").setAttribute("aria-pressed", String(stopped));
  }
  function focusTrainingDiagram(phase, animate = false) {
    if (phase < 0) return;
    const canvas = $(".training-canvas"),
      tabs = $(".training-steps");
    const behavior =
      animate && !paused && !reduced.matches ? "smooth" : "instant";
    if (canvas.scrollWidth > canvas.clientWidth + 1) {
      const focus = [
        0.15,
        0.8,
        onsetLocked,
        0.3,
        priorSample,
        tailSample,
        tailSample,
        0.65,
      ][phase];
      canvas.scrollTo({
        left:
          (X(focus) / 1100) * $("#training-svg").getBoundingClientRect().width -
          canvas.clientWidth * 0.55,
        behavior,
      });
    }
    const selected = $("button.selected", tabs);
    if (selected && tabs.scrollWidth > tabs.clientWidth + 1)
      tabs.scrollTo({
        left:
          selected.offsetLeft -
          tabs.offsetLeft -
          (tabs.clientWidth - selected.offsetWidth) / 2,
        behavior,
      });
  }
  new ResizeObserver(() => focusTrainingDiagram(trainingPhase)).observe(
    $(".training-canvas"),
  );
  function setTrainingPhase(p) {
    if (p === trainingPhase) return;
    trainingPhase = p;
    $("#training-svg").dataset.phase = p;
    const [label, title, copy, equation, explain] = phases[p];
    $("#training-phase-label").textContent = label;
    $("#training-phase-title").textContent = title;
    $("#training-phase-copy").textContent = copy;
    $("#training-equation").innerHTML =
      `<span class="equation-symbol">${equation}</span><span class="equation-explainer">${explain}</span>`;
    $("#vector-base-label").innerHTML =
      `${p === 6 ? "λ" : p === 4 ? "v* = " : ""}v<tspan baseline-shift="super" font-size="11">base</tspan>`;
    $("#vector-pres-label").innerHTML =
      `${p === 6 ? "(1 − λ)" : ""}v<tspan baseline-shift="super" font-size="11">pres</tspan>`;
    $$(".training-steps button").forEach((button) => {
      const selected =
        Number(button.dataset.trainingPhase) === (p === 1 ? 0 : p);
      button.classList.toggle("selected", selected);
      if (selected) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    focusTrainingDiagram(p, true);
  }
  function seekTraining(phase, hold = false) {
    trainingTime =
      phaseStarts[phase] + (hold ? phaseDurations[phase] * 0.82 : 0);
    trainingPaused = hold;
    trainingDone = false;
    trainingButton();
    renderTraining(trainingTime);
  }
  $$(".training-steps button").forEach((button) =>
    button.addEventListener("click", () =>
      seekTraining(Number(button.dataset.trainingPhase), paused),
    ),
  );
  $("#training-replay").addEventListener("click", () => {
    onset = onsetLocked = 0.6;
    priorSample = 0.34;
    tailSample = 0.84;
    paused = false;
    updateMotionState();
    carouselButton();
    seekTraining(0);
  });
  $("#training-play").addEventListener("click", () => {
    if (paused) {
      paused = false;
      updateMotionState();
      carouselButton();
      trainingPaused = false;
      if (trainingDone) {
        trainingTime = 0;
        trainingDone = false;
      }
    } else if (trainingDone) {
      trainingTime = 0;
      trainingDone = false;
      trainingPaused = false;
    } else trainingPaused = !trainingPaused;
    trainingButton();
  });
  $("#steering-onset").addEventListener("input", (event) => {
    onset = onsetLocked = Number(event.target.value) / 50;
    priorSample = Math.min(priorSample, onset - 0.08);
    tailSample = Math.round((onset + (terminalTime - onset) * 0.6) * 50) / 50;
    seekTraining(3, true);
  });
  $("#sample-another").addEventListener("click", () => {
    const prior = trainingPhase < 5;
    const onsetStep = Math.round(onsetLocked * 50);
    priorSample = (2 + Math.floor(Math.random() * (onsetStep - 3))) / 50;
    tailSample =
      (onsetStep + 1 + Math.floor(Math.random() * (49 - onsetStep))) / 50;
    seekTraining(prior ? 4 : 5, paused);
  });
  function setTrainingCaption(text) {
    const caption = $("#rollout-caption");
    if (caption.textContent === text) return;
    caption.replaceChildren(
      ...text
        .split(/(xₖ|vθ|v\*)/g)
        .map((part, i) =>
          i % 2 ? node("tspan", { class: "training-math" }, part) : part,
        ),
    );
  }
  function renderTraining(time) {
    let p = phaseDurations.findIndex(
      (duration, i) => time < phaseStarts[i] + duration,
    );
    if (p < 0) p = 7;
    const elapsed = time - phaseStarts[p];
    const local = clamp(elapsed / phaseDurations[p]);
    const enter = (phase, delay = 0, duration = 0.65) =>
      smooth((time - phaseStarts[phase] - delay) / duration);
    const exitPhase = (phase, duration = 0.6) =>
      1 -
      smooth(
        (time - phaseStarts[phase] - phaseDurations[phase] + duration) /
          duration,
      );
    const opacity = (id, value) => {
      $(id).style.opacity = String(value);
    };
    setTrainingPhase(p);
    if (p === 2 && !trainingPaused) {
      const u = clamp((local - 0.12) / 0.72);
      const amplitude = Math.min(0.12, onsetLocked - 0.4, 0.9 - onsetLocked);
      onset =
        onsetLocked +
        amplitude * Math.sin(u * Math.PI * 2) * Math.sin(u * Math.PI);
    } else onset = onsetLocked;
    prepareGuidedCurve();
    const inference = enter(7);
    const split = X(onset),
      onsetY = baseY(onset);
    $("#onset-value").textContent = `${Math.round(onset * 50)} / 50`;
    $("#steering-onset").value = Math.round(onset * 50);
    $("#training-progress-fill").style.transform =
      `scaleX(${clamp(time / totalTime)})`;
    const copyAlpha =
      paused || trainingPaused || trainingDone
        ? 1
        : smooth(elapsed / 0.25) * (p === 7 ? 1 : exitPhase(p, 0.25));
    $(".training-description").style.opacity = copyAlpha;

    attr($("#training-base"), {
      d: path(0, p === 0 ? smooth(local) : 1, "base"),
    });
    opacity("#training-base", 1 - 0.82 * inference);
    opacity("#original-endpoint", enter(0, 2.2, 0.8) * (1 - 0.75 * inference));
    attr($("#training-steered"), {
      d: path(0, 1, "steer", enter(1, 0.2, 2.3)),
    });
    opacity("#training-steered", enter(1));
    opacity("#target-endpoint", enter(1));
    opacity("#training-regions", enter(2, 0.25, 1) * (1 - 0.55 * inference));
    opacity("#training-onset", enter(2) * (1 - 0.75 * inference));
    attr($("#prior-region"), { width: split - 30 });
    attr($("#transition-region"), { x: split, width: 1065 - split });
    attr($("#transition-region-label"), { x: split + 25 });
    attr($("#onset-line"), { x1: split, x2: split });
    attr($("#onset-triangle"), {
      d: `M${split} ${onsetY - 12}L${split + 11} ${onsetY + 8}L${split - 11} ${onsetY + 8}Z`,
    });
    attr($("#onset-label"), { x: split, y: onsetY - 34 });
    if (p >= 3 && $("#training-anchors").dataset.onset !== String(onset)) {
      $("#training-anchors").replaceChildren();
      const anchors = [0];
      for (let t = 0.06; t < onset - 0.001; t += 0.06) anchors.push(t);
      // The triangle already marks the cached onset; no overlapping green ring.
      anchors.forEach((t) =>
        $("#training-anchors").append(
          node("circle", {
            cx: X(t),
            cy: baseY(t),
            r: 5,
            fill: "#668f46",
            stroke: "#c6dfb4",
            "stroke-width": 3,
            "data-step": Math.round(t * 50),
          }),
        ),
      );
      $("#training-anchors").dataset.onset = String(onset);
    }
    const anchors = $$("#training-anchors circle");
    anchors.forEach((anchor, i) => {
      anchor.style.opacity =
        enter(3, 0.2 + (i / Math.max(1, anchors.length - 1)) * 2.1, 0.45) *
        (1 - inference);
    });
    const sampling = p >= 4 && p <= 6;
    let sampleAlpha =
      p === 4
        ? enter(4) * exitPhase(4)
        : p === 5
          ? enter(5)
          : p === 6
            ? exitPhase(6)
            : 0;
    opacity("#sample-point", sampleAlpha);
    opacity("#anchor-focus", sampleAlpha);
    opacity("#anchor-halo", p === 4 ? 1 : 0);
    opacity("#anchor-ring", p === 4 ? 1 : 0);
    opacity("#training-vectors", 0);
    opacity(
      "#training-rollout",
      sampling ? sampleAlpha : p === 7 ? inference : 0,
    );
    opacity(
      "#rollout-particle",
      sampling ? sampleAlpha : p === 7 ? inference : 0,
    );
    opacity("#rollout-caption", sampleAlpha);
    opacity("#velocity-mix", p === 6 ? enter(6, 0.3) * sampleAlpha : 0);
    if (sampling) {
      const sample = p === 4 ? Math.min(priorSample, onset - 0.04) : tailSample;
      const start =
        p === 4
          ? Math.max(0, (Math.ceil(sample / 0.06 - 1e-8) - 1) * 0.06)
          : onset;
      const progress =
        p === 6 ? 1 : smooth((elapsed - 0.8) / (p === 4 ? 1.8 : 2.6));
      const q = point(sample),
        r = point(start + (sample - start) * progress);
      $("#training-svg").dataset.sample = sample;
      $("#training-svg").dataset.anchor = start;
      attr($("#training-rollout"), {
        d: path(start, start + (sample - start) * progress),
      });
      attr($("#rollout-particle"), { cx: r.x, cy: r.y });
      attr($("#anchor-halo"), { cx: X(start), cy: baseY(start) });
      attr($("#anchor-ring"), { cx: X(start), cy: baseY(start) });
      attr($("#anchor-label"), { x: X(start), y: baseY(start) + 38 });
      $("#anchor-label").textContent =
        p === 4 ? `Cache ${Math.round(start * 50)}` : "Fixed onset";
      attr($("#sample-halo"), { x: q.x - 18, y: q.y - 18 });
      attr($("#sample-square"), { x: q.x - 8, y: q.y - 8 });
      attr($("#sample-label"), { x: q.x, y: q.y - 48 });
      $("#sample-step").textContent = `k = ${Math.round(sample * 50)}`;
      setTrainingCaption(
        p === 4
          ? progress < 1
            ? "Nearest cache → self-rollout → xₖ"
            : "Match the base velocity at xₖ · gradients here only"
          : p === 5
            ? progress < 1
              ? "Fixed onset → self-rollout → xₖ"
              : "At this reached state: recompute both velocities"
            : "Fit vθ to v* · gradients here only",
      );
      const vectorAlpha =
        p === 6 ? 1 : smooth((elapsed - (p === 4 ? 2.6 : 3.4)) / 0.8);
      opacity("#training-vectors", vectorAlpha * sampleAlpha);
      const lambda = p === 4 ? 1 : blendWeight(sample);
      const base = { x: 870, y: baseVelocityY(sample, q.y) };
      const pres = { x: 870, y: (targetY - q.y) / (1 - sample) };
      const blended = { x: 870, y: lambda * base.y + (1 - lambda) * pres.y };
      const scale = Math.min(
        93 / Math.max(Math.hypot(base.x, base.y), Math.hypot(pres.x, pres.y)),
        (1030 - q.x) / 870,
      );
      const componentMix = p === 6 ? enter(6, 0.2, 1) : 0;
      const baseScale = scale * (1 - componentMix * (1 - lambda));
      const presScale = scale * (1 - componentMix * lambda);
      const targetProgress = p === 6 ? enter(6, 1, 1) : 0;
      const fit =
        p === 4 ? smooth((elapsed - 4) / 2.2) : p === 6 ? enter(6, 3.2, 2) : 0;
      const angle = 0.29 * (1 - fit),
        length = 0.86 + 0.14 * fit;
      const prediction = {
        x: (blended.x * Math.cos(angle) - blended.y * Math.sin(angle)) * length,
        y: (blended.x * Math.sin(angle) + blended.y * Math.cos(angle)) * length,
      };
      function vector(id, v, factor, dx = 8, dy = -9) {
        attr($("#vector-" + id), {
          x1: q.x,
          y1: q.y,
          x2: q.x + v.x * factor,
          y2: q.y + v.y * factor,
        });
        const right = q.x > 820;
        attr($("#vector-" + id + "-label"), {
          x: q.x + v.x * factor + (right ? -6 : dx),
          y: q.y + v.y * factor + dy,
          "text-anchor": right ? "end" : "start",
        });
      }
      vector("base", base, baseScale, 5, -13);
      vector("pres", pres, presScale, 6, 22);
      vector("target", blended, scale * targetProgress, 10, -4);
      vector("prediction", prediction, scale, 5, 27);
      opacity("#vector-base", baseScale > scale * 0.001 ? 1 : 0);
      opacity("#vector-base-label", baseScale > scale * 0.001 ? 1 : 0);
      opacity("#vector-pres", p >= 5 ? 1 : 0);
      opacity("#vector-pres-label", p >= 5 ? 1 : 0);
      opacity("#vector-target", targetProgress);
      opacity("#vector-target-label", smooth((targetProgress - 0.7) / 0.3));
      const predictionAlpha = p === 4 ? 1 : p === 6 ? enter(6, 2.4) : 0;
      opacity("#vector-prediction", predictionAlpha * (1 - 0.75 * fit));
      opacity("#vector-prediction-label", predictionAlpha * (1 - fit));
      const a = {
        x: q.x + prediction.x * scale,
        y: q.y + prediction.y * scale,
      };
      const b = { x: q.x + blended.x * scale, y: q.y + blended.y * scale };
      attr($("#prediction-loss"), {
        d: `M${a.x} ${a.y} Q${(a.x + b.x) / 2 + 13} ${(a.y + b.y) / 2} ${b.x} ${b.y}`,
      });
      opacity("#prediction-loss", predictionAlpha * (1 - fit));
      attr($("#preservation-guide"), {
        x1: q.x,
        y1: q.y,
        x2: 945,
        y2: targetY,
      });
      opacity("#preservation-guide", p >= 5 ? 0.28 : 0);
      attr($("#blend-base-guide"), {
        x1: q.x + lambda * base.x * scale,
        y1: q.y + lambda * base.y * scale,
        x2: b.x,
        y2: b.y,
      });
      attr($("#blend-pres-guide"), {
        x1: q.x + (1 - lambda) * pres.x * scale,
        y1: q.y + (1 - lambda) * pres.y * scale,
        x2: b.x,
        y2: b.y,
      });
      opacity("#blend-guides", p === 6 ? enter(6, 1.2) : 0);
      attr($("#velocity-mix-base"), { width: 170 * lambda });
      $("#velocity-mix-label").textContent =
        `λ = ${lambda.toFixed(2)} · 1 − λ = ${(1 - lambda).toFixed(2)}`;
    } else if (p === 7) {
      const progress = smooth((elapsed - 0.55) / 4.5),
        q = point(progress);
      attr($("#training-rollout"), { d: path(0, progress) });
      attr($("#rollout-particle"), { cx: q.x, cy: q.y });
    }
    $("#sample-another").textContent =
      p >= 5 ? "Sample a tail step ↗" : "Sample a prior step ↗";
  }

  function renderStatic() {
    animateHero(7);
    renderFrequency(50);
    renderScene(durations[scene] * 0.85);
    trainingTime = totalTime;
    trainingDone = true;
    trainingButton();
    renderTraining(totalTime);
  }
  updateChapter();
  renderTraining(0);
  renderScene(0);
  animateHero(0);
  if (paused) renderStatic();
  carouselButton();
  trainingButton();
  let previous = performance.now();
  function tick(now) {
    const dt = Math.min((now - previous) / 1000, 0.08);
    previous = now;
    if (!paused && visible) {
      if (activeChapter === "home") {
        heroTime += dt;
        animateHero(heroTime);
      }
      if (
        motivationInView &&
        !carouselPaused &&
        !curationInspect
      ) {
        if ((scene !== 1 || frequencyReady) && (scene !== 2 || compositeReady))
          sceneTime += dt;
        if (sceneTime > durations[scene]) selectScene(scene + 1);
        renderScene(sceneTime);
      }
      if (trainingInView && !trainingPaused && !trainingDone) {
        trainingTime += dt;
        if (trainingTime >= totalTime) {
          trainingTime = totalTime;
          trainingDone = true;
          trainingButton();
        }
        renderTraining(trainingTime);
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
