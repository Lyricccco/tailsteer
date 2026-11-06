/* Shared image budget: interaction (0), visible (1), look-ahead (2), idle (3). */
(() => {
  "use strict";
  const jobs = new Map(),
    active = new Set(),
    cache = new Map();
  const warmed = new Set(),
    retired = new Set();
  let timer,
    lastForeground = performance.now(),
    pageLoaded = document.readyState === "complete";
  const abortError = () =>
    new DOMException("Image request superseded", "AbortError");
  const priority = (job) =>
    Math.min(...[...job.waiters.values()].map((w) => w.priority));
  const allowIdle = () =>
    pageLoaded &&
    !document.hidden &&
    !navigator.connection?.saveData &&
    !/^(slow-)?2g$/.test(navigator.connection?.effectiveType || "");

  function remember(url, image) {
    cache.set(url, image);
    while (cache.size > 72) {
      const oldest = cache.keys().next().value;
      retired.add(cache.get(oldest).src);
      cache.delete(oldest);
    }
    // Keep object URLs used by the live page, but release evicted, unused images.
    const live = new Set([...document.images].map((im) => im.src));
    for (const src of retired)
      if (!live.has(src)) {
        URL.revokeObjectURL(src);
        retired.delete(src);
      }
  }

  async function run(job) {
    active.add(job);
    job.controller = new AbortController();
    let objectURL;
    try {
      const response = await fetch(job.url, {
        signal: job.controller.signal,
        cache: "force-cache",
        priority: priority(job) <= 1 ? "high" : "low",
      });
      if (!response.ok)
        throw new Error(`Cannot load image (${response.status})`);
      const blob = await response.blob();
      warmed.add(job.url);
      let image;
      if ([...job.waiters.values()].some((w) => !w.warm)) {
        image = new Image();
        objectURL = URL.createObjectURL(blob);
        image.src = objectURL;
        await image.decode();
        if (job.controller.signal.aborted) throw abortError();
        remember(job.url, image);
        objectURL = null;
      }
      jobs.delete(job.url);
      for (const w of job.waiters.values()) w.resolve(image);
    } catch (error) {
      if (objectURL) URL.revokeObjectURL(objectURL);
      // Preemption keeps subscribers queued; cancellation removes them.
      if (error.name !== "AbortError" || !job.waiters.size) {
        if (jobs.get(job.url) === job) jobs.delete(job.url);
        for (const w of job.waiters.values()) w.reject(error);
      }
    } finally {
      active.delete(job);
      job.controller = null;
      if (priority(job) < 3) lastForeground = performance.now();
      pump();
    }
  }

  function pump() {
    clearTimeout(timer);
    const pending = [...jobs.values()]
      .filter((j) => !active.has(j) && j.waiters.size)
      .sort((a, b) => priority(a) - priority(b));
    const first = pending[0];
    if (!first) return;
    if (priority(first) < 3) {
      for (const job of active) {
        if (
          priority(job) > priority(first) &&
          (active.size >= 4 || priority(job) === 3)
        ) {
          job.controller?.abort();
          break;
        }
      }
    }
    for (const job of pending) {
      if (active.size >= 4) break;
      if (document.hidden && priority(job) >= 2) break;
      if (priority(job) === 3) {
        if (!allowIdle() || active.size) break;
        const delay = 800 - (performance.now() - lastForeground);
        if (delay > 0) {
          timer = setTimeout(pump, delay);
          break;
        }
      }
      run(job);
    }
  }

  function request(url, p = 1, owner = "page", warm = false) {
    if (!url) return Promise.resolve();
    if (warm && warmed.has(url)) return Promise.resolve();
    if (cache.has(url)) {
      const image = cache.get(url);
      cache.delete(url);
      cache.set(url, image);
      return Promise.resolve(image);
    }
    let job = jobs.get(url);
    if (!job) {
      job = { url, waiters: new Map() };
      jobs.set(url, job);
    }
    let waiter = job.waiters.get(owner);
    if (waiter) {
      waiter.priority = Math.min(p, waiter.priority);
      waiter.warm &&= warm;
    } else {
      waiter = { priority: p, warm };
      waiter.promise = new Promise((resolve, reject) =>
        Object.assign(waiter, { resolve, reject }),
      );
      job.waiters.set(owner, waiter);
    }
    if (p < 3) lastForeground = performance.now();
    queueMicrotask(pump);
    return waiter.promise;
  }

  function cancel(owner) {
    for (const job of jobs.values()) {
      const waiter = job.waiters.get(owner);
      if (!waiter) continue;
      job.waiters.delete(owner);
      waiter.reject(abortError());
      if (!job.waiters.size) {
        jobs.delete(job.url);
        job.controller?.abort();
      }
    }
    pump();
  }
  function assign(element, url, p = 1, owner = element) {
    element.dataset.src = url;
    return request(url, p, owner).then((image) => {
      if (element.isConnected && element.dataset.src === url)
        element.src = image.src;
      return image;
    });
  }
  window.TailSteerImages = {
    load: request,
    assign,
    cancel,
    warm: (url, owner) => request(url, 3, owner, true).catch(() => {}),
  };
  addEventListener(
    "load",
    () => {
      pageLoaded = true;
      pump();
    },
    { once: true },
  );
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      for (const job of active) if (priority(job) >= 2) job.controller?.abort();
    } else pump();
  });
})();
