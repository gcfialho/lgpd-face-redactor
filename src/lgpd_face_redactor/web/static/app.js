// Polls a running job and reloads when it finishes; wires the "Blur everyone" button.
document.addEventListener("DOMContentLoaded", () => {
  const box = document.querySelector("[data-poll]");
  if (box) {
    const bar = box.querySelector("progress");
    const message = box.querySelector(".message");
    const busy = new Set(["queued", "analysing", "rendering"]);
    const tick = async () => {
      try {
        const response = await fetch(box.dataset.poll, { cache: "no-store" });
        if (!response.ok) return window.location.reload();
        const job = await response.json();
        bar.value = job.progress;
        message.textContent = job.message;
        if (!busy.has(job.status)) return window.location.reload();
      } catch (_) {
        /* transient error: try again */
      }
      setTimeout(tick, 1000);
    };
    setTimeout(tick, 1000);
  }

  document.querySelectorAll("[data-select='none']").forEach((button) =>
    button.addEventListener("click", () =>
      document.querySelectorAll("input[name='keep']").forEach((box) => (box.checked = false)),
    ),
  );
});
