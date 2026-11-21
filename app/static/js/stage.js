// The simulator stage: tracking and head streams, loaded only while their view is shown. The head stream loads only
// when capabilities.head_camera is available; otherwise its reason is shown inside its frame.
// Layout: the tracking frame keeps the stream's own aspect ratio and the square head frame sits beside it at the same
// height, because the two columns are shared in the ratio (frame width / frame height) to 1.
export function createStage({ root, trackingImg, trackingNote, headImg, headNote }) {
  function show(img, note, url, unavailableReason) {
    const want = url && !unavailableReason ? url : null;
    note.hidden = !unavailableReason;
    note.textContent = unavailableReason ? `Unavailable: ${unavailableReason}` : "";
    if (want && img.getAttribute("src") !== want) {
      img.onerror = () => { img.hidden = true; note.hidden = false; note.textContent = "The simulator stream could not be loaded."; };
      img.setAttribute("src", want);
    } else if (!want) {
      img.removeAttribute("src"); // closes the stream
    }
    img.hidden = !want;
  }
  return {
    update(visible, headCapability) {
      show(trackingImg, trackingNote, visible ? "/api/stream?camera=tracking" : null, null);
      const head = headCapability?.available ? null : headCapability?.reason || "The server does not report the head camera.";
      show(headImg, headNote, visible ? "/api/stream?camera=head" : null, visible ? head : null);
    },
    // /api/state frame {width, height}: the tracking stream's size.
    setAspect(width, height) {
      if (!(Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0)) return;
      root.style.gridTemplateColumns = `minmax(0, ${width / height}fr) minmax(0, 1fr)`;
      trackingImg.parentElement.style.aspectRatio = `${width} / ${height}`;
    },
  };
}
