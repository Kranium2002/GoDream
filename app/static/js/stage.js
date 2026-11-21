// The simulator stage: tracking and head streams, loaded only while their view is shown. The head stream loads only
// when capabilities.head_camera is available; otherwise its reason is shown.
export function createStage({ trackingImg, trackingNote, headImg, headNote }) {
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
  };
}
