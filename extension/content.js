(function () {
  const overlays = [];
  function getPageImageSrcs() {
    const imgs = Array.from(document.images || []);
    const visibleImgs = imgs.filter(img => {
      try {
        const rect = img.getBoundingClientRect();
        return rect.width >= 32 && rect.height >= 32 && img.src && img.complete;
      } catch (e) {
        return false;
      }
    });
    return visibleImgs.map(img => img.src);
  }

  function createBadgeForImg(img, text, color) {
    const rect = img.getBoundingClientRect();
    const badge = document.createElement('div');
    badge.className = 'deepfake-badge';
    badge.textContent = text;
    Object.assign(badge.style, {
      position: 'absolute',
      left: `${window.scrollX + rect.left + 6}px`,
      top: `${window.scrollY + rect.top + 6}px`,
      background: color,
      color: '#fff',
      padding: '4px 8px',
      borderRadius: '6px',
      fontSize: '12px',
      zIndex: 2147483647,
      pointerEvents: 'auto',
      boxShadow: '0 2px 6px rgba(0,0,0,0.3)'
    });
    document.body.appendChild(badge);
    overlays.push(badge);
    badge.addEventListener('click', () => {
      badge.remove();
    });
  }

  function overlayResults(results) {
    clearOverlays();

    results.forEach(r => {
      const src = r.src;
      const score = r.score;
      const label = score > 0.5 ? 'Fake' : 'Real';
      const percent = score > 0.5 ? (score * 100).toFixed(1) : ((1 - score) * 100).toFixed(1);
      const text = `${label} ${percent}%`;

      const matches = Array.from(document.images).filter(img => img.src === src);
      if (matches.length === 0) {
        const fuzzy = Array.from(document.images).find(img => img.src && img.src.includes(src));
        if (fuzzy) matches.push(fuzzy);
      }
      matches.forEach(img => {
        const color = score > 0.5 ? '#d9534f' : '#5cb85c';
        createBadgeForImg(img, text, color);
      });
    });
  }

  function clearOverlays() {
    while (overlays.length) {
      const el = overlays.pop();
      if (el && el.remove) el.remove();
    }
  }

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'GET_IMAGES') {
      const images = getPageImageSrcs();
      sendResponse({ images });
      return true;
    } else if (msg.type === 'HIGHLIGHT_RESULTS') {
      if (Array.isArray(msg.results)) {
        overlayResults(msg.results);
      }
      sendResponse({ ok: true });
      return true;
    } else if (msg.type === 'CLEAR_OVERLAYS') {
      clearOverlays();
      sendResponse({ ok: true });
      return true;
    }
  });
})();
