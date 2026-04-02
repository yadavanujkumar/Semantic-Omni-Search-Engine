/* ====================================================================
   Semantic Omni Search Engine – Application JavaScript
   ==================================================================== */

const API_BASE = window.location.origin;

// ── Utility helpers ──────────────────────────────────────────────────

function formatBytes(bytes) {
  if (!bytes) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(1)} ${units[i]}`;
}

function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}

function modalityIcon(type) {
  const icons = { text: '📄', image: '🖼', audio: '🎵', video: '🎬', pdf: '📑' };
  return icons[type] || '📄';
}

function fileIcon(filename = '', type = '') {
  const ext = filename.split('.').pop().toLowerCase();
  const icons = {
    pdf: '📑', jpg: '🖼', jpeg: '🖼', png: '🖼', gif: '🖼', webp: '🖼', bmp: '🖼',
    mp3: '🎵', wav: '🎵', ogg: '🎵', flac: '🎵', m4a: '🎵',
    mp4: '🎬', avi: '🎬', mov: '🎬', webm: '🎬', mkv: '🎬',
    txt: '📄', md: '📄', csv: '📊', html: '🌐',
  };
  return icons[ext] || modalityIcon(type) || '📄';
}

function escapeHtml(str) {
  return String(str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── State ─────────────────────────────────────────────────────────────

let currentView = 'search';
let selectedModality = '';
let lastSearchResults = [];
let _resultMap = {};  // stores results by file_id for safe modal access

// ── Navigation ────────────────────────────────────────────────────────

function switchView(viewName) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`view${capitalize(viewName)}`).classList.add('active');
  document.querySelector(`[data-view="${viewName}"]`).classList.add('active');
  currentView = viewName;

  if (viewName === 'files') loadFiles();
  if (viewName === 'history') loadHistory();
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => switchView(btn.dataset.view));
});

// ── Stats ─────────────────────────────────────────────────────────────

async function loadStats() {
  try {
    const data = await apiFetch('/stats');
    const sizes = data.vector_index_sizes || {};
    const el = document.getElementById('statsContent');
    const rows = Object.entries(sizes).map(([mod, count]) =>
      `<div class="stats-row"><span>${modalityIcon(mod)} ${mod}</span><span>${count}</span></div>`
    ).join('');
    el.innerHTML = rows || '<span class="loading-dots">No data</span>';
  } catch {
    document.getElementById('statsContent').textContent = 'Unavailable';
  }
}

loadStats();
setInterval(loadStats, 30000);

// ── Search ─────────────────────────────────────────────────────────────

document.querySelectorAll('#modalityFilter .pill').forEach(pill => {
  pill.addEventListener('click', () => {
    document.querySelectorAll('#modalityFilter .pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    selectedModality = pill.dataset.value;
  });
});

document.getElementById('searchInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') performSearch();
});

document.getElementById('searchBtn').addEventListener('click', performSearch);

async function performSearch() {
  const query = document.getElementById('searchInput').value.trim();
  if (!query) return;

  const topK = parseInt(document.getElementById('topKInput').value) || 10;
  const minScore = parseFloat(document.getElementById('minScoreInput').value) || 0;

  const status = document.getElementById('searchStatus');
  const resultsEl = document.getElementById('searchResults');

  status.className = 'search-status searching';
  status.textContent = '⏳ Searching…';
  status.classList.remove('hidden');
  resultsEl.innerHTML = '';

  const body = { query, top_k: topK, min_score: minScore };
  if (selectedModality) body.modality = selectedModality;

  try {
    const data = await apiFetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    lastSearchResults = data.results || [];
    _resultMap = {};
    lastSearchResults.forEach((r, i) => { _resultMap[i] = r; });

    if (!lastSearchResults.length) {
      status.className = 'search-status no-results';
      status.textContent = '🔍 No results found. Try a different query or upload more files.';
    } else {
      status.className = 'search-status';
      status.textContent =
        `✅ Found ${data.total} result${data.total !== 1 ? 's' : ''} ` +
        `in ${data.latency_ms.toFixed(0)}ms ` +
        `(embed: ${data.embedding_time_ms.toFixed(0)}ms, ` +
        `retrieve: ${data.retrieval_time_ms.toFixed(0)}ms)`;
      renderResults(lastSearchResults);
    }
  } catch (err) {
    status.className = 'search-status error';
    status.textContent = `❌ Search failed: ${err.message}`;
  }
}

function renderResults(results) {
  const el = document.getElementById('searchResults');
  el.innerHTML = results.map((r, i) => `
    <div class="result-card" onclick="openResultModal(${i})">
      <div class="result-card-header">
        <div class="result-rank">${r.rank}</div>
        <div class="result-filename" title="${escapeHtml(r.filename)}">${escapeHtml(r.filename)}</div>
        <span class="modality-badge badge-${r.file_type}">${modalityIcon(r.file_type)} ${r.file_type}</span>
      </div>

      <div class="score-section">
        <div class="score-bar-track">
          <div class="score-bar-fill" style="width:${(r.similarity_score * 100).toFixed(1)}%"></div>
        </div>
        <div class="score-label">
          <span>Similarity</span>
          <span class="score-value">${(r.similarity_score * 100).toFixed(1)}%</span>
        </div>
      </div>

      ${r.content_preview ? `<div class="result-preview">${escapeHtml(r.content_preview)}</div>` : ''}

      ${r.matching_keywords.length ? `
        <div class="result-keywords">
          ${r.matching_keywords.slice(0, 6).map(k => `<span class="keyword-tag">${escapeHtml(k)}</span>`).join('')}
        </div>
      ` : ''}

      <div class="result-explanation">${escapeHtml(r.explanation)}</div>

      <div class="result-meta">
        📦 ${formatBytes(r.file_size)} &nbsp;|&nbsp; 🕒 ${formatDate(r.created_at)}
      </div>
    </div>
  `).join('');
}

// ── Upload ─────────────────────────────────────────────────────────────

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('click', e => {
  if (e.target.tagName !== 'LABEL') fileInput.click();
});

fileInput.addEventListener('change', () => handleFiles(Array.from(fileInput.files)));

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  handleFiles(Array.from(e.dataTransfer.files));
});

async function handleFiles(files) {
  if (!files.length) return;

  const queue = document.getElementById('uploadQueue');
  const queueItems = document.getElementById('queueItems');
  queue.classList.remove('hidden');

  for (const file of files) {
    const itemId = `qi-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    queueItems.insertAdjacentHTML('beforeend', `
      <div class="queue-item" id="${itemId}">
        <div class="queue-item-icon">${fileIcon(file.name)}</div>
        <div class="queue-item-info">
          <div class="queue-item-name">${escapeHtml(file.name)}</div>
          <div class="queue-item-status status-uploading">Uploading…</div>
          <div class="queue-progress"><div class="queue-progress-fill" style="width:30%"></div></div>
        </div>
      </div>
    `);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const result = await apiFetch('/upload', { method: 'POST', body: formData });
      const el = document.getElementById(itemId);
      el.querySelector('.queue-item-status').className = 'queue-item-status status-success';
      el.querySelector('.queue-item-status').textContent =
        `✅ Indexed as ${result.file_type} — ${formatBytes(result.file_size)}`;
      el.querySelector('.queue-progress-fill').style.width = '100%';
      loadStats();
    } catch (err) {
      const el = document.getElementById(itemId);
      el.querySelector('.queue-item-status').className = 'queue-item-status status-error';
      el.querySelector('.queue-item-status').textContent = `❌ ${err.message}`;
      el.querySelector('.queue-progress-fill').style.width = '0%';
    }
  }

  fileInput.value = '';
}

// ── Files ─────────────────────────────────────────────────────────────

document.getElementById('refreshFilesBtn').addEventListener('click', loadFiles);
document.getElementById('filesModalityFilter').addEventListener('change', loadFiles);

async function loadFiles() {
  const modality = document.getElementById('filesModalityFilter').value;
  const el = document.getElementById('filesTable');
  el.innerHTML = '<span class="loading-dots">Loading files…</span>';

  try {
    const params = new URLSearchParams({ limit: 200 });
    if (modality) params.set('modality', modality);
    const data = await apiFetch(`/files?${params}`);

    if (!data.files.length) {
      el.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">📁</div>
          <p>No files indexed yet. Upload some files to get started.</p>
        </div>`;
      return;
    }

    el.innerHTML = `
      <table class="files-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Type</th>
            <th>Size</th>
            <th>Uploaded</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          ${data.files.map(f => `
            <tr>
              <td class="file-name" title="${escapeHtml(f.filename)}">${fileIcon(f.filename, f.file_type)} ${escapeHtml(f.filename)}</td>
              <td><span class="modality-badge badge-${escapeHtml(f.file_type)}">${escapeHtml(f.file_type)}</span></td>
              <td>${formatBytes(f.file_size)}</td>
              <td>${formatDate(f.created_at)}</td>
              <td>
                <button class="btn btn-danger btn-sm delete-file-btn"
                  data-file-id="${escapeHtml(f.file_id)}"
                  data-filename="${escapeHtml(f.filename)}">🗑 Delete</button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
      <p style="margin-top:12px;color:var(--text-muted);font-size:13px">${data.total} file${data.total !== 1 ? 's' : ''} total</p>
    `;

    // Attach delete handlers via event delegation
    el.querySelectorAll('.delete-file-btn').forEach(btn => {
      btn.addEventListener('click', function() {
        deleteFile(this.dataset.fileId, this.dataset.filename, this);
      });
    });
  } catch (err) {
    el.innerHTML = `<p style="color:var(--error)">Failed to load files: ${escapeHtml(err.message)}</p>`;
  }
}

async function deleteFile(fileId, filename, btn) {
  if (!confirm(`Delete "${filename}"? This cannot be undone.`)) return;
  btn.disabled = true;
  btn.textContent = '…';
  try {
    await apiFetch(`/file/${fileId}`, { method: 'DELETE' });
    loadFiles();
    loadStats();
  } catch (err) {
    alert(`Failed to delete: ${err.message}`);
    btn.disabled = false;
    btn.textContent = '🗑 Delete';
  }
}

// ── History ───────────────────────────────────────────────────────────

document.getElementById('refreshHistoryBtn').addEventListener('click', loadHistory);

async function loadHistory() {
  const el = document.getElementById('historyList');
  el.innerHTML = '<span class="loading-dots">Loading history…</span>';

  try {
    const data = await apiFetch('/search/history?limit=50');
    if (!data.history.length) {
      el.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">📜</div>
          <p>No searches yet. Run a search to see your history here.</p>
        </div>`;
      return;
    }

    el.innerHTML = data.history.map(h => `
      <div class="history-item" onclick="replaySearch('${escapeHtml(h.query)}')">
        <div class="history-query">${escapeHtml(h.query)}</div>
        <div class="history-meta">
          <span>🔍 ${h.query_type}</span>
          <span>📊 ${h.result_count} results</span>
          <span>⚡ ${h.latency_ms ? h.latency_ms.toFixed(0) + ' ms' : '—'}</span>
          <span>🕒 ${formatDate(h.created_at)}</span>
        </div>
      </div>
    `).join('');
  } catch (err) {
    el.innerHTML = `<p style="color:var(--error)">Failed to load history: ${escapeHtml(err.message)}</p>`;
  }
}

function replaySearch(query) {
  document.getElementById('searchInput').value = query;
  switchView('search');
  performSearch();
}

// ── Modal ──────────────────────────────────────────────────────────────

document.getElementById('modalClose').addEventListener('click', closeModal);
document.getElementById('modalBackdrop').addEventListener('click', closeModal);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

function closeModal() {
  document.getElementById('previewModal').classList.add('hidden');
}

// Make openResultModal available globally (called from inline onclick)
window.openResultModal = function(idx) {
  const result = _resultMap[idx];
  if (!result) return;
  const body = document.getElementById('modalBody');
  body.innerHTML = `
    <h3 class="modal-title">${escapeHtml(result.filename)}</h3>
    <div class="modal-field">
      <div class="modal-field-label">Modality</div>
      <div class="modal-field-value"><span class="modality-badge badge-${result.file_type}">${modalityIcon(result.file_type)} ${result.file_type}</span></div>
    </div>
    <div class="modal-field">
      <div class="modal-field-label">Similarity Score</div>
      <div class="modal-field-value score-value">${(result.similarity_score * 100).toFixed(2)}%</div>
    </div>
    <div class="modal-field">
      <div class="modal-field-label">Embedding Distance</div>
      <div class="modal-field-value">${result.embedding_distance.toFixed(4)}</div>
    </div>
    <div class="modal-field">
      <div class="modal-field-label">Why this result?</div>
      <div class="modal-field-value">${escapeHtml(result.explanation)}</div>
    </div>
    ${result.matching_keywords.length ? `
      <div class="modal-field">
        <div class="modal-field-label">Matching Keywords</div>
        <div class="modal-field-value"><div class="result-keywords" style="margin:0">${result.matching_keywords.map(k => `<span class="keyword-tag">${escapeHtml(k)}</span>`).join('')}</div></div>
      </div>
    ` : ''}
    ${result.content_preview ? `
      <div class="modal-field">
        <div class="modal-field-label">Content Preview</div>
        <div class="modal-field-value" style="white-space:pre-wrap;max-height:200px;overflow-y:auto;background:var(--bg-secondary);padding:12px;border-radius:var(--radius-sm)">${escapeHtml(result.content_preview)}</div>
      </div>
    ` : ''}
    <div class="modal-field">
      <div class="modal-field-label">File Details</div>
      <div class="modal-field-value" style="color:var(--text-secondary)">
        Size: ${formatBytes(result.file_size)} &nbsp;|&nbsp; Type: ${result.mime_type || result.file_type} &nbsp;|&nbsp; Uploaded: ${formatDate(result.created_at)}
      </div>
    </div>
  `;
  document.getElementById('previewModal').classList.remove('hidden');
};
