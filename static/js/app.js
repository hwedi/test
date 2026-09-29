// app.js - the Analyser page.
//
// The flow, in plain words:
//   1. Ask the server "are the models loaded?" (/api/health) until the answer is yes.
//   2. The visitor uploads a CSV (or picks an example).
//   3. We send it to the server (/api/predict or /api/samples/<id>) and get JSON back.
//   4. We draw that JSON as cards, bars, a range bar and four charts.
(function () {
  'use strict';

  // ---------- Shortcuts to page elements ----------
  var $ = function (id) { return document.getElementById(id); };
  var els = {
    wake: $('wake-banner'), mock: $('mock-strip'), busy: $('busy'),
    fileInput: $('file-input'), dropzone: $('dropzone'),
    sampleSelect: $('sample-select'), sampleBtn: $('sample-btn'), sampleDownload: $('sample-download'),
    errorBox: $('error-box'), errorTitle: $('error-title'), errorText: $('error-text'), errorExample: $('error-example'),
    results: $('results'), resultsTitle: $('results-title'), resultName: $('result-name')
  };

  var lastAnalysis = null;   // kept so the charts can be redrawn when the theme changes
  var charts = [];           // Chart.js objects currently on screen

  // ---------- Small helpers ----------
  // Build an element safely (textContent, never innerHTML, so server text can't inject HTML)
  function el(tag, className, text) {
    var e = document.createElement(tag);
    if (className) e.className = className;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function fmtDays(n) { return Math.round(n).toLocaleString('en'); }
  function fmtPct(p, digits) { return (p * 100).toFixed(digits === undefined ? 1 : digits) + '%'; }
  function clampPct(p) { return Math.max(0, Math.min(100, p)); }
  function classLabel(c) { return 'Class ' + c.class + ' - ' + c.name; }

  // ---------- 1. Waiting for the models ----------
  function setControlsEnabled(on) {
    els.fileInput.disabled = !on;
    els.sampleSelect.disabled = !on;
    els.sampleBtn.disabled = !on;
    els.dropzone.classList.toggle('is-disabled', !on);
  }

  function checkHealth() {
    fetch('/api/health')
      .then(function (r) { return r.json(); })
      .then(function (h) {
        els.mock.hidden = h.mode !== 'mock';
        if (h.ready) {
          els.wake.hidden = true;
          loadSamples();
        } else if (h.error) {
          showWake('The models could not be loaded', h.error);      // permanent problem: stop polling
        } else {
          showWake('Waking up the models…',
            'The server is loading the two AI models. This can take a minute or two after a period of inactivity.');
          setTimeout(checkHealth, 2500);
        }
      })
      .catch(function () {                                          // server itself not answering yet: keep trying
        showWake('Waking up the server…', 'Waiting for the server to respond.');
        setTimeout(checkHealth, 2500);
      });
  }

  function showWake(title, text) {
    els.wake.hidden = false;
    els.wake.querySelector('strong').textContent = title;
    els.wake.querySelector('div span').textContent = text;
  }

  function loadSamples() {
    fetch('/api/samples')
      .then(function (r) { return r.json(); })
      .then(function (list) {
        list.forEach(function (s) {
          var o = el('option', '', s.id.replace('.csv', ''));
          o.value = s.id;
          els.sampleSelect.appendChild(o);
        });
        updateDownloadLinks();
        setControlsEnabled(true);
      });
  }

  function updateDownloadLinks() {
    var href = '/api/samples/' + encodeURIComponent(els.sampleSelect.value) + '/csv';
    els.sampleDownload.href = href;
    els.errorExample.href = href;
  }

  // ---------- 2 + 3. Sending data to the server ----------
  function setBusy(on) {
    els.busy.hidden = !on;
    setControlsEnabled(!on);
    if (on) { els.errorBox.hidden = true; els.results.hidden = true; }
  }

  function showError(message) {
    els.errorText.textContent = message;
    els.errorBox.hidden = false;
    els.errorBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Send a request, then either draw the results or show a friendly error
  function request(url, options) {
    setBusy(true);
    fetch(url, options)
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (body) { return { ok: r.ok, body: body }; });
      })
      .then(function (res) {
        setBusy(false);
        if (res.ok) showResults(res.body);
        else showError(typeof res.body.detail === 'string' ? res.body.detail : 'Something went wrong. Please try again.');
      })
      .catch(function () {
        setBusy(false);
        showError('Could not reach the server. Check your connection and try again.');
      });
  }

  function analyseFile(file) {
    if (!file) return;
    if (file.size > 2 * 1024 * 1024) { showError('The file is larger than 2 MB. A valid file is only a few tens of kilobytes.'); return; }
    var form = new FormData();
    form.append('file', file);
    request('/api/predict', { method: 'POST', body: form });
  }

  function analyseSample() {
    request('/api/samples/' + encodeURIComponent(els.sampleSelect.value));
  }

  // ---------- 4. Drawing the results ----------
  function showResults(a) {
    lastAnalysis = a;
    els.resultName.textContent = a.name;
    renderFdd(a);
    renderRul(a);
    renderCompare(a);
    renderCharts(a);
    els.results.hidden = false;
    els.resultsTitle.focus();                       // moves keyboard / screen-reader focus to the results
    els.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // --- Fault diagnosis card ---
  function renderFdd(a) {
    var f = a.fdd;
    var gauge = $('gauge-wrap');
    gauge.classList.toggle('is-low', f.needs_review);
    gauge.setAttribute('aria-label', 'Confidence ' + fmtPct(f.confidence) + '. Review threshold ' + fmtPct(f.threshold, 0) + '.');
    $('gauge-pct').textContent = (f.confidence * 100).toFixed(1);
    $('gauge-fill').setAttribute('stroke-dasharray', (f.confidence * 100) + ' 100');   // arc length is 0-100

    // A small tick on the arc marking the low-confidence threshold
    var angle = Math.PI * (1 - f.threshold), tick = $('gauge-tick');
    tick.setAttribute('x1', 50 + 36 * Math.cos(angle)); tick.setAttribute('y1', 50 - 36 * Math.sin(angle));
    tick.setAttribute('x2', 50 + 49 * Math.cos(angle)); tick.setAttribute('y2', 50 - 49 * Math.sin(angle));

    var badge = $('class-badge');
    badge.textContent = 'Class ' + f.class;
    badge.className = 'class-badge cls-' + f.class;
    $('class-name').textContent = f.name;

    $('review-flag').hidden = !f.needs_review;
    $('review-detail').textContent = 'Confidence is below the ' + fmtPct(f.threshold, 0) + ' threshold.';

    var list = $('prob-bars');
    list.textContent = '';
    f.probabilities.forEach(function (p) {
      var top = p.class === f.class;
      var row = el('li', 'bar-row' + (top ? ' is-top cls-' + p.class : ''));
      row.appendChild(el('span', 'bar-label', 'Class ' + p.class + ' · ' + p.name + (top ? ' (predicted)' : '')));
      var track = el('span', 'bar-track'); track.setAttribute('aria-hidden', 'true');
      var fill = el('span', 'bar-fill'); track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el('span', 'bar-value', fmtPct(p.probability)));
      list.appendChild(row);
      requestAnimationFrame(function () { fill.style.width = (p.probability * 100) + '%'; });   // animate to its width
    });
  }

  // --- Remaining life card ---
  function renderRul(a) {
    var r = a.rul, actual = a.actual ? a.actual.rul_days : null;
    $('rul-days').textContent = fmtDays(r.days);
    $('rul-months').textContent = r.months.toFixed(1);

    // The bar's scale runs from 0 to a bit past the largest thing we draw
    var max = Math.max(r.cap_days * 1.1, r.high * 1.05, actual === null ? 0 : actual * 1.05);
    var pos = function (days) { return clampPct(days / max * 100) + '%'; };
    $('range-span').style.left = pos(r.low);
    $('range-span').style.width = clampPct((r.high - r.low) / max * 100) + '%';
    $('range-dot').style.left = pos(r.days);
    $('range-cap').style.left = pos(r.cap_days);
    $('range-max').textContent = fmtDays(max) + ' days';
    $('range-actual').hidden = actual === null;
    $('legend-actual').hidden = actual === null;
    if (actual !== null) $('range-actual').style.left = pos(actual);

    $('range').setAttribute('role', 'img');
    $('range').setAttribute('aria-label', 'Estimated ' + fmtDays(r.days) + ' days, likely range ' +
      fmtDays(r.low) + ' to ' + fmtDays(r.high) + ' days.');
    $('rul-range-text').textContent = 'Likely range: ' + fmtDays(r.low) + ' to ' + fmtDays(r.high) + ' days (±' +
      fmtDays(r.radius) + ' days). About 9 in 10 predictions fell inside this range in validation and testing.';

    var note = $('near-cap-note');
    note.hidden = !r.near_cap;
    note.textContent = 'The longest life in the training data is ' + fmtDays(r.cap_days) +
      ' days, so a prediction this high means “at least this long”.';
  }

  // --- Predicted vs actual (only for the example transformers) ---
  function renderCompare(a) {
    $('compare').hidden = !a.actual;
    if (!a.actual) return;
    var verdict = function (cell, good, text) { cell.textContent = text; cell.className = 'verdict ' + (good ? 'good' : 'bad'); };

    $('cmp-fdd-pred').textContent = classLabel(a.fdd);
    $('cmp-fdd-act').textContent = classLabel(a.actual.fdd);
    var same = a.fdd.class === a.actual.fdd.class;
    verdict($('cmp-fdd-res'), same, same ? 'Correct' : 'Incorrect');

    var diff = a.rul.days - a.actual.rul_days;
    var inside = a.actual.rul_days >= a.rul.low && a.actual.rul_days <= a.rul.high;
    $('cmp-rul-pred').textContent = fmtDays(a.rul.days) + ' days';
    $('cmp-rul-act').textContent = fmtDays(a.actual.rul_days) + ' days';
    verdict($('cmp-rul-res'), inside, 'Off by ' + fmtDays(Math.abs(diff)) + ' days (' +
      (inside ? 'inside' : 'outside') + ' the likely range)');
  }

  // --- Four gas charts (Chart.js) ---
  function renderCharts(a) {
    charts.forEach(function (c) { c.destroy(); });
    charts = [];
    var grid = $('gas-grid');
    grid.textContent = '';

    // Read the current theme colours from the CSS, so the charts match light/dark mode
    var css = getComputedStyle(document.documentElement);
    var line = css.getPropertyValue('--chart-line').trim();
    var gridColor = css.getPropertyValue('--chart-grid').trim();
    var textColor = css.getPropertyValue('--chart-text').trim();
    var reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

    Object.keys(a.series.gases).forEach(function (gas) {
      var values = a.series.gases[gas], days = a.series.days;
      var points = values.map(function (v, i) { return { x: days[i], y: v }; });

      var fig = el('figure');
      fig.appendChild(el('figcaption', '', gas));
      var box = el('div', 'chart-box');
      var canvas = document.createElement('canvas');
      canvas.setAttribute('role', 'img');
      canvas.setAttribute('aria-label', gas + ' readings over ' + fmtDays(days[days.length - 1]) + ' days: from ' +
        values[0].toFixed(1) + ' to ' + values[values.length - 1].toFixed(1) + ', peak ' + Math.max.apply(null, values).toFixed(1) + '.');
      box.appendChild(canvas); fig.appendChild(box); grid.appendChild(fig);

      charts.push(new Chart(canvas, {
        type: 'line',
        data: { datasets: [{ data: points, borderColor: line, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0 }] },
        options: {
          parsing: false, animation: reduceMotion ? false : { duration: 500 },
          maintainAspectRatio: false, responsive: true,
          interaction: { mode: 'nearest', axis: 'x', intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: {
              title: function (items) { return 'Day ' + items[0].parsed.x; },
              label: function (item) { return gas + ': ' + item.parsed.y.toFixed(2); }
            } }
          },
          scales: {
            x: { type: 'linear', min: 0, max: days[days.length - 1],
                 title: { display: true, text: 'Days of monitoring', color: textColor },
                 ticks: { color: textColor, maxTicksLimit: 7 }, grid: { color: gridColor } },
            y: { beginAtZero: true, ticks: { color: textColor, maxTicksLimit: 6 }, grid: { color: gridColor } }
          }
        }
      }));
    });
  }

  // ---------- Wiring up the page ----------
  els.fileInput.addEventListener('change', function () {
    analyseFile(els.fileInput.files[0]);
    els.fileInput.value = '';                       // so choosing the same file again still works
  });

  // Drag and drop
  ['dragenter', 'dragover'].forEach(function (type) {
    els.dropzone.addEventListener(type, function (e) {
      e.preventDefault();
      if (!els.fileInput.disabled) els.dropzone.classList.add('is-dragover');
    });
  });
  ['dragleave', 'drop'].forEach(function (type) {
    els.dropzone.addEventListener(type, function () { els.dropzone.classList.remove('is-dragover'); });
  });
  els.dropzone.addEventListener('drop', function (e) {
    e.preventDefault();
    if (!els.fileInput.disabled) analyseFile(e.dataTransfer.files[0]);
  });
  // A file dropped outside the box should not make the browser open it and leave the page
  window.addEventListener('dragover', function (e) { e.preventDefault(); });
  window.addEventListener('drop', function (e) { e.preventDefault(); });

  els.sampleSelect.addEventListener('change', updateDownloadLinks);
  els.sampleBtn.addEventListener('click', analyseSample);
  document.addEventListener('themechange', function () { if (lastAnalysis) renderCharts(lastAnalysis); });

  checkHealth();
})();
