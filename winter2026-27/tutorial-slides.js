/*
 * Auto-links tutorial slides on the Winter 2026–27 schedule.
 *
 * For every schedule row with week number N, looks for a PDF whose name
 * contains "tutorial" inside winter2026-27/resources/weekN/ and adds a
 * "Tutorial Slides" link to that row's Resources cell.
 *
 * To publish a new tutorial, just upload e.g. resources/week3/tutorial 3.pdf —
 * no HTML edits needed.
 *
 * Discovery: GitHub Pages cannot list directories, so the file list comes
 * from the GitHub API (one request, cached per browser session). If the API
 * is unavailable (e.g. rate-limited), common file names are probed directly.
 */
(function () {
  'use strict';

  var REPO = 'assafshocher/mcv';
  var BRANCH = 'main';
  var SEMESTER_DIR = 'winter2026-27';
  var RESOURCES_DIR = 'resources';            // relative to this page
  var CACHE_KEY = 'mcv-winter-tutorials-v1';
  var CACHE_TTL_MS = 10 * 60 * 1000;
  var TUTORIAL_RE = new RegExp(
    '^' + SEMESTER_DIR + '/' + RESOURCES_DIR + '/week(\\d+)/([^/]*tutorial[^/]*\\.pdf)$', 'i');

  function encodePath(p) {
    return p.split('/').map(encodeURIComponent).join('/');
  }

  function readCache() {
    try {
      var c = JSON.parse(sessionStorage.getItem(CACHE_KEY));
      if (c && Date.now() - c.t < CACHE_TTL_MS) return c.map;
    } catch (e) { /* storage unavailable */ }
    return null;
  }

  function writeCache(map) {
    try { sessionStorage.setItem(CACHE_KEY, JSON.stringify({ t: Date.now(), map: map })); } catch (e) {}
  }

  // Returns {weekNumber: fileName}
  function fromGitHubApi() {
    var url = 'https://api.github.com/repos/' + REPO + '/git/trees/' + BRANCH + '?recursive=1';
    return fetch(url, { headers: { Accept: 'application/vnd.github+json' } })
      .then(function (r) {
        if (!r.ok) throw new Error('GitHub API ' + r.status);
        return r.json();
      })
      .then(function (data) {
        var map = {};
        (data.tree || []).forEach(function (item) {
          if (item.type !== 'blob') return;
          var m = TUTORIAL_RE.exec(item.path);
          if (!m) return;
          var wk = parseInt(m[1], 10);
          // Prefer the shortest name if a folder somehow holds several tutorial PDFs.
          if (!map[wk] || m[2].length < map[wk].length) map[wk] = m[2];
        });
        return map;
      });
  }

  function candidateNames(wk) {
    return [
      'tutorial ' + wk + '.pdf', 'tutorial.pdf', 'tutorial_' + wk + '.pdf', 'tutorial' + wk + '.pdf',
      'Tutorial ' + wk + '.pdf', 'Tutorial.pdf', 'Tutorial_' + wk + '.pdf', 'Tutorial' + wk + '.pdf'
    ];
  }

  function exists(url) {
    return fetch(url, { method: 'HEAD', cache: 'no-cache' })
      .then(function (r) { return r.ok; })
      .catch(function () { return false; });
  }

  function fromProbing(weeks) {
    var map = {};
    return Promise.all(weeks.map(function (wk) {
      var names = candidateNames(wk);
      var i = 0;
      function next() {
        if (i >= names.length) return null;
        var name = names[i++];
        return exists(RESOURCES_DIR + '/week' + wk + '/' + encodeURIComponent(name))
          .then(function (ok) { if (ok) { map[wk] = name; return null; } return next(); });
      }
      return next();
    })).then(function () { return map; });
  }

  function addLink(row, wk, fileName) {
    var cell = row.cells[row.cells.length - 1];
    if (!cell || cell.querySelector('a.tutorial-slides-link')) return;

    var a = document.createElement('a');
    a.href = RESOURCES_DIR + '/week' + wk + '/' + encodeURIComponent(fileName);
    a.className = 'resource-link tutorial-slides-link';
    a.textContent = 'Tutorial Slides';

    if (cell.textContent.trim() === '—' && !cell.querySelector('a')) {
      cell.textContent = '';
      cell.appendChild(a);
      return;
    }

    // Place after the last lecture-related link (as on the Spring page), else first.
    var links = cell.querySelectorAll('a');
    var anchor = null;
    for (var i = 0; i < links.length; i++) {
      if (/lecture/i.test(links[i].textContent)) anchor = links[i];
    }
    if (anchor) {
      var next = anchor.nextElementSibling;
      if (next && next.tagName === 'BR') {
        // "...Lecture link<br>" + "Tutorial Slides<br>" + rest
        next.after(a, document.createElement('br'));
      } else {
        // Lecture link was the last item: "<br>Tutorial Slides"
        anchor.after(document.createElement('br'), a);
      }
    } else {
      cell.insertBefore(document.createElement('br'), cell.firstChild);
      cell.insertBefore(a, cell.firstChild);
    }
  }

  function run() {
    var rows = [];
    document.querySelectorAll('.schedule-table tbody tr').forEach(function (row) {
      var numCell = row.querySelector('.wk-num');
      var wk = numCell && parseInt(numCell.textContent, 10);
      if (wk) rows.push({ row: row, wk: wk });
    });
    if (!rows.length) return;

    var cached = readCache();
    var source = cached ? Promise.resolve(cached)
      : fromGitHubApi()
          .then(function (map) { writeCache(map); return map; })
          .catch(function () { return fromProbing(rows.map(function (r) { return r.wk; })); });

    source.then(function (map) {
      rows.forEach(function (r) { if (map[r.wk]) addLink(r.row, r.wk, map[r.wk]); });
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run);
  else run();
})();
