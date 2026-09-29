// common.js - small helpers shared by every page: the light/dark switch.
(function () {
  'use strict';
  var root = document.documentElement;
  var button = document.getElementById('theme-toggle');

  function currentTheme() { return root.dataset.theme === 'dark' ? 'dark' : 'light'; }

  // The button always offers the OTHER theme
  function updateButton() {
    if (button) button.textContent = currentTheme() === 'dark' ? 'Light mode' : 'Dark mode';
  }

  if (button) {
    button.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      try { localStorage.setItem('theme', next); } catch (e) {}   // remember the choice (if the browser allows)
      updateButton();
      // Tell other scripts (e.g. the charts) to redraw with the new colours
      document.dispatchEvent(new Event('themechange'));
    });
  }
  updateButton();
})();
