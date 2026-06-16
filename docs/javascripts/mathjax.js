window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  startup: {
    pageReady: function() {
      return MathJax.startup.defaultPageReady().then(function() {
        document.addEventListener("DOMContentSwitch", function() {
          MathJax.typesetPromise();
        });
      });
    }
  }
};
