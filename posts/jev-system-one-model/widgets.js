/* Widget for "Calibrated Decisions, Part 1: Jev and the System One Model".
 *
 * widget-constrained: a toy language model spells a department label in two
 * tokens. Valid labels: "billing" (bill + ing) and "technical" (tech + nical).
 * Invalid strings: "refund" (a label outside the schema), "billed" and
 * "techno". Three distributions over the two valid labels:
 *   1. unconstrained: the probability of each full string, plus the mass on
 *      invalid strings;
 *   2. constrained decoding: at each step, mask invalid tokens and renormalise
 *      what is left, the usual grammar-constrained generation;
 *   3. conditioned: P(label | output is a valid label), the distribution the
 *      same model assigns to the two labels once invalid strings are ruled out.
 * With the defaults, (2) and (3) pick different labels.
 *
 * The model lives in `CAL1`, separate from the drawing, and is exported under
 * Node so the prose's claims can be checked. Built on widget-kit (WK); colours
 * are theme tokens.
 */
"use strict";

var CAL1 = (function () {
  // w: unnormalised first-token weights {bill, tech, refund};
  // d = P(ing | bill), e = P(nical | tech).
  function evaluate(w, d, e) {
    var z = w.bill + w.tech + w.refund;
    var a = w.bill / z, b = w.tech / z, c = w.refund / z;
    var billing = a * d, technical = b * e, invalid = c + a * (1 - d) + b * (1 - e);
    var local = { billing: a / (a + b), technical: b / (a + b) };
    var valid = billing + technical;
    var cond = valid > 0 ? { billing: billing / valid, technical: technical / valid } : { billing: NaN, technical: NaN };
    return {
      first: { bill: a, tech: b, refund: c },
      strings: { billing: billing, technical: technical, invalid: invalid },
      local: local,
      conditioned: cond,
      argmaxLocal: local.billing >= local.technical ? "billing" : "technical",
      argmaxCond: cond.billing >= cond.technical ? "billing" : "technical"
    };
  }
  return { evaluate: evaluate };
})();

if (typeof module === "object" && module.exports) module.exports = CAL1;

if (typeof WK !== "undefined") {
  WK.mount("widget-constrained", function (root, WK) {
    var state = { bill: 4.5, tech: 3.5, refund: 2, d: 0.6, e: 0.9 };
    var f = WK.frame({
      title: "Constrained decoding is not conditioning",
      note: "A toy model spells a department label in two tokens. Dashed branches spell strings " +
        "outside the schema. Constrained decoding masks them one step at a time; conditioning " +
        "drops them from the whole-string distribution and renormalises once."
    });
    [["bill", "Weight of first token \"bill\"", 0, 10, 0.5],
     ["tech", "Weight of first token \"tech\"", 0, 10, 0.5],
     ["refund", "Weight of first token \"refund\" (not a label)", 0, 10, 0.5],
     ["d", "P(\"ing\" | \"bill\"); the rest is \"ed\"", 0.01, 1, 0.01],
     ["e", "P(\"nical\" | \"tech\"); the rest is \"no\"", 0.01, 1, 0.01]].forEach(function (s) {
      var sl = WK.slider({ label: s[1], min: s[2], max: s[3], step: s[4], value: state[s[0]], fmt: WK.fmt.num,
        oninput: function (v) { state[s[0]] = v; draw(); } });
      f.controls.appendChild(sl.root);
    });

    var W = 520, H = 450;
    var svg = WK.svg(W, H, { "aria-label": "Token tree and three distributions over the valid labels" });
    f.body.appendChild(svg);
    var stLocal = WK.stat({ label: "Constrained decoding picks", value: "" });
    var stCond = WK.stat({ label: "Conditioning picks", value: "" });
    var stInv = WK.stat({ label: "Mass on invalid strings", value: "" });
    [stLocal, stCond, stInv].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    function node(x, y, label, valid) {
      svg.appendChild(WK.h("rect", { x: x - 44, y: y - 13, width: 88, height: 26, rx: 5,
        fill: "surface", stroke: valid ? "c1" : "muted", "stroke-dasharray": valid ? null : "4 3" }));
      svg.appendChild(WK.h("text", { x: x, y: y + 5, "font-size": 13, "text-anchor": "middle",
        fill: valid ? "ink" : "muted" }, label));
    }
    function edge(x1, y1, x2, y2, p, valid) {
      svg.appendChild(WK.h("line", { x1: x1 + 44, y1: y1, x2: x2 - 44, y2: y2,
        stroke: valid ? "c1" : "muted", "stroke-width": 1 + 5 * p, "stroke-dasharray": valid ? null : "5 4",
        "stroke-opacity": valid ? 0.9 : 0.6 }));
      svg.appendChild(WK.h("text", { x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 6, "font-size": 12,
        "text-anchor": "middle", fill: valid ? "ink" : "muted" }, p.toFixed(2)));
    }

    function draw() {
      var r = CAL1.evaluate(state, state.d, state.e);
      WK.clear(svg);
      // Tree.
      var rx = 50, mx = 230, lx = 420;
      node(rx, 120, "start", true);
      node(mx, 40, "bill", true); node(mx, 120, "tech", true); node(mx, 200, "refund ✗", false);
      node(lx, 16, "billing", true); node(lx, 64, "billed ✗", false);
      node(lx, 104, "technical", true); node(lx, 150, "techno ✗", false);
      edge(rx, 120, mx, 40, r.first.bill, true);
      edge(rx, 120, mx, 120, r.first.tech, true);
      edge(rx, 120, mx, 200, r.first.refund, false);
      edge(mx, 40, lx, 16, state.d, true);
      edge(mx, 40, lx, 64, 1 - state.d, false);
      edge(mx, 120, lx, 104, state.e, true);
      edge(mx, 120, lx, 150, 1 - state.e, false);
      // Bars.
      var groups = [
        ["Unconstrained strings", [r.strings.billing, r.strings.technical, r.strings.invalid]],
        ["Constrained decoding", [r.local.billing, r.local.technical]],
        ["Conditioned on a valid label", [r.conditioned.billing, r.conditioned.technical]]
      ];
      var by = 250, bh = 150, gw = 160, bw = 34, colours = ["c1", "c2", "muted"], names = ["billing", "technical", "invalid"];
      var sy = WK.lin([0, 1], [by + bh, by]);
      svg.appendChild(WK.h("line", { x1: 10, x2: W - 10, y1: by + bh, y2: by + bh, stroke: "rule" }));
      groups.forEach(function (g, k) {
        var gx = 20 + k * (gw + 10);
        svg.appendChild(WK.h("text", { x: gx + gw / 2 - 10, y: by - 12, "font-size": 12, "font-weight": 600,
          "text-anchor": "middle", fill: "ink" }, g[0]));
        g[1].forEach(function (v, j) {
          var x = gx + j * (bw + 10);
          svg.appendChild(WK.h("rect", { x: x, y: sy(v), width: bw, height: by + bh - sy(v), fill: colours[j],
            "fill-opacity": j === 2 ? 0.5 : 0.85 }));
          svg.appendChild(WK.h("text", { x: x + bw / 2, y: sy(v) - 4, "font-size": 12, "text-anchor": "middle",
            fill: "ink" }, v.toFixed(2)));
        });
      });
      // Legend: one swatch per bar colour.
      var lx0 = 70;
      names.forEach(function (nm, j) {
        var lx = lx0 + j * 140;
        svg.appendChild(WK.h("rect", { x: lx, y: H - 30, width: 14, height: 14, fill: colours[j],
          "fill-opacity": j === 2 ? 0.5 : 0.85 }));
        svg.appendChild(WK.h("text", { x: lx + 20, y: H - 18, "font-size": 13, fill: "ink" },
          j === 2 ? "invalid strings" : "\u201c" + nm + "\u201d"));
      });
      stLocal.set(r.argmaxLocal);
      stCond.set(r.argmaxCond);
      stInv.set(r.strings.invalid.toFixed(2));
    }
    draw();
  });
}
