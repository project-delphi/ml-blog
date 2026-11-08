/* The numbers behind the two widgets, apart from the drawing.
 *
 * `data` is window.WK_DATA (widget-data/data.js): `summary` holds one row per
 * trained configuration (3DShapes, uniform content law, Gaussian target) and
 * `lam` the lambda sweep. Nothing is interpolated: a configuration that was not
 * trained returns no row. Also exports under node so src/check_claims.js can
 * test every sentence the post aims at the reader.
 */
"use strict";
(function (root) {
  var ARM_ORDER = ["VICReg", "SIGReg", "sliced KSD", "KSD", "MMD", "EMA", "oracle", "none"];
  var LAM_ARMS = ["VICReg", "SIGReg", "sliced KSD", "KSD", "MMD", "oracle"];
  // Theme tokens, not hex: the kit repaints them with the light/dark toggle.
  var ARM_TOKEN = { "VICReg": "c1", "SIGReg": "c2", "sliced KSD": "c3", "KSD": "c4",
                    "MMD": "c5", "EMA": "ink", "oracle": "c6", "none": "muted" };

  /** Rows trained at (d, sigma, alpha), in ARM_ORDER, limited to `arms`. */
  function cell(data, d, sigma, alpha) {
    return data.summary.filter(function (r) {
      return r.d === d && r.sigma === sigma && r.alpha === alpha;
    }).sort(function (a, b) { return ARM_ORDER.indexOf(a.arm) - ARM_ORDER.indexOf(b.arm); });
  }

  /** Lambda sweep rows for one arm, lowest lambda first. */
  function lamSeries(data, arm) {
    return data.lam.filter(function (r) { return r.arm === arm; })
      .sort(function (a, b) { return a.lam_rel - b.lam_rel; });
  }

  var api = { ARM_ORDER: ARM_ORDER, LAM_ARMS: LAM_ARMS, ARM_TOKEN: ARM_TOKEN,
              cell: cell, lamSeries: lamSeries };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SteinModel = api;
})(typeof window !== "undefined" ? window : this);
