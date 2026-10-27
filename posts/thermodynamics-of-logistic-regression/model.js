/* The same ten synthetic observations drive both scenes. No random draws. */
"use strict";

var ThermoModel = (function () {
  var samples = [
    [-1.55, -1.15, 0], [-1.2, 0.6, 0], [-0.75, 1.35, 0],
    [-0.35, -0.6, 0], [0.15, 1.55, 0], [0.55, -1.25, 1],
    [1.15, -0.6, 1], [1.65, 0.15, 1], [0.95, 1.35, 1],
    [-0.55, -1.65, 1]
  ];
  var probe = 5;

  function score(x1, x2, p) { return p.w1 * x1 + p.w2 * x2 + p.b; }
  function sigmoid(t) { return t >= 0 ? 1 / (1 + Math.exp(-t)) : Math.exp(t) / (1 + Math.exp(t)); }
  function softplus(t) { return Math.max(0, t) + Math.log1p(Math.exp(-Math.abs(t))); }
  function loss(p, flip) {
    var total = 0;
    samples.forEach(function (row, i) {
      var y = flip && i === probe ? 1 - row[2] : row[2];
      var eta = score(row[0], row[1], p);
      total += softplus(eta) - y * eta;
    });
    return total / samples.length;
  }
  function probeStats(p, beta) {
    var row = samples[probe], eta = score(row[0], row[1], p);
    var probability = sigmoid(beta * eta);
    return { eta: eta, probability: probability, surprisal: -Math.log(probability) };
  }
  return { samples: samples, probe: probe, score: score, sigmoid: sigmoid,
    softplus: softplus, loss: loss, probeStats: probeStats };
})();

if (typeof module !== "undefined") module.exports = ThermoModel;
