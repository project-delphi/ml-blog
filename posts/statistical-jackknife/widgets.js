/* Widgets for "The Statistical Jackknife"
 *
 * Dependency-free: no CDN, no module loading, no framework.
 * Built with DOM APIs and inline SVG for seamless local and offline viewing.
 */
"use strict";

const JK = (function () {
  const NS = "http://www.w3.org/2000/svg";
  const C = {
    primary: "#1D5C6E",
    secondary: "#C98A12",
    accent: "#7A3B6B",
    danger: "#C62828",
    success: "#2E7D32",
    ink: "#171C1B",
    muted: "#67706E",
    rule: "#D2D6D1",
    bgLight: "#F8F9FA",
    white: "#FFFFFF"
  };

  // DOM helpers
  function el(tag, attrs, kids) {
    const e = document.createElementNS(NS, tag);
    for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
    for (const c of (kids || [])) e.appendChild(c);
    return e;
  }
  function h(tag, attrs, kids) {
    const e = document.createElement(tag);
    for (const k in (attrs || {})) {
      if (k === "html") e.innerHTML = attrs[k];
      else if (k === "text") e.textContent = attrs[k];
      else if (k === "style" && typeof attrs[k] === "object") {
        Object.assign(e.style, attrs[k]);
      } else e.setAttribute(k, attrs[k]);
    }
    for (const c of (kids || [])) e.appendChild(c);
    return e;
  }
  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // --- WIDGET 1: Pseudovalues & Influence Playground ---
  function initWidget1(containerId, initialData) {
    const root = document.getElementById(containerId);
    if (!root) return;
    clear(root);

    let points = (initialData && initialData.points) 
      ? JSON.parse(JSON.stringify(initialData.points))
      : [
          {id: 1, x: 2, y: 3}, {id: 2, x: 3, y: 4}, {id: 3, x: 4, y: 5},
          {id: 4, x: 5, y: 6}, {id: 5, x: 6, y: 7}, {id: 6, x: 7, y: 8},
          {id: 7, x: 8, y: 9}, {id: 8, x: 5, y: 9.5}
        ];

    const width = 680;
    const height = 340;
    const pad = 40;

    // Controls & Summary Header
    const statPanel = h("div", {
      style: {
        display: "flex",
        flexWrap: "wrap",
        gap: "1.2rem",
        marginBottom: "0.8rem",
        fontSize: "0.9rem",
        background: "#F4F6F6",
        padding: "0.6rem 1rem",
        borderRadius: "6px",
        border: `1px solid ${C.rule}`
      }
    });

    const svgWrap = h("div", { style: { display: "flex", gap: "1rem", flexWrap: "wrap" } });
    
    // Left: 2D scatter svg
    const scatterSvg = el("svg", {
      viewBox: `0 0 380 300`,
      width: "380",
      height: "300",
      style: "background:#fff;border:1px solid #E2E6E1;border-radius:6px;cursor:grab;"
           + "max-width:100%;height:auto"
    });

    // Right: Influence bar chart svg
    const barSvg = el("svg", {
      viewBox: `0 0 280 300`,
      width: "280",
      height: "300",
      style: "background:#fff;border:1px solid #E2E6E1;border-radius:6px;"
           + "max-width:100%;height:auto"
    });

    svgWrap.appendChild(scatterSvg);
    svgWrap.appendChild(barSvg);

    const note = h("div", {
      class: "widget-note",
      text: "Drag any point on the left. Watch its influence bar (right) and the Jackknife standard error recalculate instantly."
    });

    root.appendChild(statPanel);
    root.appendChild(svgWrap);
    root.appendChild(note);

    let activeDrag = null;

    function toScreenX(x) { return pad + (x / 10) * (380 - 2 * pad); }
    function toScreenY(y) { return (300 - pad) - (y / 10) * (300 - 2 * pad); }
    function toDataX(sx) { return Math.max(0.5, Math.min(9.5, ((sx - pad) / (380 - 2 * pad)) * 10)); }
    function toDataY(sy) { return Math.max(0.5, Math.min(9.5, (((300 - pad) - sy) / (300 - 2 * pad)) * 10)); }

    function update() {
      const n = points.length;
      // Estimator: Correlation r (or mean y)
      // Let's use correlation r as a classic non-linear statistic
      const sumX = points.reduce((s, p) => s + p.x, 0);
      const sumY = points.reduce((s, p) => s + p.y, 0);
      const meanX = sumX / n;
      const meanY = sumY / n;

      function calcSlope(pts) {
        const mX = pts.reduce((s, p) => s + p.x, 0) / pts.length;
        const mY = pts.reduce((s, p) => s + p.y, 0) / pts.length;
        let num = 0, den = 0;
        for (const p of pts) {
          num += (p.x - mX) * (p.y - mY);
          den += (p.x - mX) * (p.x - mX);
        }
        return den === 0 ? 0 : num / den;
      }

      const thetaHat = calcSlope(points);

      // Leave-one-out
      const theta_i = [];
      const pseudovalues = [];
      for (let i = 0; i < n; i++) {
        const sub = points.filter((_, idx) => idx !== i);
        const t_i = calcSlope(sub);
        theta_i.push(t_i);
        const J_i = n * thetaHat - (n - 1) * t_i;
        pseudovalues.push(J_i);
      }

      const thetaDot = theta_i.reduce((s, v) => s + v, 0) / n;
      const thetaJack = n * thetaHat - (n - 1) * thetaDot;
      const biasEst = (n - 1) * (thetaDot - thetaHat);

      // Jackknife variance
      let sumSqDiff = 0;
      for (let i = 0; i < n; i++) {
        sumSqDiff += Math.pow(theta_i[i] - thetaDot, 2);
      }
      const vJack = ((n - 1) / n) * sumSqDiff;
      const seJack = Math.sqrt(vJack);

      // Influence functions
      const influences = pseudovalues.map(j => j - thetaJack);

      // Update statPanel
      clear(statPanel);
      statPanel.appendChild(h("div", {
        html: `<strong>Sample slope:</strong> <span class="metric-pill">${thetaHat.toFixed(3)}</span>`
      }));
      statPanel.appendChild(h("div", {
        html: `<strong>Jackknife-corrected slope:</strong> <span class="metric-pill positive">${thetaJack.toFixed(3)}</span>`
      }));
      statPanel.appendChild(h("div", {
        html: `<strong>Est. Bias:</strong> <span class="metric-pill ${Math.abs(biasEst) > 0.05 ? 'warning' : ''}">${biasEst.toFixed(3)}</span>`
      }));
      statPanel.appendChild(h("div", {
        html: `<strong>Jackknife SE:</strong> <span class="metric-pill">${seJack.toFixed(3)}</span>`
      }));

      // Render Scatter SVG
      clear(scatterSvg);
      // Grid lines
      for (let g = 2; g <= 8; g += 2) {
        scatterSvg.appendChild(el("line", {
          x1: toScreenX(g), y1: pad, x2: toScreenX(g), y2: 300 - pad,
          stroke: "#EEF0F0", "stroke-width": "1"
        }));
        scatterSvg.appendChild(el("line", {
          x1: pad, y1: toScreenY(g), x2: 380 - pad, y2: toScreenY(g),
          stroke: "#EEF0F0", "stroke-width": "1"
        }));
      }

      // Regression line
      const lineX1 = 0.5, lineX2 = 9.5;
      const lineY1 = meanY + thetaHat * (lineX1 - meanX);
      const lineY2 = meanY + thetaHat * (lineX2 - meanX);
      scatterSvg.appendChild(el("line", {
        x1: toScreenX(lineX1), y1: toScreenY(lineY1),
        x2: toScreenX(lineX2), y2: toScreenY(lineY2),
        stroke: C.primary, "stroke-width": "2", "stroke-dasharray": "4 2"
      }));

      // Points
      points.forEach((p, idx) => {
        const cx = toScreenX(p.x);
        const cy = toScreenY(p.y);
        const infl = Math.abs(influences[idx]);
        const isHigh = infl > 0.3;

        const circle = el("circle", {
          cx, cy, r: isHigh ? "8" : "6",
          fill: isHigh ? C.danger : C.primary,
          stroke: "#fff",
          "stroke-width": "2",
          style: "cursor:grab"
        });

        circle.addEventListener("mousedown", (e) => {
          activeDrag = idx;
          e.preventDefault();
        });
        circle.addEventListener("touchstart", (e) => {
          activeDrag = idx;
          e.preventDefault();
        }, { passive: false });

        scatterSvg.appendChild(circle);

        // Point label
        scatterSvg.appendChild(el("text", {
          x: cx + 8, y: cy - 6,
          fill: C.muted, "font-size": "10", "font-family": "monospace"
        }, [document.createTextNode(`#${p.id}`)]));
      });

      // Render Bar SVG (Influence)
      clear(barSvg);
      barSvg.appendChild(el("text", {
        x: 140, y: 22, "text-anchor": "middle",
        fill: C.ink, "font-size": "12", "font-weight": "bold"
      }, [document.createTextNode("Empirical Influence (Jᵢ - β̂_jack)")]));

      const midX = 140;
      barSvg.appendChild(el("line", {
        x1: midX, y1: 35, x2: midX, y2: 280,
        stroke: C.rule, "stroke-width": "1.5"
      }));

      const maxAbsInfl = Math.max(0.6, ...influences.map(Math.abs));
      const rowHeight = (240) / n;

      influences.forEach((inf, idx) => {
        const y = 45 + idx * rowHeight;
        const barW = (inf / maxAbsInfl) * 110;
        const isPos = inf >= 0;

        const bar = el("rect", {
          x: isPos ? midX : midX + barW,
          y: y - 6,
          width: Math.abs(barW),
          height: Math.max(4, rowHeight - 4),
          fill: Math.abs(inf) > 0.3 ? C.danger : C.primary,
          rx: "2"
        });
        barSvg.appendChild(bar);

        barSvg.appendChild(el("text", {
          x: isPos ? midX - 8 : midX + 8,
          y: y + 4,
          "text-anchor": isPos ? "end" : "start",
          fill: C.muted, "font-size": "10", "font-family": "monospace"
        }, [document.createTextNode(`#${points[idx].id}`)]));
      });
    }

    function handleMove(clientX, clientY) {
      if (activeDrag === null) return;
      // The SVG is CSS-scalable (max-width:100%), so client pixels are not viewBox
      // units on a narrow screen. Rescale by the rendered size before converting.
      const rect = scatterSvg.getBoundingClientRect();
      const kx = rect.width ? 380 / rect.width : 1;
      const ky = rect.height ? 300 / rect.height : 1;
      const sx = (clientX - rect.left) * kx;
      const sy = (clientY - rect.top) * ky;
      points[activeDrag].x = toDataX(sx);
      points[activeDrag].y = toDataY(sy);
      update();
    }

    window.addEventListener("mousemove", (e) => handleMove(e.clientX, e.clientY));
    window.addEventListener("touchmove", (e) => {
      if (activeDrag !== null && e.touches.length > 0) {
        handleMove(e.touches[0].clientX, e.touches[0].clientY);
      }
    }, { passive: false });

    window.addEventListener("mouseup", () => { activeDrag = null; });
    window.addEventListener("touchend", () => { activeDrag = null; });

    update();
  }

  // --- WIDGET 2: Referral Syndicate Unmasker ---
  function initWidget2(containerId, initialData) {
    const root = document.getElementById(containerId);
    if (!root) return;
    clear(root);

    const accounts = (initialData && initialData.accounts) 
      ? JSON.parse(JSON.stringify(initialData.accounts))
      : [];

    let bonus = 20;
    let botSpokes = 8;

    // Controls
    const ctrlWrap = h("div", { class: "widget-controls" });

    const bonusSlider = h("input", {
      type: "range", min: "10", max: "50", step: "5", value: String(bonus),
      style: { width: "130px" }
    });
    const bonusVal = h("span", { text: `$${bonus}`, style: { fontWeight: "bold" } });
    bonusSlider.addEventListener("input", (e) => {
      bonus = parseInt(e.target.value);
      bonusVal.textContent = `$${bonus}`;
      render();
    });

    const botSlider = h("input", {
      type: "range", min: "0", max: "15", step: "1", value: String(botSpokes),
      style: { width: "130px" }
    });
    const botVal = h("span", { text: `${botSpokes} bots`, style: { fontWeight: "bold" } });
    botSlider.addEventListener("input", (e) => {
      botSpokes = parseInt(e.target.value);
      botVal.textContent = `${botSpokes} bots`;
      render();
    });

    ctrlWrap.appendChild(h("label", { style: { display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.88rem" } }, [
      h("span", { text: "Promo Credit:" }), bonusSlider, bonusVal
    ]));
    ctrlWrap.appendChild(h("label", { style: { display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.88rem" } }, [
      h("span", { text: "Syndicate Bot Nodes:" }), botSlider, botVal
    ]));

    const cardContainer = h("div", {
      style: {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
        gap: "1rem",
        marginTop: "1rem"
      }
    });

    const statusBanner = h("div", { style: { marginTop: "1rem" } });

    root.appendChild(ctrlWrap);
    root.appendChild(cardContainer);
    root.appendChild(statusBanner);

    function render() {
      // Build dynamic user list
      const activeUsers = accounts.filter(a => a.type === "legit" || a.type === "syndicate_hub");
      for (let b = 1; b <= botSpokes; b++) {
        activeUsers.push({
          id: `BOT-${b}`,
          type: "syndicate_spoke",
          name: `Synthetic Node ${b}`,
          promo_cost: bonus,
          net_spend: 0,
          referrals: 0
        });
      }
      // Charge each account the credits it caused: its own signup bonus plus one
      // bonus per referee it recruited. The hub's referral count tracks the bot slider,
      // so its cost is the whole syndicate's payout, not a single $20 credit.
      const priced = activeUsers.map(u => {
        const referrals = u.type === "syndicate_hub" ? botSpokes : (u.referrals || 0);
        return { ...u, referrals, promo_cost: bonus * (1 + referrals) };
      });

      const n = priced.length;
      const totalPromo = priced.reduce((s, u) => s + u.promo_cost, 0);
      const totalSpend = priced.reduce((s, u) => s + u.net_spend, 0);
      const clusterRatio = totalSpend > 0 ? (totalPromo / totalSpend) : 999;

      // Leave-one-out
      const results = priced.map((u, idx) => {
        const sub = priced.filter((_, i) => i !== idx);
        const subP = sub.reduce((s, x) => s + x.promo_cost, 0);
        const subS = sub.reduce((s, x) => s + x.net_spend, 0);
        const subR = subS > 0 ? (subP / subS) : 999;
        const pseudovalue = n * clusterRatio - (n - 1) * subR;
        const leverage = clusterRatio - subR; // How much health drops when this user is PRESENT
        return { ...u, subR, pseudovalue, leverage };
      });

      // Sort by absolute leverage / pseudovalue anomaly
      results.sort((a, b) => b.leverage - a.leverage);

      clear(cardContainer);

      // Summary Card
      const summaryCard = h("div", {
        style: {
          background: "#FFFFFF",
          border: `1px solid ${C.rule}`,
          borderRadius: "8px",
          padding: "1rem"
        }
      }, [
        h("h4", { text: "Cohort Unit Economics", style: { margin: "0 0 0.5rem 0", color: C.ink } }),
        h("div", { html: `<strong>Cohort Size:</strong> ${n} users (${botSpokes} syndicate bots)` }),
        h("div", { html: `<strong>Total Promo Burn:</strong> $${totalPromo}` }),
        h("div", { html: `<strong>Total Net Revenue:</strong> $${totalSpend}` }),
        h("div", {
          style: { marginTop: "0.5rem", fontSize: "1.1rem" },
          html: `<strong>Promo-to-Spend Ratio:</strong> <span class="metric-pill ${clusterRatio > 0.5 ? 'danger' : 'positive'}">${clusterRatio.toFixed(3)}</span>`
        })
      ]);
      cardContainer.appendChild(summaryCard);

      // Top Anomaly Table Card
      const topAnomalies = results.slice(0, 5);
      const anomalyCard = h("div", {
        style: {
          background: "#FFFFFF",
          border: `1px solid ${C.rule}`,
          borderRadius: "8px",
          padding: "1rem"
        }
      }, [
        h("h4", { text: "Top Jackknife Leverage (Who Breaks the Metric?)", style: { margin: "0 0 0.5rem 0", color: C.ink } }),
        h("table", { style: { width: "100%", fontSize: "0.85rem", borderCollapse: "collapse" } }, [
          h("thead", {}, [
            h("tr", { style: { borderBottom: `1px solid ${C.rule}`, textAlign: "left", color: C.muted } }, [
              h("th", { text: "User" }),
              h("th", { text: "Spend" }),
              h("th", { text: "Ratio without them" }),
              h("th", { text: "Leverage" }),
              h("th", { text: "Pseudovalue" })
            ])
          ]),
          h("tbody", {}, topAnomalies.map(u => {
            const isRingleader = u.type === "syndicate_hub";
            return h("tr", {
              style: {
                borderBottom: "1px solid #F0F2F2",
                background: isRingleader ? "#FFF5F5" : "transparent",
                fontWeight: isRingleader ? "bold" : "normal",
                color: isRingleader ? C.danger : C.ink
              }
            }, [
              h("td", { text: u.name }),
              h("td", { text: `$${u.net_spend}` }),
              h("td", { text: u.subR.toFixed(3) }),
              h("td", { text: `+${u.leverage.toFixed(3)}` }),
              h("td", { text: u.pseudovalue.toFixed(2) })
            ]);
          }))
        ])
      ]);
      cardContainer.appendChild(anomalyCard);

      // Status alert
      clear(statusBanner);
      const ringleader = results.find(r => r.type === "syndicate_hub");
      const runnerUp = results.find(r => r.type !== "syndicate_hub");
      if (ringleader) {
        statusBanner.appendChild(h("div", {
          class: "jackknife-hero",
          style: { borderColor: C.danger, background: "rgba(198, 40, 40, 0.05)" },
          html: `<strong>🎯 Jackknife Isolation:</strong> Removing <em>${ringleader.name}</em> (${botSpokes} referrals, $${ringleader.net_spend} spend) drops the cohort Promo-to-Spend ratio from <strong>${clusterRatio.toFixed(3)}</strong> to <strong>${ringleader.subR.toFixed(3)}</strong> — leverage <strong>${ringleader.leverage.toFixed(3)}</strong>, <strong>${(ringleader.leverage / runnerUp.leverage).toFixed(1)}×</strong> the next account. ${botSpokes > 0 ? "The jackknife isolates the coordinator without any network-graph clustering." : "With no spokes to coordinate, the hub is just a low-spending customer and its lead all but vanishes."}`
        }));
      }
    }

    render();
  }

  // --- WIDGET 3: Biomarker Stability Stress-Tester ---
  function initWidget3(containerId, initialData) {
    const root = document.getElementById(containerId);
    if (!root) return;
    clear(root);

    const genes = (initialData && initialData.genes) 
      ? JSON.parse(JSON.stringify(initialData.genes))
      : [];

    let injectOutlier = false;

    const ctrlWrap = h("div", { class: "widget-controls", style: { alignItems: "center" } });
    const toggleBtn = h("button", {
      text: "⚡ Inject artifact patients into LINC009 and FAM83A",
      style: {
        background: C.primary,
        color: "#FFF",
        border: "none",
        padding: "0.5rem 1rem",
        borderRadius: "4px",
        cursor: "pointer",
        fontWeight: "600"
      }
    });
    toggleBtn.addEventListener("click", () => {
      injectOutlier = !injectOutlier;
      toggleBtn.textContent = injectOutlier 
        ? "🔄 Remove artifact patients (restore baseline)"
        : "⚡ Inject artifact patients into LINC009 and FAM83A";
      toggleBtn.style.background = injectOutlier ? C.danger : C.primary;
      render();
    });

    ctrlWrap.appendChild(toggleBtn);
    ctrlWrap.appendChild(h("span", {
      style: { fontSize: "0.85rem", color: C.muted },
      text: "Watch naive fold change promote both artifacts to the top, while the stability index leaves the genuine targets in place."
    }));

    const displayGrid = h("div", {
      style: {
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "1rem",
        marginTop: "1rem"
      }
    });

    root.appendChild(ctrlWrap);
    root.appendChild(displayGrid);

    function simulateGene(gene, n = 30) {
      // Create N patient expression differences
      let diffs = [];
      for (let i = 0; i < n; i++) {
        // Deterministic stand-in for patient-to-patient noise: a fixed sinusoid, not a
        // random draw, so the widget renders identically on every load.
        const base = (gene.tumor_mean - gene.normal_mean) + (Math.sin(i * 3 + gene.name.length) * gene.std);
        diffs.push(base);
      }
      // A gene declaring an outlier patient in the data file gets that patient's
      // reading replaced when the artifact is injected. Genes without one are untouched.
      let injected = false;
      if (injectOutlier && gene.outlier_patient !== undefined && gene.outlier_patient < n) {
        diffs[gene.outlier_patient] = gene.outlier_val;
        injected = true;
      }

      const meanLFC = diffs.reduce((s, v) => s + v, 0) / n;

      // Jackknife
      const subMeans = diffs.map((_, idx) => {
        const sub = diffs.filter((_, i) => i !== idx);
        return sub.reduce((s, v) => s + v, 0) / (n - 1);
      });
      const thetaDot = subMeans.reduce((s, v) => s + v, 0) / n;
      const jackLFC = n * meanLFC - (n - 1) * thetaDot;
      const vJack = ((n - 1) / n) * subMeans.reduce((s, m) => s + Math.pow(m - thetaDot, 2), 0);
      const seJack = Math.sqrt(vJack);
      const stability = seJack > 0 ? Math.abs(jackLFC) / seJack : 0;

      return {
        name: gene.name,
        type: gene.type,
        injected,
        meanLFC,
        jackLFC,
        seJack,
        stability
      };
    }

    function render() {
      const results = genes.map(g => simulateGene(g, 30));

      // Sort by naive |LFC| so down-regulated targets (BRCA1, TP53) compete on the
      // same scale as up-regulated ones, matching how the stability panel ranks.
      const naiveSorted = [...results].sort((a, b) => Math.abs(b.meanLFC) - Math.abs(a.meanLFC));
      // Sort by Jackknife Stability
      const jackSorted = [...results].sort((a, b) => b.stability - a.stability);

      clear(displayGrid);

      // Left Column: Naive Ranking
      const leftCol = h("div", {
        style: { background: "#FFFFFF", border: `1px solid ${C.rule}`, borderRadius: "8px", padding: "1rem" }
      }, [
        h("h4", { text: "Naive Ranking (Mean Log-Fold Change)", style: { margin: "0 0 0.5rem 0", color: C.ink } }),
        h("div", { style: { fontSize: "0.8rem", color: C.muted, marginBottom: "0.5rem" }, text: "Ranks genes solely by cohort sample average." }),
        h("table", { style: { width: "100%", fontSize: "0.85rem", borderCollapse: "collapse" } }, [
          h("thead", {}, [
            h("tr", { style: { borderBottom: `1px solid ${C.rule}`, textAlign: "left", color: C.muted } }, [
              h("th", { text: "Rank" }),
              h("th", { text: "Gene" }),
              h("th", { text: "Mean LFC" })
            ])
          ]),
          h("tbody", {}, naiveSorted.slice(0, 5).map((g, i) => {
            const isLinc = g.injected;
            return h("tr", {
              style: {
                borderBottom: "1px solid #F0F2F2",
                background: isLinc && injectOutlier ? "#FFF5F5" : "transparent",
                fontWeight: isLinc && injectOutlier ? "bold" : "normal",
                color: isLinc && injectOutlier ? C.danger : C.ink
              }
            }, [
              h("td", { text: `#${i + 1}` }),
              h("td", { text: `${g.name} ${isLinc && injectOutlier ? '⚠️' : ''}` }),
              h("td", { text: g.meanLFC.toFixed(2) })
            ]);
          }))
        ])
      ]);

      // Right Column: Jackknife Stability Ranking
      const rightCol = h("div", {
        style: { background: "#FFFFFF", border: `1px solid ${C.rule}`, borderRadius: "8px", padding: "1rem" }
      }, [
        h("h4", { text: "Jackknife Stability Ranking (|LFC| / SE_jack)", style: { margin: "0 0 0.5rem 0", color: C.primary } }),
        h("div", { style: { fontSize: "0.8rem", color: C.muted, marginBottom: "0.5rem" }, text: "Penalizes genes with high leave-one-out variance." }),
        h("table", { style: { width: "100%", fontSize: "0.85rem", borderCollapse: "collapse" } }, [
          h("thead", {}, [
            h("tr", { style: { borderBottom: `1px solid ${C.rule}`, textAlign: "left", color: C.muted } }, [
              h("th", { text: "Rank" }),
              h("th", { text: "Gene" }),
              h("th", { text: "SE_jack" }),
              h("th", { text: "Stability" })
            ])
          ]),
          h("tbody", {}, jackSorted.slice(0, 5).map((g, i) => {
            const isLinc = g.injected;
            return h("tr", {
              style: {
                borderBottom: "1px solid #F0F2F2",
                background: isLinc && injectOutlier ? "#FFF5F5" : "transparent",
                color: isLinc && injectOutlier ? C.danger : C.ink
              }
            }, [
              h("td", { text: `#${i + 1}` }),
              h("td", { text: g.name }),
              h("td", { text: g.seJack.toFixed(2) }),
              h("td", { text: g.stability.toFixed(2), style: { fontWeight: "bold" } })
            ]);
          }))
        ])
      ]);

      displayGrid.appendChild(leftCol);
      displayGrid.appendChild(rightCol);
    }

    render();
  }

  // --- WIDGET 4: Smooth vs. Non-Smooth Stress Test ---
  function initWidget4(containerId) {
    const root = document.getElementById(containerId);
    if (!root) return;
    clear(root);

    let statType = "mean";
    const sample = [1.2, 1.8, 2.3, 2.9, 3.4, 3.8, 4.2, 4.9, 5.5, 6.2, 9.8];

    const ctrlWrap = h("div", { class: "widget-controls" });
    const radioMean = h("input", { type: "radio", id: "w4-opt-mean", name: "w4-stat", value: "mean", checked: true });
    const radioMed = h("input", { type: "radio", id: "w4-opt-med", name: "w4-stat", value: "median" });

    radioMean.addEventListener("change", () => { statType = "mean"; render(); });
    radioMed.addEventListener("change", () => { statType = "median"; render(); });

    ctrlWrap.appendChild(h("label", { style: { display: "inline-flex", alignItems: "center", gap: "0.3rem", fontSize: "0.9rem", cursor: "pointer" } }, [
      radioMean, h("span", { text: "Smooth: Sample Mean" })
    ]));
    ctrlWrap.appendChild(h("label", { style: { display: "inline-flex", alignItems: "center", gap: "0.3rem", fontSize: "0.9rem", cursor: "pointer" } }, [
      radioMed, h("span", { text: "Non-Smooth: Sample Median" })
    ]));

    const canvasWrap = h("div", { style: { background: "#FFF", border: `1px solid ${C.rule}`, borderRadius: "8px", padding: "1rem", marginTop: "1rem" } });

    root.appendChild(ctrlWrap);
    root.appendChild(canvasWrap);

    function calcMedian(arr) {
      const s = [...arr].sort((a, b) => a - b);
      const m = Math.floor(s.length / 2);
      return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
    }
    function calcMean(arr) {
      return arr.reduce((s, v) => s + v, 0) / arr.length;
    }

    function render() {
      clear(canvasWrap);
      const n = sample.length;
      const fullEst = statType === "mean" ? calcMean(sample) : calcMedian(sample);
      const subEsts = sample.map((_, idx) => {
        const sub = sample.filter((_, i) => i !== idx);
        return statType === "mean" ? calcMean(sub) : calcMedian(sub);
      });

      const deltas = subEsts.map(sub => fullEst - sub);

      const svgEl = el("svg", {
        viewBox: "0 0 600 200",
        width: "100%",
        height: "200",
        style: "display:block"
      });

      // Axis
      svgEl.appendChild(el("line", {
        x1: "50", y1: "150", x2: "550", y2: "150",
        stroke: C.rule, "stroke-width": "1.5"
      }));

      // Plot stem lines for leave-one-out delta
      const maxDelta = Math.max(0.2, ...deltas.map(Math.abs));
      deltas.forEach((d, idx) => {
        const x = 70 + idx * 44;
        const barH = (d / maxDelta) * 80;
        const y = 150 - barH;

        svgEl.appendChild(el("line", {
          x1: x, y1: "150", x2: x, y2: y,
          stroke: statType === "mean" ? C.primary : C.danger,
          "stroke-width": "3"
        }));
        svgEl.appendChild(el("circle", {
          cx: x, cy: y, r: "5",
          fill: statType === "mean" ? C.primary : C.danger
        }));

        svgEl.appendChild(el("text", {
          x, y: "170", "text-anchor": "middle",
          fill: C.muted, "font-size": "10", "font-family": "monospace"
        }, [document.createTextNode(`-${idx + 1}`)]));
      });

      canvasWrap.appendChild(svgEl);

      const expl = h("div", {
        style: { marginTop: "0.8rem", fontSize: "0.88rem", color: C.ink },
        html: statType === "mean"
          ? `<strong>Smooth Response:</strong> When leaving out observations from the mean, each point creates a gentle, continuous shift proportional to its distance from the sample mean. The influence function is differentiable.`
          : `<strong>Discontinuous Jumps:</strong> For the median, every point outside the centre moves the estimate by exactly the same <strong>\u00b10.20</strong> \u2014 the gap between two neighbouring central values \u2014 no matter how extreme it is, while dropping the central point itself changes nothing at all (<strong>0.00</strong>). The leave-one-out spread has stopped measuring influence, so the jackknife variance formula fails: the sample median lacks Fr\u00e9chet differentiability.`
      });
      canvasWrap.appendChild(expl);
    }

    render();
  }

  return {
    initWidget1,
    initWidget2,
    initWidget3,
    initWidget4
  };
})();

// Auto-initialize when DOM and JSON script data are ready
document.addEventListener("DOMContentLoaded", () => {
  const dataScript = document.getElementById("jackknife-data");
  let data = {};
  if (dataScript) {
    try {
      data = JSON.parse(dataScript.textContent);
    } catch (e) {
      console.error("Error parsing jackknife data payload", e);
    }
  }

  if (document.getElementById("widget-pseudovalues")) {
    JK.initWidget1("widget-pseudovalues", data.toy);
  }
  if (document.getElementById("widget-syndicate")) {
    JK.initWidget2("widget-syndicate", data.syndicate);
  }
  if (document.getElementById("widget-biomarkers")) {
    JK.initWidget3("widget-biomarkers", data.biomarkers);
  }
  if (document.getElementById("widget-smooth-jagged")) {
    JK.initWidget4("widget-smooth-jagged");
  }
});
