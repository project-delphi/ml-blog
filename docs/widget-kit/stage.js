/* widget-kit/stage.js -- a small three.js stage for the blog's 3D widgets.
 *
 * Loaded by the posts that draw in 3D, after kit.js and before their own
 * widgets.js. Exposes one global, `WKStage`. It owns the parts every 3D widget
 * would otherwise repeat: fetching three.js once and only when a scene scrolls
 * near the viewport, the renderer and its resize, a drag-to-orbit camera, HTML
 * labels pinned to 3D points, and materials that follow the theme tokens.
 *
 * Coordinates are the ones the maths is written in: right-handed with z up.
 * `V` maps them onto three.js's y-up frame, and every helper here takes
 * `[x, y, z]` arrays in maths coordinates.
 *
 * A scene renders on demand. Call `stage.render()` after changing anything;
 * `stage.animate` drives a tween and renders each frame of it. There is no
 * idle loop, so four scenes on a page cost nothing while the reader reads.
 *
 * three.js is the one runtime dependency the kit takes from a CDN (pinned
 * below). When it does not load -- offline, a blocked CDN, no WebGL -- the
 * stage shows a sentence saying so and the build callback never runs, so a
 * widget must keep its numbers (stat tiles, readouts) independent of the scene.
 */
"use strict";

var WKStage = (function () {
  var THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.min.js";
  var pending = null;
  function loadThree() {
    if (!pending) pending = import(THREE_URL);
    return pending;
  }

  var CSS =
    ".wk-stage{position:relative;width:100%;aspect-ratio:16/10;max-height:540px;min-height:260px;" +
    "border:1px solid var(--w-rule);border-radius:8px;background:var(--w-paper);overflow:hidden}" +
    ".wk-stage canvas{display:block;width:100%;height:100%;touch-action:pan-y;cursor:grab;outline-offset:-3px}" +
    ".wk-stage canvas:active{cursor:grabbing}" +
    ".wk-stage canvas:focus-visible{outline:3px solid var(--w-accent)}" +
    ".wk-stage-labels{position:absolute;inset:0;pointer-events:none;overflow:hidden}" +
    ".wk-stage-label{position:absolute;left:0;top:0;white-space:nowrap;font-size:0.8rem;font-weight:700;" +
    "line-height:1;padding:1px 3px;border-radius:3px;background:color-mix(in srgb,var(--w-paper) 72%,transparent);" +
    "font-variant-numeric:tabular-nums;will-change:transform}" +
    ".wk-stage-msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;" +
    "padding:1rem;text-align:center;font-size:0.86rem;color:var(--w-muted)}" +
    ".wk-stage-hint{position:absolute;right:8px;bottom:6px;font-size:0.72rem;color:var(--w-muted);" +
    "pointer-events:none;transition:opacity 0.6s}" +
    "@media (max-width:640px){.wk-stage{aspect-ratio:4/3.4}.wk-stage-label{font-size:0.72rem}}";
  var styled = false;
  function style() {
    if (styled) return;
    styled = true;
    var s = document.createElement("style");
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  var RAD = Math.PI / 180;
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
  function reducedMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  /** create(host, {label, view: {az, el, dist, target, fov, elMin, elMax}}, build).
   *  Appends the stage box to `host` at once and calls `build(stage)` when
   *  three.js has loaded, which happens the first time the box nears the viewport. */
  function create(host, opts, build) {
    style();
    opts = opts || {};
    var box = WK.el("div", { class: "wk-stage" });
    var msg = WK.el("div", { class: "wk-stage-msg", text: "Loading the 3D scene…" });
    box.appendChild(msg);
    host.appendChild(box);
    var started = false;
    function start() {
      if (started) return;
      started = true;
      loadThree().then(function (T) {
        var stage;
        try { stage = init(T, box, opts); }
        catch (err) {
          msg.textContent = "This 3D scene needs WebGL, which this browser did not provide. " +
            "The numbers under it still update.";
          if (window.console) console.error("WKStage", err);
          return;
        }
        box.removeChild(msg);
        build(stage);
        stage.render();
      }, function () {
        msg.textContent = "This 3D scene draws with three.js, which did not load from cdn.jsdelivr.net. " +
          "The numbers under it still update.";
      });
    }
    if (window.IntersectionObserver) {
      var io = new IntersectionObserver(function (entries) {
        if (entries.some(function (e) { return e.isIntersecting; })) { io.disconnect(); start(); }
      }, { rootMargin: "600px" });
      io.observe(box);
    } else start();
    return box;
  }

  function init(T, box, opts) {
    var view = Object.assign({ az: 35, el: 25, dist: 8, target: [0, 0, 0], fov: 32, elMin: 4, elMax: 86 },
      opts.view || {});
    var home = { az: view.az, el: view.el, dist: view.dist };
    var renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.localClippingEnabled = true;
    var canvas = renderer.domElement;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("tabindex", "0");
    if (opts.label) canvas.setAttribute("aria-label", opts.label + " Drag, or use the arrow keys, to rotate.");
    var layer = WK.el("div", { class: "wk-stage-labels" });
    var hint = WK.el("div", { class: "wk-stage-hint", text: "drag to rotate" });
    box.appendChild(canvas);
    box.appendChild(layer);
    box.appendChild(hint);

    var scene = new T.Scene();
    var camera = new T.PerspectiveCamera(view.fov, 1.6, 0.05, 400);
    scene.add(new T.AmbientLight(0xffffff, 1.9));
    var sun = new T.DirectionalLight(0xffffff, 1.5);
    sun.position.set(3, 8, 5);
    scene.add(sun);

    function V(p) { return new T.Vector3(p[0], p[2], -p[1]); }

    // ---------------------------------------------------------- theme
    var themed = [], themeCallbacks = [];
    function colour(token) { return new T.Color(WK.palette()[token] || token); }
    /** A material whose colour is a theme token and follows the toggle. */
    function material(kind, token, params) {
      var C = { basic: T.MeshBasicMaterial, lambert: T.MeshLambertMaterial,
        line: T.LineBasicMaterial, dashed: T.LineDashedMaterial }[kind];
      var m = new C(Object.assign({ color: colour(token) }, params || {}));
      themed.push({ m: m, token: token });
      return m;
    }
    WK.onTheme(function () {
      themed.forEach(function (t) { t.m.color.copy(colour(t.token)); });
      themeCallbacks.forEach(function (cb) { cb(); });
      render();
    });

    // --------------------------------------------------------- camera
    function place() {
      var a = view.az * RAD, e = view.el * RAD, t = view.target;
      camera.position.copy(V([
        t[0] + view.dist * Math.cos(e) * Math.cos(a),
        t[1] + view.dist * Math.cos(e) * Math.sin(a),
        t[2] + view.dist * Math.sin(e)]));
      camera.lookAt(V(t));
    }
    function orbit(daz, del) {
      view.az += daz;
      view.el = clamp(view.el + del, view.elMin, view.elMax);
      hint.style.opacity = "0";
      render();
    }
    canvas.addEventListener("pointerdown", function (ev) {
      var x = ev.clientX, y = ev.clientY;
      canvas.setPointerCapture(ev.pointerId);
      function move(e) {
        orbit(-(e.clientX - x) * 0.4, (e.clientY - y) * 0.4);
        x = e.clientX; y = e.clientY;
      }
      function up() {
        canvas.removeEventListener("pointermove", move);
        canvas.removeEventListener("pointerup", up);
        canvas.removeEventListener("pointercancel", up);
      }
      canvas.addEventListener("pointermove", move);
      canvas.addEventListener("pointerup", up);
      canvas.addEventListener("pointercancel", up);
    });
    canvas.addEventListener("keydown", function (ev) {
      var step = { ArrowLeft: [6, 0], ArrowRight: [-6, 0], ArrowUp: [0, 4], ArrowDown: [0, -4] }[ev.key];
      if (!step) return;
      ev.preventDefault();
      orbit(step[0], step[1]);
    });
    canvas.addEventListener("dblclick", function () {
      view.az = home.az; view.el = home.el; view.dist = home.dist;
      render();
    });

    // --------------------------------------------------------- render
    var labels = [], queued = false, width = 0, height = 0;
    function resize() {
      var w = box.clientWidth, h = box.clientHeight;
      if (!w || !h || (w === width && h === height)) return;
      width = w; height = h;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    function frame() {
      queued = false;
      resize();
      place();
      renderer.render(scene, camera);
      labels.forEach(function (l) {
        if (!l.on) { l.node.style.display = "none"; return; }
        var p = V(l.at).project(camera);
        if (p.z > 1 || Math.abs(p.x) > 1.05 || Math.abs(p.y) > 1.05) { l.node.style.display = "none"; return; }
        l.node.style.display = "";
        l.node.style.transform = "translate(" + ((p.x * 0.5 + 0.5) * width + l.dx).toFixed(1) + "px," +
          ((-p.y * 0.5 + 0.5) * height + l.dy).toFixed(1) + "px) translate(-50%,-50%)";
      });
    }
    function render() {
      if (queued) return;
      queued = true;
      requestAnimationFrame(frame);
    }
    if (window.ResizeObserver) new ResizeObserver(render).observe(box);
    else window.addEventListener("resize", render);

    /** animate(ms, step): calls step(t) with an eased t from 0 to 1, rendering
     *  each frame. Returns a cancel function. Jumps to the end under
     *  prefers-reduced-motion. */
    function animate(ms, step) {
      var t0 = null, live = true;
      if (reducedMotion() || ms <= 0) { step(1); render(); return function () {}; }
      function tick(now) {
        if (!live) return;
        if (t0 === null) t0 = now;
        var t = clamp((now - t0) / ms, 0, 1);
        step(ease(t));
        render();
        if (t < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
      return function () { live = false; };
    }

    // -------------------------------------------------------- objects
    function add(obj) { obj.frustumCulled = false; scene.add(obj); return obj; }
    function shower(obj) { return function (on) { obj.visible = !!on; }; }

    /** clipZ(lo, hi): clipping planes that keep lo <= z <= hi, for `clip`. */
    function clipZ(lo, hi) {
      return [new T.Plane(new T.Vector3(0, 1, 0), -lo), new T.Plane(new T.Vector3(0, -1, 0), hi)];
    }
    /** clipBox([x0, x1], [y0, y1], [z0, z1]): keep the inside of a box. */
    function clipBox(x, y, z) {
      return clipZ(z[0], z[1]).concat([
        new T.Plane(new T.Vector3(1, 0, 0), -x[0]), new T.Plane(new T.Vector3(-1, 0, 0), x[1]),
        new T.Plane(new T.Vector3(0, 0, -1), -y[0]), new T.Plane(new T.Vector3(0, 0, 1), y[1])]);
    }

    /** An arrow from one point to another: arrow("c1", {radius}).set(from, to). */
    function arrow(token, o) {
      o = o || {};
      var r = o.radius || 0.022, headLen = r * 7, up = new T.Vector3(0, 1, 0);
      var mat = material("lambert", token, o.clip ? { clippingPlanes: o.clip } : {});
      var shaftGeo = new T.CylinderGeometry(r, r, 1, 14);
      shaftGeo.translate(0, 0.5, 0);
      var headGeo = new T.ConeGeometry(r * 2.8, headLen, 18);
      headGeo.translate(0, headLen / 2, 0);
      var g = new T.Group(), shaft = add(new T.Mesh(shaftGeo, mat)), head = add(new T.Mesh(headGeo, mat));
      g.add(shaft); g.add(head);
      scene.add(g);
      var on = true;
      return {
        set: function (from, to) {
          var a = V(from), d = V(to).sub(a), len = d.length();
          if (len < 1e-6) { g.visible = false; return; }
          g.visible = on;
          var k = Math.min(1, len * 0.5 / headLen);
          g.position.copy(a);
          g.quaternion.setFromUnitVectors(up, d.normalize());
          shaft.scale.y = len - headLen * k;
          head.position.y = len - headLen * k;
          head.scale.setScalar(k);
        },
        show: function (v) { on = !!v; g.visible = on; }
      };
    }

    /** A polyline: line("muted", {dashed, opacity}).set([[x,y,z], ...]). */
    function line(token, o) {
      o = o || {};
      var params = { transparent: o.opacity !== undefined, opacity: o.opacity === undefined ? 1 : o.opacity };
      if (o.clip) params.clippingPlanes = o.clip;
      if (o.dashed) { params.dashSize = o.dash || 0.08; params.gapSize = o.gap || 0.06; }
      var obj = add(new (o.pairs ? T.LineSegments : T.Line)(new T.BufferGeometry(),
        material(o.dashed ? "dashed" : "line", token, params)));
      return {
        set: function (pts) {
          obj.geometry.dispose();
          obj.geometry = new T.BufferGeometry().setFromPoints(pts.map(V));
          if (o.dashed) obj.computeLineDistances();
        },
        show: shower(obj)
      };
    }

    /** A flat four-cornered patch: quad("c1", {opacity}).set(a, b, c, d). */
    function quad(token, o) {
      o = o || {};
      var params = { side: T.DoubleSide, transparent: true, opacity: o.opacity === undefined ? 0.25 : o.opacity,
        depthWrite: false };
      if (o.clip) params.clippingPlanes = o.clip;
      var geo = new T.BufferGeometry();
      var pos = new T.BufferAttribute(new Float32Array(12), 3);
      geo.setAttribute("position", pos);
      geo.setIndex([0, 1, 2, 0, 2, 3]);
      var mat = material("basic", token, params);
      var obj = add(new T.Mesh(geo, mat));
      return {
        set: function (a, b, c, d) {
          [a, b, c, d].forEach(function (p, i) { var v = V(p); pos.setXYZ(i, v.x, v.y, v.z); });
          pos.needsUpdate = true;
        },
        opacity: function (v) { mat.opacity = v; },
        show: shower(obj)
      };
    }

    /** Small spheres: dots("c2", {radius, max}).set([[x,y,z], ...]). */
    function dots(token, o) {
      o = o || {};
      var max = o.max || 256, m4 = new T.Matrix4();
      var params = o.clip ? { clippingPlanes: o.clip } : {};
      if (o.opacity !== undefined) { params.transparent = true; params.opacity = o.opacity; }
      var obj = add(new T.InstancedMesh(new T.SphereGeometry(o.radius || 0.05, 14, 10),
        material("lambert", token, params), max));
      obj.count = 0;
      return {
        set: function (pts) {
          obj.count = Math.min(pts.length, max);
          for (var i = 0; i < obj.count; i++) {
            var v = V(pts[i]);
            obj.setMatrixAt(i, m4.makeTranslation(v.x, v.y, v.z));
          }
          obj.instanceMatrix.needsUpdate = true;
        },
        show: shower(obj)
      };
    }

    /** A height-coloured grid surface with optional wire lines:
     *  surface(nu, nv, {wire, opacity, clip}).set(function (u, v) { return [x, y, z]; },
     *                                            function (u, v, p) { return THREE.Color; }) */
    function surface(nu, nv, o) {
      o = o || {};
      var count = (nu + 1) * (nv + 1);
      var geo = new T.BufferGeometry();
      var pos = new T.BufferAttribute(new Float32Array(count * 3), 3);
      var col = new T.BufferAttribute(new Float32Array(count * 3), 3);
      var index = [], i, j;
      for (i = 0; i < nu; i++) for (j = 0; j < nv; j++) {
        var a = i * (nv + 1) + j, b = a + nv + 1;
        index.push(a, b, a + 1, b, b + 1, a + 1);
      }
      geo.setAttribute("position", pos);
      geo.setAttribute("color", col);
      geo.setIndex(index);
      var params = { vertexColors: true, side: T.DoubleSide };
      if (o.opacity !== undefined) { params.transparent = true; params.opacity = o.opacity; }
      if (o.clip) params.clippingPlanes = o.clip;
      var mesh = add(new T.Mesh(geo, new T.MeshLambertMaterial(params)));
      var wire = null, wirePos = null, every = o.wire || 0;
      if (every) {
        var lines = Math.floor(nu / every) + 1, wireCount = 2 * (lines * nv + (Math.floor(nv / every) + 1) * nu);
        wirePos = new T.BufferAttribute(new Float32Array(wireCount * 3), 3);
        var wg = new T.BufferGeometry();
        wg.setAttribute("position", wirePos);
        var wp = { transparent: true, opacity: o.wireOpacity === undefined ? 0.35 : o.wireOpacity };
        if (o.clip) wp.clippingPlanes = o.clip;
        wire = add(new T.LineSegments(wg, material("line", o.wireToken || "ink", wp)));
      }
      return {
        set: function (where, shade) {
          var i, j, k = 0, grid = [];
          for (i = 0; i <= nu; i++) {
            grid.push([]);
            for (j = 0; j <= nv; j++, k++) {
              var p = where(i / nu, j / nv), v = V(p), c = shade(i / nu, j / nv, p);
              grid[i].push(v);
              pos.setXYZ(k, v.x, v.y, v.z);
              col.setXYZ(k, c.r, c.g, c.b);
            }
          }
          pos.needsUpdate = true; col.needsUpdate = true;
          geo.computeVertexNormals();
          if (!wire) return;
          k = 0;
          function seg(a, b) {
            wirePos.setXYZ(k++, a.x, a.y, a.z);
            wirePos.setXYZ(k++, b.x, b.y, b.z);
          }
          for (i = 0; i <= nu; i += every) for (j = 0; j < nv; j++) seg(grid[i][j], grid[i][j + 1]);
          for (j = 0; j <= nv; j += every) for (i = 0; i < nu; i++) seg(grid[i][j], grid[i + 1][j]);
          wirePos.needsUpdate = true;
        },
        show: function (on) { mesh.visible = !!on; if (wire) wire.visible = !!on; }
      };
    }

    /** A colour ramp between theme tokens: ramp(["paper", "c2", "c1"])(t) -> THREE.Color. */
    function ramp(tokens) {
      return function (t) {
        var cs = tokens.map(colour), x = clamp(t, 0, 1) * (cs.length - 1);
        var i = Math.min(cs.length - 2, Math.floor(x));
        return cs[i].clone().lerp(cs[i + 1], x - i);
      };
    }

    /** An HTML label pinned to a 3D point: label("x₁", "c1").set([x, y, z], text). */
    function label(text, token, o) {
      o = o || {};
      var node = WK.el("div", { class: "wk-stage-label", text: text });
      node.style.color = WK.cssVar(token || "ink");
      layer.appendChild(node);
      var l = { node: node, at: [0, 0, 0], on: true, dx: o.dx || 0, dy: o.dy || 0 };
      labels.push(l);
      return {
        set: function (at, t) { l.at = at; if (t !== undefined) node.textContent = t; },
        show: function (v) { l.on = !!v; }
      };
    }

    /** A flat grid in the plane z = z0 over [x0, x1] x [y0, y1]. */
    function grid(x, y, step, z0, token, opacity) {
      var pts = [], t;
      for (t = x[0]; t <= x[1] + 1e-9; t += step) pts.push([t, y[0], z0], [t, y[1], z0]);
      for (t = y[0]; t <= y[1] + 1e-9; t += step) pts.push([x[0], t, z0], [x[1], t, z0]);
      var g = line(token || "muted", { pairs: true, opacity: opacity === undefined ? 0.3 : opacity });
      g.set(pts);
      return g;
    }

    return {
      THREE: T, scene: scene, camera: camera, view: view, V: V,
      render: render, animate: animate,
      colour: colour, material: material, ramp: ramp,
      onTheme: function (cb) { themeCallbacks.push(cb); },
      clipZ: clipZ, clipBox: clipBox,
      arrow: arrow, line: line, quad: quad, dots: dots, surface: surface, label: label, grid: grid
    };
  }

  return { create: create };
})();
