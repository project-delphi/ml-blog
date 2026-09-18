# widget-kit

The shared runtime and chrome for the blog's interactive widgets.

- `kit.js` — one global, `WK`: element helpers, a responsive SVG, a linear scale
  with ticks, sliders, toggles, checkboxes, stat tiles, the frame, pointer dragging
  in SVG units, and a `mount` that runs your builder once the DOM is ready.
- `chrome.scss` — the frame (`.widget-container` and friends), compiled into both
  halves of the site theme, so it follows the reader's light/dark choice.

Every post loads `kit.js` through `posts/_metadata.yml`, so a widget file assumes
`WK` exists and does not load anything itself.

## Writing a widget

1. In `index.qmd`, list the sidecars as resources and give the widget a mount point
   and the script tags. No Python cell is involved, so a prose-only post can carry a
   widget and a freeze-backed post does not re-execute when the widget changes.

   ```yaml
   resources:
     - widgets.js
     - widget-data/data.js   # only if the widget ships precomputed data
   ```

   ```markdown
   ::: {#widget-attention}
   :::

   ```{=html}
   <script src="widget-data/data.js"></script>
   <script src="widgets.js"></script>
   ```
   ```

2. In `widgets.js`, build inside `WK.mount`:

   ```js
   "use strict";
   WK.mount("widget-attention", function (root, WK) {
     var f = WK.frame({ title: "Softmax attention", note: "Drag the temperature." });
     var temp = WK.slider({ label: "Temperature", min: 0.1, max: 5, step: 0.1,
                            value: 1, fmt: WK.fmt.num, oninput: draw });
     f.controls.appendChild(temp.root);
     var svg = WK.svg(600, 300);
     f.body.appendChild(svg);
     root.appendChild(f.root);
     function draw() {
       WK.clear(svg);
       svg.appendChild(WK.h("rect", { x: 10, y: 10, width: 100, height: 40, fill: "c1" }));
     }
     draw();
   });
   ```

3. Colour is a token name, never a hex value: `ink`, `muted`, `rule`, `paper`,
   `surface`, `accent`, `c1`..`c6`. `WK.h` writes `fill: "c1"` as `style="fill:
   var(--w-c1)"`, so the theme toggle repaints the widget with no redraw. A canvas
   widget reads `WK.palette()` and redraws in `WK.onTheme`.

4. Precomputed data goes in `widget-data/data.js` as `window.WK_DATA = {...}` (a
   `fetch()` of JSON fails from `file://`), read with `WK.data("key")`. The script
   that writes it lives in the post's `src/`.

5. Keep the model separate from the drawing: a function of the control values that
   returns numbers, and a `draw` that renders them. The prose then tells the reader
   what to drag and what they will see, and the model is what a review checks that
   claim against.

## Why not the inline print cell

The older widgets printed their `widgets.js` into the page from a Python cell so the
bundle sat inside the frozen record. That tied a widget change to a re-execution of
the post, needed a kernel for a prose-only post, and left the bundle outside Quarto's
freeze hash. A `<script src>` resource is copied on every render instead.
