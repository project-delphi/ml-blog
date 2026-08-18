# Warp the kite photographs themselves into a common frame, and morph between them.
#
# Everything else in this post works on six landmarks. This script works on the
# pixels: it transports each photograph into one shared frame, once by the best
# similarity map and once by the best affine map, and animates the result. The
# claim the post makes numerically -- that the affine quotient registers these
# photographs and similarity does not -- becomes something you can watch.
#
# Not run by the render. Regenerate by hand:
#   Rscript posts/kite-shape-analysis/src/warp_movie.R
#
# Needs shapes, jpeg, png and gifski, all from the CRAN macOS binaries for
# R 4.0 -- see src/r-requirements.txt. Morpho would have supplied tps2d, but its
# CRAN binary links libRblas.dylib, which this Homebrew R does not ship, so the
# thin-plate spline below is written out longhand instead. That is no loss: it
# is the same algebra the post's Python cells use, and worth seeing once.

options(rgl.useNULL = TRUE)
suppressMessages({
  library(shapes)
  library(jpeg)
  library(png)
  library(gifski)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
post <- normalizePath(file.path(here, ".."))

ORDER <- c("nose", "armA_out", "armA_in", "apex_in", "armB_in", "armB_out")
BOX <- 220        # output panel is BOX x BOX pixels
FILL <- 0.52      # fraction of the panel the mean sail spans
STEPS <- 6        # in-between frames per transition
RAW_CROP <- 300   # native-pixel window for the unaligned baseline row
HOLD <- 2         # frames held on each photograph

# ---- landmarks -------------------------------------------------------------
lm <- read.csv(file.path(post, "landmarks.csv"), stringsAsFactors = FALSE)
frames <- sort(unique(lm$frame))
n <- length(frames)
k <- length(ORDER)
X <- array(NA_real_, c(k, 2, n))
for (i in seq_along(frames)) {
  f <- lm[lm$frame == frames[i], ]
  X[, , i] <- as.matrix(f[match(ORDER, f$landmark), c("x", "y")])
}

photos <- lapply(frames, function(f)
  readJPEG(file.path(post, sprintf("photos/kite-%02d.jpg", f))))

# ---- the shared frame ------------------------------------------------------
# procGPA supplies the mean shape, exactly as in shapes_analysis.R. Scale it to
# sit in the middle of the output panel so the warps land somewhere sensible.
gpa <- procGPA(X, scale = TRUE, reflect = TRUE)
mu <- gpa$mshape
mu <- mu - matrix(colMeans(mu), k, 2, byrow = TRUE)
mu <- mu / max(abs(mu)) * (BOX * FILL / 2)
mu <- mu + BOX / 2

# ---- the two transports ----------------------------------------------------
# Both map a frame's landmarks onto `mu`. Affine has six parameters to
# similarity's four, and that gap is the post's whole argument.
#
# Note this cannot collapse the way the iterative affine Procrustes did: there
# the target was re-estimated every pass, so the group could drag the mean flat
# and follow it. Here the target is fixed, so a degenerate fit has nothing to
# gain.
fit_affine <- function(src, dst) {
  design <- cbind(1, src)
  qr.solve(design, dst)                       # 3 x 2
}

fit_similarity <- function(src, dst) {
  mu_s <- colMeans(src); mu_d <- colMeans(dst)
  s0 <- sweep(src, 2, mu_s); d0 <- sweep(dst, 2, mu_d)
  sv <- svd(t(d0) %*% s0)
  rot <- sv$u %*% t(sv$v)                     # reflection allowed, as elsewhere
  scale <- sum(sv$d) / sum(s0^2)
  rbind(mu_d - scale * (mu_s %*% t(rot)), scale * t(rot))
}

apply_map <- function(m, pts) cbind(1, pts) %*% m

invert_map <- function(m) {
  a <- m[2:3, ]                               # linear part
  inv <- solve(a)
  rbind(-m[1, ] %*% inv, inv)
}

# ---- thin-plate spline -----------------------------------------------------
# U(r) = r^2 log r^2 is the 2D biharmonic kernel: the shape a thin metal plate
# takes when pinned at the landmarks. Solving L %*% W = [target; 0] gives an
# affine part plus one radial term per landmark, and the whole map is the
# smoothest one that carries every source landmark exactly onto its target.
tps_kernel <- function(d2) ifelse(d2 > 0, d2 * log(d2), 0)

tps_fit <- function(src, dst) {
  d2 <- as.matrix(dist(src))^2
  K <- tps_kernel(d2)
  P <- cbind(1, src)
  L <- rbind(cbind(K, P), cbind(t(P), matrix(0, 3, 3)))
  W <- solve(L, rbind(dst, matrix(0, 3, 2)))
  list(src = src, w = W[seq_len(nrow(src)), , drop = FALSE],
       a = W[nrow(src) + 1:3, , drop = FALSE])
}

tps_apply <- function(fit, pts) {
  d2 <- outer(pts[, 1], fit$src[, 1], "-")^2 + outer(pts[, 2], fit$src[, 2], "-")^2
  cbind(1, pts) %*% fit$a + tps_kernel(d2) %*% fit$w
}

# ---- sampling --------------------------------------------------------------
# Bilinear lookup. Coordinates are (x, y) in image pixels; the JPEG array is
# indexed [row, col, channel], so y picks the row.
sample_image <- function(img, pts, box = BOX) {
  h <- dim(img)[1]; w <- dim(img)[2]
  x <- pts[, 1]; y <- pts[, 2]
  x0 <- floor(x); y0 <- floor(y)
  fx <- x - x0; fy <- y - y0
  inside <- x0 >= 1 & y0 >= 1 & x0 < w & y0 < h
  cx0 <- pmin(pmax(x0, 1), w - 1); cy0 <- pmin(pmax(y0, 1), h - 1)
  out <- array(0.62, c(box, box, 3))          # neutral grey outside the photo
  for (ch in 1:3) {
    plane <- img[, , ch]
    v <- (1 - fx) * (1 - fy) * plane[cbind(cy0, cx0)] +
         fx * (1 - fy) * plane[cbind(cy0, cx0 + 1)] +
         (1 - fx) * fy * plane[cbind(cy0 + 1, cx0)] +
         fx * fy * plane[cbind(cy0 + 1, cx0 + 1)]
    v[!inside] <- 0.62
    out[, , ch] <- matrix(v, box, box)
  }
  out
}

# Output pixel grid, in the shared frame. Row-major so matrix(v, BOX, BOX) fills
# correctly: column 1 varies fastest down the rows.
grid_xy <- as.matrix(expand.grid(y = seq_len(BOX), x = seq_len(BOX))[, c("x", "y")])

# Warp photograph `i` into the shared frame, optionally after bending the frame
# to the intermediate landmark set `target`.
warp_frame <- function(i, transport, target = NULL) {
  lm_shared <- apply_map(transport[[i]], X[, , i])
  pts <- if (is.null(target)) grid_xy else tps_apply(tps_fit(target, lm_shared), grid_xy)
  sample_image(photos[[i]], apply_map(invert_map(transport[[i]]), pts))
}

transports <- list(
  similarity = lapply(seq_len(n), function(i) fit_similarity(X[, , i], mu)),
  affine     = lapply(seq_len(n), function(i) fit_affine(X[, , i], mu))
)

# ---- static figure: as photographed, then both transports ------------------
# A plain window on the original photograph. The only thing done to it is
# centring on the kite, so that it stays in frame at all; orientation is
# untouched and the window is the same size in source pixels for every frame,
# so relative size differences between frames survive. It is the baseline the
# two transports improve on, but it is not literally raw -- translation is
# already gone, which is why the post credits the next row with size and
# rotation rather than all three.
crop_raw <- function(i) {
  centre <- colMeans(X[, , i])
  offset <- sweep(grid_xy, 2, c(BOX, BOX) / 2) * (RAW_CROP / BOX)
  sample_image(photos[[i]], sweep(offset, 2, centre, "+"))
}

png(file.path(post, "fig-r-warped-stack.png"), width = n * BOX + 190,
    height = 3 * BOX + 60, res = 120)
par(mfrow = c(3, n), mar = c(0.4, 0.4, 1.9, 0.4), oma = c(0, 7.5, 0, 0))
rows <- list(
  list(lab = "as photographed", get = function(i) crop_raw(i)),
  list(lab = "similarity-aligned",
       get = function(i) warp_frame(i, transports$similarity)),
  list(lab = "affine-aligned",
       get = function(i) warp_frame(i, transports$affine))
)
for (r in seq_along(rows)) {
  for (i in seq_len(n)) {
    plot.new(); plot.window(c(0, 1), c(0, 1), asp = 1)
    rasterImage(rows[[r]]$get(i), 0, 0, 1, 1, interpolate = TRUE)
    if (r == 1) title(main = sprintf("frame %d", frames[i]), cex.main = 1.05)
    if (i == 1) mtext(rows[[r]]$lab, side = 2, line = 1.1, cex = 0.82, las = 0)
  }
}
invisible(dev.off())
cat("wrote fig-r-warped-stack.png\n")

# ---- plotshapes: raw versus aligned configurations -------------------------
png(file.path(post, "fig-r-plotshapes.png"), width = 1100, height = 520, res = 120)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plotshapes(X, joinline = c(seq_len(k), 1))
title(main = "raw landmarks, five frames")
plotshapes(gpa$rotated, joinline = c(seq_len(k), 1))
title(main = "after procGPA")
invisible(dev.off())
cat("wrote fig-r-plotshapes.png\n")

# ---- the two numbers the post quotes -----------------------------------------
# Neither of these is used to draw anything; they exist so the prose is checkable
# against a run of this script instead of against a scratch file.
report_statistics <- function() {
  # 1. Landmark registration: how far each frame's landmarks still sit from the
  #    shared mean, in panel pixels, under each transport.
  resid <- sapply(names(transports), function(nm) {
    mean(sapply(seq_len(n), function(i) {
      sqrt(mean(rowSums((apply_map(transports[[nm]][[i]], X[, , i]) - mu)^2)))
    }))
  })
  cat(sprintf("landmark registration: similarity %.2f px, affine %.2f px (%.0f%% better)\n",
              resid[["similarity"]], resid[["affine"]],
              100 * (1 - resid[["affine"]] / resid[["similarity"]])))

  # 2. Pixel agreement: the SD across the five aligned frames at each pixel and
  #    channel, averaged. Restricted to the landmarks' bounding box, because
  #    over the whole panel most of the frame is empty sky, which dilutes both
  #    numbers equally and makes neither interpretable. The post says which.
  rows <- seq(floor(min(mu[, 2])), ceiling(max(mu[, 2])))
  cols <- seq(floor(min(mu[, 1])), ceiling(max(mu[, 1])))
  for (nm in names(transports)) {
    stack <- vapply(seq_len(n), function(i) warp_frame(i, transports[[nm]]),
                    array(0, c(BOX, BOX, 3)))
    sd_map <- apply(stack, c(1, 2, 3), sd)
    cat(sprintf("per-pixel SD across the 5 aligned frames, %-10s sail box %.4f, whole panel %.4f\n",
                nm, mean(sd_map[rows, cols, ]), mean(sd_map)))
  }
}
report_statistics()

# ---- the morph -------------------------------------------------------------
# Between consecutive frames the landmarks are interpolated linearly and both
# photographs are bent onto that intermediate shape, then cross-dissolved. The
# thin-plate spline is what carries the pixels.
#
# The landmarks are drawn on, against the fixed mean shape in white, and that is
# not decoration: measured on the pixels alone the two panels are nearly
# indistinguishable. Pixel disagreement here is dominated by things no 2D
# alignment can fix -- which colour bands a viewpoint exposes, and the
# streamers, which are not in correspondence at all. What the affine map
# genuinely improves is the landmark registration, and on a BOX-pixel panel that
# is a couple of pixels, invisible without magnifying it.
#
# Both numbers the post quotes are printed by report_statistics() below, so the
# prose can be checked against a run rather than taken on trust.
gap <- 16
tmp <- file.path(tempdir(), "morph")
dir.create(tmp, showWarnings = FALSE)
paths <- character(0)
counter <- 0
hull <- c(1:k, 1)
DOT <- hcl.colors(k, "Dark 3")

MAG <- 3   # residuals are drawn magnified, as tpsgrid's `mag` does

panel <- function(img, marks, label) {
  plot.new()
  plot.window(c(0, BOX), c(BOX, 0), asp = 1)
  rasterImage(img, 0, BOX, BOX, 0, interpolate = TRUE)
  lines(mu[hull, 1], mu[hull, 2], col = "white", lwd = 2.4)
  lines(mu[hull, 1], mu[hull, 2], col = "black", lwd = 0.7, lty = 3)
  # How far each landmark still sits from the mean shape, drawn MAG times life
  # size. At life size the gap is a few pixels on a 240-pixel panel and simply
  # cannot be seen; magnifying it is the only way the comparison reads, and the
  # caption says so.
  ends <- mu + MAG * (marks - mu)
  segments(mu[, 1], mu[, 2], ends[, 1], ends[, 2], col = "black", lwd = 2.6)
  segments(mu[, 1], mu[, 2], ends[, 1], ends[, 2], col = "white", lwd = 1.2)
  points(ends[, 1], ends[, 2], pch = 21, bg = DOT, col = "black", cex = 1.2, lwd = 0.8)
  title(main = label, cex.main = 1.05, line = 0.5)
}

emit <- function(state) {
  counter <<- counter + 1
  path <- file.path(tmp, sprintf("f%03d.png", counter))
  png(path, width = 2 * BOX + gap + 30, height = BOX + 42, res = 100)
  par(mfrow = c(1, 2), mar = c(0.3, 0.3, 1.9, 0.3), bg = "white")
  panel(state$similarity$img, state$similarity$marks, "similarity-aligned")
  panel(state$affine$img, state$affine$marks, "affine-aligned")
  invisible(dev.off())
  paths <<- c(paths, path)
}

for (i in seq_len(n)) {
  j <- if (i == n) 1 else i + 1
  for (rep in seq_len(HOLD)) {
    emit(lapply(transports, function(tr) list(
      img = warp_frame(i, tr), marks = apply_map(tr[[i]], X[, , i]))))
  }
  for (step in seq_len(STEPS)) {
    t <- step / (STEPS + 1)
    emit(lapply(transports, function(tr) {
      li <- apply_map(tr[[i]], X[, , i])
      lj <- apply_map(tr[[j]], X[, , j])
      target <- (1 - t) * li + t * lj
      list(img = (1 - t) * warp_frame(i, tr, target) + t * warp_frame(j, tr, target),
           marks = target)
    }))
  }
}
cat("rendered", length(paths), "morph frames\n")

gif <- file.path(post, "fig-r-morph.gif")
info <- dim(readPNG(paths[1]))
gifski(paths, gif_file = gif, width = info[2], height = info[1],
       delay = 1 / 12, progress = FALSE)
cat(sprintf("  raw gifski output: %.2f MB\n", file.size(gif) / 1048576))

# gifski writes a full-colour GIF; gifsicle gets it under the ~0.5 MB the repo's
# other GIFs sit at. Quality is spent on the sail, so cut palette before frames.
if (nzchar(Sys.which("gifsicle"))) {
  system2("gifsicle", c("--optimize=3", "--colors", "96", "--lossy=40",
                        shQuote(gif), "-o", shQuote(gif)))
}
cat(sprintf("wrote fig-r-morph.gif (%.2f MB, %d frames)\n",
            file.size(gif) / 1048576, length(paths)))
