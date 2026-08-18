# Kendall shape analysis of the kite sail, using Morphometrics in R (`shapes`).
#
# This is the independent check on the from-scratch Python pipeline in the post.
# It reads the same hand-digitised landmarks.csv, runs Procrustes twice -- once
# quotienting out similarity (what Kendall shape analysis does) and once
# quotienting out affine maps (what a distant camera actually applies) -- and
# writes the figures and numbers the post displays.
#
# Not run by the render. Regenerate by hand:
#   Rscript posts/kite-shape-analysis/src/shapes_analysis.R
#
# Needs `shapes` and `jsonlite`. The C++ toolchain on this machine cannot build
# rgl (a hard dependency of shapes) from source, so both came from the CRAN
# macOS binaries for R 4.0; see src/r-requirements.txt.

options(rgl.useNULL = TRUE)   # nothing here needs a 3D device
suppressMessages(library(shapes))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
post <- normalizePath(file.path(here, ".."))

ORDER <- c("nose", "armA_out", "armA_in", "apex_in", "armB_in", "armB_out")

# ---- load: landmarks.csv -> k x m x n array, the shape of every shapes:: input
lm <- read.csv(file.path(post, "landmarks.csv"), stringsAsFactors = FALSE)
frames <- sort(unique(lm$frame))
k <- length(ORDER); m <- 2; n <- length(frames)
X <- array(NA_real_, c(k, m, n))
for (i in seq_along(frames)) {
  f <- lm[lm$frame == frames[i], ]
  f <- f[match(ORDER, f$landmark), ]
  X[, , i] <- as.matrix(f[, c("x", "y")])
}
stopifnot(!any(is.na(X)))

# ---- centre and scale one configuration to unit centroid size
cs <- function(cfg) {
  cfg <- scale(cfg, center = TRUE, scale = FALSE)
  cfg / sqrt(sum(cfg^2))
}

# ---- quotient out the AFFINE group, without letting it collapse.
# Fitting affine maps iteratively to a mean does not work: the affine group can
# squash a configuration onto a line and still leave it at unit size, so the
# loop happily drives the whole sample flat (aspect ratio 2e-4 here) and reports
# a tiny residual for what is really a degenerate answer.
#
# The fix is to remove the affine part in closed form instead. Centre the
# configuration, then rescale its principal axes to be equal -- its second
# moment becomes isotropic. Any affine image of a configuration lands on the
# same canonical form, so what survives is exactly the non-affine shape, and
# nothing can collapse.
whiten <- function(cfg) {
  cfg <- scale(cfg, center = TRUE, scale = FALSE)
  w <- svd(cfg)$u * sqrt(nrow(cfg))   # singular values replaced by 1
  w / sqrt(sum(w^2))
}

# ---- Procrustes under each group.
# reflect = TRUE: the sail is bilaterally symmetric, so no single photograph
# says which physical wing is which. Allowing reflection stops that free choice
# showing up as shape difference.
W <- array(apply(X, 3, whiten), dim(X))

sim <- procGPA(X, scale = TRUE, reflect = TRUE)
aff <- procGPA(W, scale = TRUE, reflect = TRUE)

# procGPA leaves its aligned configurations at a common non-unit size. Put both
# runs on unit centroid size before comparing, or the two Procrustes sums of
# squares are in different units.
normalise_gpa <- function(fit) {
  rot <- array(apply(fit$rotated, 3, cs), dim(fit$rotated))
  list(rotated = rot, mshape = cs(apply(rot, c(1, 2), mean)))
}
sim_n <- normalise_gpa(sim)
aff_n <- normalise_gpa(aff)

resid_to_mean <- function(A, mu) apply(A, 3, function(cfg) sqrt(sum((cfg - mu)^2)))
sim_res <- resid_to_mean(sim_n$rotated, sim_n$mshape)
aff_res <- resid_to_mean(aff_n$rotated, aff_n$mshape)

# ---- pairwise Riemannian (Kendall) distance in shape space, raw landmarks
riem <- matrix(0, n, n)
for (i in 1:n) for (j in 1:n) if (i != j) riem[i, j] <- riemdist(X[, , i], X[, , j])

cat(sprintf("procGPA (similarity): Procrustes SS = %.5f, RMS residual = %.4f\n",
            sum(sim_res^2), mean(sim_res)))
cat(sprintf("affine GPA          : Procrustes SS = %.5f, RMS residual = %.4f\n",
            sum(aff_res^2), mean(aff_res)))
cat(sprintf("uniform (affine) share of shape variation: %.1f%%\n",
            100 * (1 - sum(aff_res^2) / sum(sim_res^2))))
cat(sprintf("aspect ratio of affine-quotient configs (1.0 = no collapse): %.3f\n",
            mean(apply(aff_n$rotated, 3, function(c) { s <- svd(c)$d; s[2] / s[1] }))))
cat("pairwise Riemannian distance (raw):\n"); print(round(riem, 4))

# ---- figures ---------------------------------------------------------------
# tpsgrid draws the thin-plate spline deformation carrying the mean shape onto
# one specimen. Under similarity alignment the grids have to absorb the camera's
# whole viewpoint change; under affine alignment only the sail's own bending is
# left for them to show.
grid_panel <- function(A, mu, file) {
  png(file, width = 1500, height = 380, res = 108)
  par(mfrow = c(1, n), mar = c(1, 1, 2.4, 1))
  for (i in seq_len(n)) {
    tpsgrid(mu, A[, , i], mag = 1, ngrid = 16, opt = 1, ext = 0.12)
    title(main = sprintf("frame %d", frames[i]), cex.main = 1.0)
  }
  invisible(dev.off())
  cat("wrote", basename(file), "\n")
}
grid_panel(sim_n$rotated, sim_n$mshape, file.path(post, "fig-r-tps-raw.png"))
grid_panel(aff_n$rotated, aff_n$mshape, file.path(post, "fig-r-tps-rectified.png"))

# ---- numbers the post reads back -------------------------------------------
# No shapepca figure here on purpose: with five configurations in an eight-
# dimensional shape space, the share of variance on PC1 is mostly an artefact of
# n, and would invite a reading the sample cannot support.
out <- list(
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  shapes_version = as.character(packageVersion("shapes")),
  n_frames = n, n_landmarks = k,
  similarity = list(procrustes_ss = sum(sim_res^2), rms_residual = mean(sim_res),
                    residuals = as.numeric(sim_res)),
  affine = list(procrustes_ss = sum(aff_res^2), rms_residual = mean(aff_res),
                residuals = as.numeric(aff_res)),
  uniform_share = 1 - sum(aff_res^2) / sum(sim_res^2),
  riemannian = riem
)
write_json(out, file.path(post, "r_results.json"), auto_unbox = TRUE, digits = 8, pretty = TRUE)
cat("wrote r_results.json\n")
