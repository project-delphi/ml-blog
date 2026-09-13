"""Build the two committed matrices this post reads at render time.

Run it yourself; the post never touches the network.

    uv run --no-project --with requests --with numpy --python 3.12 \
        python posts/six-views-of-pca/src/fetch_data.py

**MovieLens.** GroupLens ml-100k is 943 users x 1682 films with 100,000
ratings, so 93.7% of the user-by-film grid is empty. Every equivalence this
post derives assumes a fully observed matrix, and the largest complete
rectangle a greedy peel can find is tiny -- the script measures that lower
bound and records it in `manifest.json`, which the post quotes. So the matrix
the post uses is the complete users-by-genre table: for each user, the mean
rating they gave in each genre. A cell is defined whenever that user rated at
least MIN_PER_GENRE films carrying that genre tag, and the script keeps the
users for whom every kept genre clears that bar. No holes, no imputation.

**Metabolome.** Li et al. (2023) deposited LC-MS peak areas for neutrophils
from 75 people as Metabolomics Workbench ST002477 (CC BY 4.0). The two TSVs are
copied byte-for-byte from posts/neutrophil-metabolome-axis/data/, which is
where they were first landed; copying keeps this post self-contained and keeps
both copies matching a fresh fetch.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data"
SIBLING = HERE.parent.parent / "neutrophil-metabolome-axis" / "data"

# Keep the 12 genres the most users have rated, and require this many films per
# genre per user so each cell is a mean over more than one opinion. Three is the
# largest bar that still leaves a few hundred users; see the module docstring.
N_GENRES = 12
MIN_PER_GENRE = 3


def download() -> tuple[str, str, str]:
    """Return (u.data, u.item, u.genre) as decoded text."""
    # files.grouplens.org presented an expired certificate when this was last
    # run; the sibling tensor post's fetcher carries the same note.
    resp = requests.get(
        MOVIELENS_URL,
        headers={"User-Agent": UA},
        timeout=120,
        allow_redirects=True,
        verify=False,
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return (
            zf.read("ml-100k/u.data").decode("ascii"),
            zf.read("ml-100k/u.item").decode("latin-1"),
            zf.read("ml-100k/u.genre").decode("latin-1"),
        )


def parse(data: str, items: str, genres: str):
    rows = [ln.split("\t") for ln in data.splitlines() if ln]
    user = np.array([int(r[0]) for r in rows])
    film = np.array([int(r[1]) for r in rows])
    score = np.array([float(r[2]) for r in rows])

    pairs = sorted(
        (int(ln.rsplit("|", 1)[1]), ln.rsplit("|", 1)[0])
        for ln in genres.splitlines()
        if ln.strip()
    )
    names = [name for _, name in pairs]

    n_films = max(int(ln.split("|")[0]) for ln in items.splitlines() if ln) + 1
    flags = np.zeros((n_films, len(names)), dtype=bool)
    for ln in items.splitlines():
        if ln:
            parts = ln.split("|")
            flags[int(parts[0])] = [c == "1" for c in parts[5:]]
    return user, film, score, names, flags


def greedy_complete_film_block(user, film, score) -> tuple[int, int]:
    """Find a complete users-by-films rectangle by greedy peeling.

    Seeds on the 400 densest users and 200 densest films, then drops whichever
    single row or column is emptiest until no cell is missing. The result is a
    **lower bound**, not the maximum: finding the largest complete submatrix is
    the maximum edge biclique problem, which is NP-hard, and the seed window
    puts any block outside it out of reach. The post says so rather than
    claiming a maximum, and compares the number against what aggregating to
    genres returns instead.

    Returns (users, films).
    """
    grid = np.zeros((user.max() + 1, film.max() + 1))
    grid[user, film] = score
    seen = grid > 0
    rows = list(np.argsort(-seen.sum(axis=1))[:400])
    cols = list(np.argsort(-seen.sum(axis=0))[:200])
    while rows and cols:
        sub = seen[np.ix_(rows, cols)]
        if sub.all():
            return len(rows), len(cols)
        row_missing = 1 - sub.mean(axis=1)
        col_missing = 1 - sub.mean(axis=0)
        if row_missing.max() >= col_missing.max():
            rows.pop(int(np.argmax(row_missing)))
        else:
            cols.pop(int(np.argmax(col_missing)))
    return 0, 0


def genre_table(user, film, score, names, flags):
    """Complete users x genres table of mean ratings, plus the kept labels."""
    n_users = user.max() + 1
    totals = np.zeros((n_users, len(names)))
    counts = np.zeros((n_users, len(names)))
    for gi in range(len(names)):
        sel = flags[film, gi]
        np.add.at(totals[:, gi], user[sel], score[sel])
        np.add.at(counts[:, gi], user[sel], 1.0)

    coverage = (counts > 0).sum(axis=0)
    keep = sorted(np.argsort(-coverage)[:N_GENRES])
    enough = (counts[:, keep] >= MIN_PER_GENRE).all(axis=1)
    enough[0] = False  # user ids start at 1
    users = np.where(enough)[0]
    table = totals[np.ix_(users, keep)] / counts[np.ix_(users, keep)]
    kept_counts = counts[np.ix_(users, keep)]

    # Pooled variance of single ratings *within* a user-by-genre cell. This is
    # the scale of the sampling noise in each cell mean, and it is much smaller
    # than the variance of all 100,000 ratings, because the user's own level and
    # the genre's own level are already out of it.
    ss = 0.0
    df = 0
    for gi in keep:
        sel = flags[film, gi]
        u_sel, s_sel = user[sel], score[sel]
        for uid in users:
            vals = s_sel[u_sel == uid]
            if vals.size > 1:
                ss += float(((vals - vals.mean()) ** 2).sum())
                df += vals.size - 1
    within = ss / df
    return users, [names[i] for i in keep], table, kept_counts, within


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    user, film, score, names, flags = parse(*download())

    block_users, block_films = greedy_complete_film_block(user, film, score)
    users, kept, table, kept_counts, within = genre_table(
        user,
        film,
        score,
        names,
        flags,
    )

    header = ["user_id"] + [g.replace(",", ";") for g in kept]
    lines = [",".join(header)]
    for uid, row in zip(users, table):
        lines.append(",".join([str(uid)] + [f"{v:.4f}" for v in row]))
    dest = OUT / "movielens_genres.csv"
    dest.write_text("\n".join(lines) + "\n")
    print(f"wrote {dest} ({len(users)} users x {len(kept)} genres, no missing cells)")

    # How many films each cell averages over. A mean of three ratings and a mean
    # of two hundred are both one number in the table, and the post needs to say
    # which cells are which.
    lines = [",".join(header)]
    for uid, row in zip(users, kept_counts):
        lines.append(",".join([str(uid)] + [str(int(v)) for v in row]))
    dest = OUT / "movielens_counts.csv"
    dest.write_text("\n".join(lines) + "\n")
    print(f"wrote {dest} (films behind each cell)")

    manifest = {
        "movielens": {
            "url": MOVIELENS_URL,
            "release": "ml-100k",
            "grid_density": round(float(len(score) / ((user.max()) * (film.max()))), 4),
            "greedy_complete_film_block": [block_users, block_films],
            "n_genres": len(kept),
            "min_films_per_genre": MIN_PER_GENRE,
            "n_users": int(len(users)),
            "within_cell_rating_variance": round(within, 4),
            "genres": kept,
        },
        "metabolome": {
            "study": "ST002477",
            "url": "https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST002477",
            "licence": "CC BY 4.0",
            "copied_from": "posts/neutrophil-metabolome-axis/data/",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT / 'manifest.json'}")
    print(f"greedy complete film block: {block_users} users x {block_films} films")

    for name in ("abundances.tsv", "samples.tsv"):
        shutil.copyfile(SIBLING / name, OUT / name)
        print(f"copied {OUT / name} ({(OUT / name).stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
