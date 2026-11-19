# Data for the "Two Cultures" series

Public datasets used to revisit Breiman (2001), "Statistical Modeling: The Two Cultures",
*Statistical Science* 16(3):199–231. Every file under `raw/` is a gzip-compressed CSV with
a header row (gzip `mtime` set to 0, so a rebuild is byte-identical). Columns keep the
source's order. Missing-value codes (`?` in UCI files, `NA` in R) are empty cells. Where
the source ships a train/test split, it is kept as two files.

Access date for sections 1–7: **2026-11-18**; section 8 gives its own. The SHA-256 values are for the files as
downloaded, before any conversion. Sections 1–7 were converted with pandas and pyreadr by `../src/convert_raw.py`, which reads only the downloads (kept outside the repo) and writes `raw/`; section 8's files come from `../src/fetch_delhi.py`. Nothing was
imputed, recoded, deduplicated or reordered unless the section says so.

## Summary

| File(s) | Rows × cols | Classes (target) | Missing cells | Size (bytes) | Source |
|---|---|---|---|---|---|
| `hepatitis.csv.gz` | 155 × 20 | 2 (`Class`: 1=DIE 32, 2=LIVE 123) | 167 (75 rows) | 2,057 | UCI 46 |
| `bupa.csv.gz` | 345 × 7 | `selector` 1: 145, 2: 200 (**not a label**) | 0 | 2,550 | UCI 60 |
| `ozone_mlbench.csv.gz` | 366 × 13 | regression (V4) | 203 (163 rows) | 6,332 | mlbench 2.1-11 |
| `ozone_faraway.csv.gz` | 330 × 10 | regression (O3) | 0 | 4,760 | faraway 1.0.9 |
| `breast_cancer_wisconsin.csv.gz` | 699 × 11 | 2 (2: 458, 4: 241) | 16 (`Bare_Nuclei`) | 5,714 | UCI 15 |
| `ionosphere.csv.gz` | 351 × 35 | 2 (g: 225, b: 126) | 0 | 28,022 | UCI 52 |
| `pima_diabetes.csv.gz` | 768 × 9 | 2 (neg: 500, pos: 268) | 0 coded (zeros, see below) | 8,950 | mlbench 2.1-9 (archived) |
| `glass.csv.gz` | 214 × 11 | 6 (`Type`) | 0 | 3,785 | UCI 42 |
| `soybean_train.csv.gz` / `soybean_test.csv.gz` | 307 × 36 / 376 × 36 | 19 | 712 / 1,625 | 2,593 / 2,810 | UCI 90 |
| `letters.csv.gz` | 20,000 × 17 | 26 | 0 | 193,755 | UCI 59 |
| `satimage_train.csv.gz` / `satimage_test.csv.gz` | 4,435 × 37 / 2,000 × 37 | 6 | 0 | 60,070 / 38,563 | UCI 146 |
| `shuttle_train.csv.gz` / `shuttle_test.csv.gz` | 43,500 × 10 / 14,500 × 10 | 7 | 0 | 313,528 / 105,394 | UCI 148 |
| `dna_train.csv.gz` / `dna_test.csv.gz` | 2,000 × 181 / 1,186 × 181 | 3 | 0 | 50,623 / 30,531 | mlbench 2.1-11 |
| `usps_zip_train.csv.gz` / `usps_zip_test.csv.gz` | 7,291 × 257 / 2,007 × 257 | 10 | 0 | 1,574,974 / 436,920 | ESL data page |
| `lymphoma_breiman.csv.gz` | 81 × 4,683 | 3 (0: 29, 1: 43, 2: 9) | 0 (imputed upstream) | 771,570 | Breiman's RF site (Wayback) |
| `default_credit.csv.gz` | 30,000 × 25 | 2 (0: 23,364, 1: 6,636) | 0 | 979,797 | UCI 350 |
| `lalonde_nsw.csv.gz` | 445 × 11 | `treat` 1: 185, 0: 260 | 0 | 4,820 | Dehejia (NBER) |
| `lalonde_psid.csv.gz` | 2,490 × 11 | `treat` 0: 2,490 | 0 | 25,423 | Dehejia (NBER) |
| `delhi_pm25_daily.csv.gz` | 3,652 × 4 | regression (`pm25`) | 1,490 (745 days without a daily mean) | 26,727 | US Embassy New Delhi via OpenAQ |
| `delhi_era5_daily.csv.gz` | 3,652 × 15 | (predictors) | 543 (`blh_*`, 181 days) | 131,971 | ERA5 via Open-Meteo |

All of `raw/` together, documentation and section 8's files included, is 4,940,278 bytes (about 4.9 MB).

**Mapping to Breiman's Table 1.** The column counts there are covariates only. To get
them, drop `Sample_code_number` and `Class` from Cancer (9 covariates), `Id` and `Type`
from Glass (9), and the class column everywhere else. DNA's 60 variables are the 60
nucleotides. The file has the Statlog 180-indicator encoding (see the DNA section).

**Licences.** Every UCI page used here says: "This dataset is licensed under a Creative
Commons Attribution 4.0 International (CC BY 4.0) license." The CRAN packages are GPL
(mlbench: `GPL-2`; faraway: `GPL`, version not given). Sections that differ say so.

**Conversion code is not in the repo yet.** The download and convert steps were run
from a session scratch script (`convert.py`). Move it to `src/` if the series needs a
one-command rebuild.

**Not in git yet.** The repo's `.gitignore` ignores `data/` everywhere (line 167), so this
directory needs a carve-out (the directory *and* `/**`) before it can be committed.

---

## 1. Hepatitis (UCI 46)

- Source: <https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data>,
  `hepatitis.names` in the same directory. Landing page
  <https://archive.ics.uci.edu/dataset/46/hepatitis>, DOI 10.24432/C5Q59J. Licence CC BY 4.0.
- SHA-256: `hepatitis.data` `9971f9d18abb4fc9dd6749eefb0f864b36fa7bdd59f54188c0b148a5c28214bc`;
  `hepatitis.names` `d8ada8cb753e2cbba61ba7254c5a4f25a316e3f38cc77011a1ea6007c1c86e95` (kept as `raw/hepatitis.names`).
- Written: 155 × 20. Coding is unchanged from the source: `Class` 1=DIE and 2=LIVE, binary
  covariates 1=no and 2=yes, `SEX` 1=male and 2=female.
- Missing (`?`): 167 cells in 75 rows: STEROID 1, FATIGUE 1, MALAISE 1, ANOREXIA 1,
  LIVER_BIG 10, LIVER_FIRM 11, SPLEEN_PALPABLE 5, SPIDERS 5, ASCITES 5, VARICES 5,
  BILIRUBIN 6, ALK_PHOSPHATE 29, SGOT 4, ALBUMIN 16, PROTIME 67.

Attribute order, as in `hepatitis.names`. The `.names` number counts the class as
attribute 1; the covariate number leaves the class out:

| .names # | Covariate # | Attribute |
|---|---|---|
| 1 | – | Class (DIE, LIVE) |
| 2 | 1 | AGE |
| 3 | 2 | SEX |
| 4 | 3 | STEROID |
| 5 | 4 | ANTIVIRALS |
| 6 | 5 | FATIGUE |
| 7 | 6 | MALAISE |
| 8 | 7 | ANOREXIA |
| 9 | 8 | LIVER BIG |
| 10 | 9 | LIVER FIRM |
| 11 | 10 | SPLEEN PALPABLE |
| 12 | 11 | SPIDERS |
| 13 | 12 | ASCITES |
| 14 | 13 | VARICES |
| 15 | 14 | BILIRUBIN |
| 16 | 15 | ALK PHOSPHATE |
| 17 | 16 | SGOT |
| 18 | 17 | ALBUMIN |
| 19 | 18 | PROTIME |
| 20 | 19 | HISTOLOGY |

Column names in the CSV are these names with spaces replaced by `_`.


**Binary codes, read from the data.** `hepatitis.names` lists each clinical finding as "no, yes" without numeric codes. The data say 1 = present, 2 = absent: 14 of the 20 patients coded 1 for ASCITES died (70%), against 17 of 130 coded 2 (13%), and their mean ALBUMIN is 3.0 against 4.0. Codes are kept as given.
## 2. Liver Disorders / BUPA (UCI 60)

- Source: <https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data>,
  `bupa.names`. Landing page <https://archive.ics.uci.edu/dataset/60/liver+disorders>.
  Licence CC BY 4.0.
- SHA-256: `bupa.data` `a166a3e7a6f4dc41aaaedc59a107e57d8adcaeb8821f0873d756982f1ea74c92`;
  `bupa.names` `b8c348733ec58bd1cb489a5307409f60226b9afa30fd99b24fb593e9abc08dfe`.
- Written: 345 × 7 (`mcv, alkphos, sgpt, sgot, gammagt, drinks, selector`). No missing values.
  `selector`: 1 → 145, 2 → 200.
- **`selector` is not a disease label.** The current UCI page says: "The 7th field
  (selector) has been widely misinterpreted in the past as a dependent variable
  representing presence or absence of a liver disorder. This is incorrect [1]. The 7th
  field was created by BUPA researchers as a train/test selector." It adds that the
  dataset has no variable for presence or absence of a liver disorder, and that the donor
  dichotomised `drinks` to use as the dependent variable. The paper behind [1] is
  McDermott & Forsyth (2016), *Pattern Recognition Letters* 73:41–43,
  doi:10.1016/j.patrec.2016.01.004. Its abstract opens: "A large majority of the many
  hundreds of papers which use the UCI BUPA Liver Disorders data set as a benchmark for
  classification misunderstand the data and use an unsuitable dependent variable."
  (Abstract via <https://gpbib.cs.ucl.ac.uk/gp-html/MCDERMOTT201641.html>.) The bundled
  `bupa.names` is older and calls field 7 only a "field used to split data into two sets".

## 3. LA ozone

### mlbench `Ozone`

- Source: <https://cran.r-project.org/src/contrib/mlbench_2.1-11.tar.gz> (current CRAN
  release, published 2026-08-14), `data/Ozone.rda`. Licence GPL-2.
- SHA-256: tarball `8e92564ae6e456960028440ba1ffea8dbd4790585242d52721bb83242d9a69b5`;
  `Ozone.rda` `bfa5e55f1be00b6d38e1b5c82906ca44a5aee7696c5fa506e8b6c26663336b7c`.
- Written: 366 × 13 (`V1`…`V13`). V1–V3 are R factors (month, day of month, day of week),
  written as their labels (integers). Help text: `raw/Ozone_mlbench.Rd.txt`.
- Missing: 203 cells in 163 rows (V4 5, V5 12, V7 15, V8 2, V9 139, V10 15, V11 1, V12 14),
  which leaves 203 complete rows.

### faraway `ozone`

- Source: <https://cran.r-project.org/src/contrib/faraway_1.0.9.tar.gz> (current CRAN
  release, published 2025-02-07), `data/ozone.rda`. Licence GPL.
- SHA-256: tarball `1074ed85a0cf018fc2fd3e2a48bed666b2fa06b8461b7a3820aeb67b34a52d5d`;
  `ozone.rda` `32d2ee0ae06fb2b653a378a5e7564b23ea87b32c8af9863688f05693b9452b63`.
- Written: 330 × 10 (`O3, vh, wind, humidity, temp, ibh, dpg, ibt, vis, doy`), complete.
  Help text: `raw/ozone_faraway.Rd.txt`. It cites Breiman & Friedman (1985), and says
  that "a number of cases with missing variables have been removed".
- Cross-check: drop mlbench's V9 (El Monte temperature, 139 NAs) and take its complete
  rows. That leaves exactly 330 rows, and 8 of faraway's 10 columns match them row for
  row (O3=V4, vh=V5, wind=V6, humidity=V7, temp=V8, ibh=V10, dpg=V11, vis=V13). `ibt` does
  **not** equal V12 (e.g. 87 against 30.56) and is not a clean °C→°F conversion of it.
  `doy` runs 33–390, so it is not calendar day-of-year. Neither difference was resolved.

## 4. Breiman's Table 1 datasets

### Cancer: Wisconsin breast cancer, original (UCI 15)

- Source: <https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data>
  and `.names`. Landing page <https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original>. Licence CC BY 4.0.
- SHA-256: data `402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64`;
  names `4bcd6e4e0e4f8ae54e313ebbf18013cc83b6827a207123707687f8f19d29dd2c`.
- Written: 699 × 11 (sample code number, 9 cytology scores, `Class` 2=benign 458,
  4=malignant 241). 16 missing (`?`), all in `Bare_Nuclei`.

### Ionosphere (UCI 52)

- Source: <https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data> and `.names`.
  Landing page <https://archive.ics.uci.edu/dataset/52/ionosphere>. Licence CC BY 4.0.
- SHA-256: data `46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd`;
  names `a368e4fd9b910a3ac97fa306add17446c13105367e17f5b691455db2ea8411cf`.
- Written: 351 × 35 (`a1`…`a34`, `class`). g=225, b=126. No missing values.

### Diabetes: Pima Indians (mlbench `PimaIndiansDiabetes`, archived release)

- **No longer distributed by UCI or by current mlbench.** The UCI file
  (`…/pima-indians-diabetes/pima-indians-diabetes.data`) returns 404. The mlbench NEWS for
  2.1-10 says: "Removed the Pima Indians Diabetes data sets, following their removals from
  the UCI Repository". Neither source gives a reason.
- Source used: the last release that still has it,
  <https://cran.r-project.org/src/contrib/Archive/mlbench/mlbench_2.1-9.tar.gz> (published
  2026-07-13), `data/PimaIndiansDiabetes.rda`. This is the original, not
  `PimaIndiansDiabetes2`. Licence GPL-2. The data's originator is the National Institute
  of Diabetes and Digestive and Kidney Diseases, and the donor is V. Sigillito (per the
  help file `raw/PimaIndiansDiabetes_mlbench.Rd.txt`).
- SHA-256: tarball `7d07ef94ea1e62a9f34b2c6f742e074a1ce27dfac3a46cef41409179bec2673f`;
  `.rda` `29009898a176a9bc937ac2bd06324b08b803ca2619709675c0b0a70faa90f5bb`.
- Written: 768 × 9. neg=500, pos=268. No coded missing values. The physically impossible
  zeros are left as they are: glucose 5, pressure 35, triceps 227, insulin 374, mass 11
  (pregnant has 111 zeros, which are legitimate).
- Before publishing, consider why UCI withdrew this dataset.

### Glass (UCI 42)

- Source: <https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data> and `.names`.
  Landing page <https://archive.ics.uci.edu/dataset/42/glass+identification>. Licence CC BY 4.0.
- SHA-256: data `dd67373f4baf2807345df02cbfef2093d342e61ad0d82a4fb79af43ef8ce449d`;
  names `5921d4b93ed323aa092a2c17cc0bab3bafbf66b7085f5d98880d9c5ea0c51d42`.
- Written: 214 × 11 (`Id, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, Type`). Type counts:
  1: 70, 2: 76, 3: 17, 5: 13, 6: 9, 7: 29 (there is no type 4). No missing values.

### Soybean (large) (UCI 90, original split)

- Chose UCI's `soybean-large.data` (307, the original training set) and
  `soybean-large.test` (376), which together make Breiman's 683. Landing page
  <https://archive.ics.uci.edu/dataset/90/soybean+large>. Licence CC BY 4.0.
- SHA-256: `.data` `04b99f2728ded9f544022d2b4f6cce0ebe11ef5c9d44f10e21acd7093876507e`;
  `.test` `ff1c5f5c41ddc9a8e3746648a2fe5cafb6b0eef86bd49575decbb665e1b6a104`;
  `.names` `93b8763b12745e0207ce41c9e302dc07657e4405d2e6cd34d821788e9d33e559`.
- Written: 307 × 36 and 376 × 36 (`class` then the 35 attributes, in `.names` order and
  spelling, with the two spaced names hyphenated). Attributes are integer codes.
  Missing (`?`): 712 cells in 41 rows (train) and 1,625 cells in 80 rows (test).
- Cross-check: train followed by test equals mlbench 2.1-11 `Soybean` (683 × 36) row for
  row, in values, missing pattern (2,337 NAs) and class. mlbench's help:
  `raw/Soybean_mlbench.Rd.txt`.
- Class counts (train / test): 2-4-d-injury 1/15, alternarialeaf-spot 40/51, anthracnose 20/24,
  bacterial-blight 10/10, bacterial-pustule 10/10, brown-spot 40/52, brown-stem-rot 20/24,
  charcoal-rot 10/10, cyst-nematode 6/8, diaporthe-pod-&-stem-blight 6/9,
  diaporthe-stem-canker 10/10, downy-mildew 10/10, frog-eye-leaf-spot 40/51,
  herbicide-injury 4/4, phyllosticta-leaf-spot 10/10, phytophthora-rot 40/48,
  powdery-mildew 10/10, purple-seed-stain 10/10, rhizoctonia-root-rot 10/10.

### Letters (UCI 59)

- Source: <https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data> and `.names`.
  Landing page <https://archive.ics.uci.edu/dataset/59/letter+recognition>. Licence CC BY 4.0.
- SHA-256: data `2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2`;
  names `128b85adc32bd58a93ed6e1df136b1d7c636f85f290ba3db655ccaa19c86d324`.
- Written: the full 20,000 × 17 (`lettr` then 16 integer features), in file order.
  26 classes, 734–813 each. No missing values.
- **Split:** Breiman used 15,000 train / 5,000 test. The split is done in code: rows
  1–15,000 train, rows 15,001–20,000 test. This is not the donors' split: `.names` says
  "We typically train on the first 16000 items and then use the resulting model to predict
  the letter category for the remaining 4000."

### Satellite: Statlog Landsat (UCI 146)

- Source: <https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn>,
  `sat.tst`, `sat.doc` (kept as `raw/satimage.doc.txt`). Landing page
  <https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite>. Licence CC BY 4.0.
- SHA-256: `sat.trn` `e896dc88a960fa2404160fc4c3cb3dc53fcf4afd80ba920bf2d261bd42d12613`;
  `sat.tst` `4b9167b8a92baafafed7c8809aef86d0683a5e685c19d98d119fa6974f0f2479`;
  `sat.doc` `07f0312277b3bc4600548a0b75a848d72f1c88d9de416711b05cc2535d961dac`.
- Written: 4,435 × 37 and 2,000 × 37 (`A1`…`A36`, `class`). Classes 1, 2, 3, 4, 5, 7
  (there is no class 6). Train: 1072/479/961/415/470/1038. Test: 461/224/397/211/237/470.

### Shuttle: Statlog (UCI 148)

- Source: <https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z>
  (Unix-compressed, decompressed with `gzip -dc`), `shuttle.tst`, `shuttle.doc` (kept as
  `raw/shuttle.doc.txt`). Landing page <https://archive.ics.uci.edu/dataset/148/statlog+shuttle>. Licence CC BY 4.0.
- SHA-256: `shuttle.trn.Z` `ffd2cbf221d4982ccc77c674621c23291447f3fd9484c00d8078858965e0e7ed`;
  `shuttle.tst` `f776934a628d9b94c482cb058a76ddaa63823e0f74aec065813c9dff3b89d661`;
  `shuttle.doc` `c46e3f4481b514df9311046cb64a4a1adefd13663ccad375e27c404e3d829a37`.
- Written: 43,500 × 10 and 14,500 × 10 (`A1`…`A9`, `class`). Train classes 1–7:
  34108/37/132/6748/2458/6/11. Test: 11478/13/39/2155/809/4/2.

### DNA: Statlog primate splice junctions (mlbench `DNA`)

- UCI no longer serves `statlog/dna/` (it returns NOT FOUND). Source used:
  mlbench 2.1-11 `data/DNA.rda` (3,186 × 181: `V1`…`V180` binary indicators, then `Class`
  ei/ie/n). SHA-256 of the `.rda`:
  `da9cb09a54663474a6cda54172b6b0d37058ac365c89f42e62c8a684071ab567`. Licence GPL-2.
  Help: `raw/DNA_mlbench.Rd.txt`.
- **Split verified.** The libsvm copies
  <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale> (2,000)
  and `dna.scale.t` (1,186) keep the Statlog split. Their SHA-256 values are
  `03d39477c5e48415e9cd18e006efc8f7a3b5c3deaee8289a72511c35070e3cfe` and
  `d2eeed170e4e38ae2aba6aa4cd27a8ca4ecceb01a7751e8e509bb4e1a6bcef39`. Stacked train then
  test, they equal mlbench's 3,186 rows **in order**: all 180 indicators match on every
  row (2,000/2,000 and 1,186/1,186). The labels map one to one (ei↔1: 767, ie↔2: 765,
  n↔3: 1,654). So mlbench rows 1–2,000 are the training set and 2,001–3,186 the test set,
  and they are written as `dna_train` / `dna_test`.
- Class counts: train ei 464, ie 485, n 1,051; test ei 303, ie 280, n 603.
- Encoding (from the help file): each of the 60 nucleotides is three indicators,
  A→100, C→010, G→001, T→000. Decode position *k* from columns 3k−2…3k to recover
  Breiman's 60 four-valued variables.

### Digit: USPS ZIP code (ESL)

- Source: <https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz>, `zip.test.gz`,
  `zip.info.txt` (kept as `raw/usps_zip.info.txt`). The page states no licence. It is
  distributed as data for Hastie, Tibshirani & Friedman, *The Elements of Statistical
  Learning*, from Le Cun et al. (1990).
- SHA-256: `zip.train.gz` `66ebf231ab51f2cf6ac1a892d0f98ccffee1079d7a5db638c1f11f366046ca16`;
  `zip.test.gz` `8dacc3c6e136af6c3488de9e5113a7daf140c8c2f1f0b61c54ee02f03f1a962f`;
  `zip.info.txt` `927ca750c4117b61a71def180d5e4f00f1323a4b2c9cd990f576aa8def3505f1`.
- Written: 7,291 × 257 and 2,007 × 257 (`digit`, then `p1`…`p256`: 16×16 grey levels in
  [−1, 1], row-major). The trailing-whitespace empty column was dropped.
  Train digits 0–9: 1194/1005/731/658/652/556/664/645/542/644. Test:
  359/264/198/166/200/160/170/147/166/177.

## 5. Lymphoma microarray (Breiman's own file)

This is **the exact matrix Breiman used**, recovered from his random-forests site through
the Wayback Machine.

- Breiman's RF page (archived 2004,
  <http://web.archive.org/web/20040510073250/http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm>)
  says: "we illustrate them on an early microarray lymphoma data set with 81 cases, 3
  classes, and 4682 variables corresponding to gene expressions."
- His example program `cc_examples/micro.f` (kept as `raw/breiman_micro.f.txt`, SHA-256
  `d178280f2546f397f54054f71ffa64cf692107f6f52234502864c266a58ffcbb`) sets
  `mdim=4682, ntrain=81, nclass=3`. It reads `tryw.txt` with
  `read(16,*) (x(m,n),m=1,mdim), cl(n)`: one row per case, with 4,682 values followed by
  the class.
- Data file: `cc_examples/data/tryw`. The 2004 captures are truncated at 786,432 bytes
  (27 complete rows). The **2005-05-17 capture is complete**:
  <http://web.archive.org/web/20050517155919id_/http://www.stat.berkeley.edu:80/~breiman/RandomForests/cc_examples/data/tryw>,
  2,404,506 bytes, SHA-256
  `9b169380732aa10151ad6e9105cbe43b91eecf3cd0129f90a33891324712783f`. It has 81 rows of
  4,683 whitespace-separated numbers (CRLF). The truncated 2004 capture is a byte prefix of it.
- Written: 81 × 4,683 (`g1`…`g4682`, `class`). Values are unchanged. No gene identifiers
  are in the file, so the columns are positional only.
- Classes: 0 = 29 (rows 1–29), 2 = 9 (rows 30–38), 1 = 43 (rows 39–81).
- Provenance: Dudoit, Fridlyand & Speed (2002, *JASA* 97:77–87, §3.1.1) describe Alizadeh
  et al.'s data as "p = 4,682 genes in n = 81 mRNA samples … 29 cases of B-CLL, 9 cases of
  FL, and 43 cases of DLBCL". The values are log2 Cy5/Cy3 ratios, and missing values were
  imputed by k=5 nearest-neighbour genes (§3.2; mean 6.6% missing per array). Breiman's
  counts match exactly. That gives the inferred mapping **0 = B-CLL, 1 = DLBCL, 2 = FL**;
  the file has no labels, so this rests on the counts alone. Most values have two
  decimals. 28,018 of 379,242 (7.4%) have three or more, which fits imputed averages.
  The rows are **not** standardised (row SDs are about 0.55–0.77), so this is the matrix
  before Dudoit et al.'s §3.3 per-array standardisation.
- Licence/terms: none stated for the data. The site's copyright page covers the Fortran
  code (GPL). The data originate with Alizadeh et al. (2000), *Nature* 403:503–511.
- Not needed, so not downloaded or checked: `spls::lymphoma` (62 × 4,026, as reported)
  and `datamicroarray::alizadeh`. The original source is no longer needed either.

## 6. Default of credit card clients (UCI 350)

- Source: <https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip>,
  which holds `default of credit card clients.xls`. Landing page
  <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>. Licence CC BY 4.0.
  Yeh & Lien (2009), *Expert Systems with Applications*, doi:10.1016/j.eswa.2007.12.020.
- SHA-256: zip `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`;
  xls `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.
- Written: 30,000 × 25. The sheet's first row (`X1`…`X23`, `Y`) was dropped, and its
  second row is the header: `ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0,
  PAY_2…PAY_6, BILL_AMT1…6, PAY_AMT1…6, default payment next month`. `ID` is kept.
  Default 1: 6,636, 0: 23,364. No missing values.
- BILL_AMT1/PAY_AMT1 are September 2005 and …6 is April 2005. PAY_* is repayment status
  (−1 = paid duly, 1–9 = months of delay). Values outside the documented codes are left
  as they are: −2 and 0 in PAY_*, 0, 5 and 6 in EDUCATION, and 0 in MARRIAGE.

## 7. Lalonde NSW, Dehejia–Wahba sample, and PSID-1

- Source: Rajeev Dehejia's data page <https://users.nber.org/~rdehejia/data/.nswdata2.html>,
  files <http://users.nber.org/~rdehejia/data/nsw_dw.dta> and `psid_controls.dta`. The
  `.txt` equivalents `nswre74_treated.txt`, `nswre74_control.txt` and `psid_controls.txt`
  were downloaded for a cross-check. The page states no licence. It says Lalonde gave
  permission to use and publish the data, and asks for citation of Lalonde (1986) and
  Dehejia & Wahba (1999, 2002).
- SHA-256: `nsw_dw.dta` `d1bd2680a1c6f799f1c6d2455bf29633fdf19be01cb19490621c20a560b4e072`;
  `psid_controls.dta` `7beebae8928035d6abf662b994ce77a4fbea7ae6575037ff94e92fe099b43e14`;
  `nswre74_treated.txt` `e7b742fe0ff07a0f45e129b4ff108bb9611cd83d53604732c48a8a0a3e20eda3`;
  `nswre74_control.txt` `a1364cea459d953dc691a667d99194b4ad335d6d550354fe23a5d2dc58d729b5`;
  `psid_controls.txt` `73ffcdd8feb7f965f8e1de9bfdb80a095542c7a95ce7f56b2e2b2a9716e858cc`.
- Written: `lalonde_nsw` 445 × 11 and `lalonde_psid` 2,490 × 11, with columns
  `data_id, treat, age, education, black, hispanic, married, nodegree, re74, re75, re78`.
  Indicators, age and education were cast from Stata float to integer after checking they
  are whole numbers. Earnings are left as float32, so the CSV shows e.g. `9930.046`.
- Counts: NSW treated 185 and control 260; PSID controls 2,490. The `.txt` files have
  185, 260 and 2,490 rows.
- **Experimental benchmark:** mean re78 is 6,349.14 for the treated and 4,554.80 for the
  controls, a difference of **$1,794.34**. The `.txt` files give 1,794.34 as well.

## 8. New Delhi PM2.5 and weather, November 2016 to October 2026

Unlike sections 1–7, these two files are written by a script in the repo,
`../src/fetch_delhi.py`, which downloads the hourly sources into a directory outside
the repo and writes the daily files here. Access date **2026-11-19**. One row per local
(India Standard Time) calendar day, 2016-11-01 to 2026-10-31.

### `delhi_pm25_daily.csv.gz`

- Source: the US Embassy monitor in New Delhi (28.63576 N, 77.22445 E), reported to
  AirNow by the US Department of State, as archived by OpenAQ: location 8118,
  "US Diplomatic Post: New Delhi", sensor 23534, one gzip CSV per day under
  `s3://openaq-data-archive/records/csv.gz/locationid=8118/` (public bucket, no key).
  OpenAQ's archive of this monitor starts on 2016-11-09; AirNow's own files reach back
  further but `dosairnowdata.org` did not resolve on the access date.
- Terms: OpenAQ passes through the original source's terms and asks for attribution to
  both the original provider and OpenAQ (<https://docs.openaq.org/docs/data-policy>).
  AirNow describes embassy data as provisional, not fully verified or validated.
- SHA-256 of the concatenated hourly download (`openaq_8118_hourly_2016-11-01_to_2026-10-31.csv`, 79,677 rows,
  stamped 2016-11-09 23:00 to 2026-11-01 00:00; the script cuts the download at the last
  day, so a later run gives the same file): `481df96f2278d128245c1419131f0ca406982e847dcb0d3dc31fce381390e95b`.
- Hourly cleaning: 6,157 hours coded `-999`, 191 other negative values and 22 zeros are
  missing; 321 hours at exactly 1985 µg/m³ are the instrument's fault code (they come in
  runs, next to ordinary readings) and are missing too. OpenAQ stamps each hour at its
  end (a day's file runs 01:00 to the next 00:00), so an hour counts on the day it
  started. July–November 2023 carries half-hour stamps as well. They get the same
  one-hour shift and are averaged with the hour they land in, so a stamp from 00:30 to
  00:59 counts on the previous day. The archive does not say which half hour these
  stamps cover.
- Columns: `date`, `pm25` (24-hour mean, µg/m³), `pm25_max` (highest hour),
  `n_hours`. A day with fewer than 18 valid hours has no mean (366 days), and 379 days
  have no valid hour at all: 745 days, 20% of the record, without `pm25`. The longest
  gap is 10 May to 21 September 2022 (135 days).

### `delhi_era5_daily.csv.gz`

- Source: ERA5 reanalysis (Hersbach et al. 2020), hourly, from the Open-Meteo historical
  archive API (`https://archive-api.open-meteo.com/v1/archive`, `models=era5`,
  `timezone=Asia/Kolkata`) at the monitor's coordinates. The nearest ERA5 grid point,
  which the API returns, is 28.75 N, 77.25 E, about 13 km north of the monitor.
  Open-Meteo's data are CC BY 4.0; ERA5 is Copernicus Climate Change Service data.
- SHA-256 of the hourly download (`era5_hourly_2016-11-01_to_2026-10-31.csv`):
  `5f83a80fa0d23912a991aa4e4212d327e7bdab84f53b13e80211ff26520deb47`. The request runs
  one day either side of the span (2016-10-31 to 2026-11-01).
- Day boundaries: instantaneous values (temperature, humidity, pressure, wind,
  boundary-layer height) are labelled at their hour and grouped by that hour's date.
  Precipitation and shortwave radiation are totals or means over the hour ending at the
  label, so they are shifted back an hour first, as PM2.5 is.
- Columns (daily aggregates of 24 local hours): `temp_mean`, `temp_max`, `temp_min` (°C, 2 m);
  `rh_mean` (%); `dewpoint_mean` (°C); `pressure_mean` (surface, hPa); `precip_sum` (mm);
  `wind_speed_mean` (km/h, 10 m); `wind_u_mean`, `wind_v_mean` (km/h, the mean of the
  hourly wind vectors; u > 0 blows toward the east, v > 0 toward the north);
  `radiation_sum` (shortwave, kWh/m²); `blh_mean`, `blh_max`, `blh_min` (boundary-layer
  height, m).
- Missing: boundary-layer height is null in the API for every hour from 2024-01-02 to
  2024-06-30 (181 days) under `era5` and `era5_seamless` alike, and the archive serves no
  pressure-level fields to rebuild it from. Part 1 therefore does not use the `blh_*`
  columns. Every other column is complete.
