# Source notes: Breiman (2001) and discussion

Read in full on 2026-11-18 from the CMU journal-club copy
(<https://www.stat.cmu.edu/~ryantibs/journalclub/breiman_2001.pdf>), which reproduces the
*Statistical Science* 16(3) pagination, 199–231; Project Euclid record
<https://doi.org/10.1214/ss/1009213726>. Page numbers below are journal pages.

Every number the series quotes from the paper is listed here with its page. Direct quotes
are kept under 15 words and to one or two per source; everything else is paraphrase.
"B" = Breiman's paper; "C", "E", "H", "P" = the Cox, Efron, Hoadley, Parzen comments;
"R" = the rejoinder.

## Breiman, main paper (pp. 199–215)

### Section 1: the two cultures (p. 199)

- Black box: inputs x go in, responses y come out, nature inside.
- Two goals, named **Prediction** and **Information** (how nature associates y with x).
- Data modeling culture: assume a stochastic model, e.g. y = f(predictors, noise,
  parameters); validate by yes–no goodness-of-fit tests and residual examination.
  Estimated population **98% of all statisticians** (p. 199).
- Algorithmic modeling culture: inside of the box complex and unknown; find a function
  f(x) that predicts y; validate by predictive accuracy. **2% of statisticians**, many in
  other fields (p. 199).
- Claims the data-model focus has led to irrelevant theory and questionable conclusions,
  kept statisticians from more suitable algorithmic models and from new problems
  (pp. 199–200).

### Section 2: road map (p. 200)

- Seven years as an academic probabilist, then thirteen years of free-lance consulting,
  joined Berkeley Statistics in 1980.
- Three lessons named Rashomon (multiplicity of good models), Occam (simplicity vs
  accuracy), Bellman (dimensionality, curse or blessing).

### Section 3: consulting projects (pp. 200–201)

- Ozone (3.1, pp. 200–201): LA Basin, mid-1970s EPA project to predict ozone 12 hours
  ahead. Over **450 meteorological variables**, daily and hourly, for **seven years**;
  predictors include several days back. First **five years train, last two test**. Large
  linear regressions, variable selection, then quadratic terms and interactions, then
  selection again. Outcome: a failure, the **false alarm rate** too high (p. 201).
- Chlorine (3.2, p. 201): mass spectra of **30,000 compounds**, split **25,000 train /
  5,000 test**; LDA and QDA struggled with variable dimensionality; a tree on **1,500
  yes–no questions** built from domain knowledge gave **95% accuracy** on both chlorines
  and non-chlorines (citing Breiman, Friedman, Olshen and Stone, 1984).
- Perceptions (3.3, p. 201): (a) find a good solution; (b) live with the data first;
  (c) search for a model, algorithmic or data; (d) predictive accuracy on test sets is
  the criterion; (e) computers are indispensable.

### Section 4: return to the university (p. 202)

- Colleague's first question about trees: "What's the model for the data?" (quote, p. 202).
- Places himself at the "very low end" of the spectrum on the usefulness of Annals theory.

### Section 5: the use of data models (pp. 202–204)

- Core point (p. 202): conclusions drawn from a fitted model are about the model's
  mechanism, not nature's; a poor emulation gives possibly wrong conclusions.
- 5.1 (pp. 202–203): linear regression with 5% significance; R² as the only fit check;
  faculty-salary gender study with 25 covariates as the example of design questions
  ignored; Mosteller & Tukey (1977) and Freedman as rare critics.
- 5.2 (p. 203): **a simulated regression problem in seven dimensions with a controlled
  amount of nonlinearity**; standard goodness-of-fit tests did not reject linearity
  until the nonlinearity was extreme. No further details of the setup are given (no n,
  no form of the nonlinearity, no named tests). Cites Bickel, Ritov and Stoker (2001,
  unpublished): goodness-of-fit tests have little power unless the alternative's direction
  is specified, so omnibus tests have little power. Tests are not applicable after the
  model is tinkered with on the data. Cleveland (Berkeley seminar, 1993): residual
  analysis cannot uncover lack of fit beyond **four to five dimensions**.
- 5.3 (pp. 203–204): multiplicity of data models: two statisticians, each satisfied with
  fit, can reach different conclusions; yes–no fit checks cannot rank them. Cox model and
  logistic regression as de facto publication standards in medicine.
- 5.4 (p. 204): predictive accuracy as the obvious measure of how well the model box
  emulates nature's box; cross-validation (Stone 1974) or a test set; rarely reported in
  JASA.

### Section 6: limitations of data models (pp. 204–205)

- Multivariate tools "frozen" at discriminant analysis, logistic regression, linear
  regression; the hammer-and-nail saying; Bayesian MCMC models growing more cumbersome.

### Section 7: algorithmic modeling (pp. 205–206)

- New community from mid-1980s around neural nets and trees (p. 205).
- Theory assumes only i.i.d. draws from an unknown distribution (p. 205).
- Vapnik's bounds led to SVMs; Breiman (2000) "My kingdom for some good theory" (p. 205).

### Section 8: Rashomon (p. 206)

- Named after the Japanese film: four witnesses, same facts, different stories.
- Rashomon Effect: many different f(x) in a class with about the same minimum error.
- Subset example: **30 variables, best five-variable regressions, about 140,000 subsets**;
  generally many five-variable equations within **1.0% of the lowest RSS** (citing
  Breiman 1996a); same with test error. Three illustrative equations (Pictures 1–3) on
  disjoint variable sets. (C(30,5) = 142,506.)
- Trees: perturbing the training set by removing a random **2–3%** gives a quite
  different tree with almost the same test error.
- Neural net: **100 runs** of a small net on simple **three-dimensional** data with small
  random initial weights gave **32 distinct minima**, each a different picture, about
  equal test error.
- Instability (Breiman 1996a): competing models crowded together; small perturbations
  skip between them. Deleting covariates in logistic or Cox models, e.g. **15 to 4**,
  makes the model unstable.
- Arena et al. (2000) bagged logistic regressions on toxic/non-toxic chemicals with best
  subset selection **from 15 to 4** covariates; bagged model significantly more accurate
  on a test set, and more stable. Reference (p. 215) is a workshop presentation
  (Second Indo-U.S. Workshop on Mathematical Chemistry), not a journal paper.

### Section 9: Occam (pp. 206–208)

- Accuracy and simplicity in conflict in prediction (p. 206).
- Colorado court-delay tree and the District N anecdote (pp. 206–207). Trees: A+ on
  interpretability, B on prediction (p. 207).
- Forests: perturb training set, grow tree, repeat, vote; mentions bagging, boosting,
  arcing, additive logistic regression; random forests his preferred method (p. 207).
- **Table 1 (p. 207)**: Cancer 699/—/9 vars/2 classes; Ionosphere 351/—/34/2;
  Diabetes 768/—/8/2; Glass 214/—/9/6; Soybean 683/—/35/19; Letters 15,000/5,000/16/26;
  Satellite 4,435/2,000/36/6; Shuttle 43,500/14,500/9/7; DNA 2,000/1,186/60/3;
  Digit 7,291/2,007/256/10.
- **Protocol (p. 207)**: five smaller sets, leave out a random 10%, fit CART and forest on
  the other 90%, error on the 10%, **repeated 100 times** and averaged. Larger sets used
  their separate test sets.
- **Table 2 (p. 207), test misclassification %, Forest vs Single tree**: Breast cancer
  2.9 / 5.9; Ionosphere 5.5 / 11.2; Diabetes 24.2 / 25.3; Glass 22.0 / 30.4; Soybean
  5.7 / 8.6; Letters 3.4 / 12.4; Satellite 8.6 / 14.8; Shuttle (×10³) 7.0 / 62.0;
  DNA 3.9 / 6.2; Digit 6.2 / 17.1. The Shuttle row is the error in percent multiplied by
  10³, i.e. 0.007% vs 0.062%.
- Statlog (Michie et al. 1994, 18 classifiers): on the four Statlog sets with separate
  test sets the forest ranks 1, 1, 1, 1; next best average rank **7.3** (p. 208).
- Digit is the 16×16 ZIP-code data; a handcrafted net got **5.1%** vs **6.2%** for a
  standard forest run (p. 208).
- Forests: A+ predictors, F on interpretability. The Occam dilemma: accuracy generally
  requires more complex methods; go for accuracy first, then try to understand why
  (p. 208).

### Section 10: Bellman (pp. 208–209)

- Refers to Bellman's phrase "the curse of dimensionality" (p. 208). Old recipe: reduce to
  few features; Meisel (1972) quoted that features must be few.
- Proposal: increase dimensionality by adding many functions of the predictors (p. 208).
- Amit–Geman shape forest (p. 208): NIST 1992 competition, **223,000** handwritten
  numerals by **over 2,000** writers; thousands of geometric features; shallow trees with
  **100 features chosen at random** per node; **over 1,000 trees**; **100,000 train /
  50,000 test**; **0.7%** test error.
- SVM (pp. 208–209): optimal separating hyperplane; add all quadratic, then cubic
  monomials; 30 variables give **about 40,000 features** up to degree four. ZIP Code:
  separation with fourth-degree monomials, test error **4.1%**; large NIST subset,
  **1.1%**. Vapnik bound: E(generalization error) ≤ E(number of support vectors)/(N − 1).
  Soft-margin version when separation needs too many support vectors (p. 209).

### Section 11: information from a black box (pp. 209–213)

- Two black boxes, ours only slightly less inscrutable than nature's (p. 209).
- Doctors "can interpret logistic regression" anecdote (p. 209).
- "The goal is not interpretability, but accurate information." (p. 210, quote)
- **Hepatitis (11.1, pp. 210–212)**: 155 patients, 19 covariates, survival; UCI,
  contributed by Gail Gong; previously analysed by Diaconis and Efron (1983) and
  Cestnik, Kononenko and Bratko (1987), lowest reported error 17% in the latter.
  - Peter Gregory's analysis: important variables **6, 12, 14, 19**, estimated 20% error.
  - Diaconis & Efron: **500 bootstrap samples**; none of the four originally selected
    variables chosen in more than **60%** of samples (their quote, p. 210).
  - Logistic regression error **17.4%** from **100 runs** of random 10% holdouts (p. 210).
  - Fig. 1: standardized coefficients; **variables 7 and 11** look most important; LR on
    those two alone: **22.9%** (p. 210).
  - About 4,000 four-variable subsets (C(19,4) = 3,876) (p. 210).
  - Random forest error **12.3%**, "almost a 30% reduction" (p. 211). (12.3/17.4 = 0.707.)
  - Permutation importance defined on out-of-bag cases, percent increase in error
    (p. 211). The text says the measure is "shown in Figure 1" but the importance plot is
    Fig. 2 (a typo in the paper).
  - RF singles out **variables 12 and 17**; each alone in RF: **14.3%**; both together no
    better (p. 211).
  - Fig. 3: smoothed probability of death vs standardized variable, almost linear and
    similar for 12 and 17 (p. 211).
  - 12 and 17 highly correlated; LR rerun deleting one or the other: little change
    (p. 212).
  - Permutation importance for LR: 12 and 17 not among its top 3 (p. 212).
  - LR on single variables: **12 gives 15.7%**, **17 gives 19.3%** (p. 212).
  - Diagnosis: 12 and 17 are surrogates, so each appears in under 50% of bootstrap
    selections; of the ten samples Diaconis–Efron list, **12 or 17 appears in seven**
    (p. 212).
  - Remarks (p. 213): **32 deaths, 123 survivors**; all-survivor baseline **20.6%**; LR's
    17.4% extracts little information, which may explain its failure to find the
    important variables.
  - Diaconis & Efron quoted: unwise to fit 19 variables with only 155 data points (p. 213).
  - Numbering: Efron's comment (p. 219) says Breiman's 12 or 17 are **13 or 18 in Efron
    and Gong's numbering**, so Breiman numbers covariates only (class excluded) and
    Efron & Gong count the class as attribute 1, as `hepatitis.names` does.
- **Bupa liver (11.2, pp. 212–213)**: 345 patients, covariates 1 mcv, 2 alkphos, 3 sgpt,
  4 sgot, 5 gammagt, 6 drinks; described as two classes by severity of liver
  malfunction, class two severe (p. 212). RF error **28%**; importance: blood tests **3
  and 5** most important, then **4** (p. 213). RF proximity clustering found **two
  clusters in class two**: high on tests 3, 4, 5 vs low on them (Fig. 5, p. 213).
- **Lymphoma microarray (11.3, p. 213)**: three classes, **81 samples, 4,682 genes**, no
  variable selection; points to Dudoit, Fridlyand and Speed (2000, tech report). Error
  "was low" (no number given). Fig. 6 gene importance.
- Lessons (p. 214): higher accuracy goes with more reliable information; algorithmic
  models can give better accuracy and better information.

### Section 12: final remarks (p. 214)

- Not against data models per se; emphasis on the problem and the data.
- Dempster (1998) on the 1990 census adjustment as an example of faith in modeling.
- Astronomy and microarrays as statistical problems where data models fit poorly.

## D. R. Cox, comment (pp. 216–218)

- Calls the paper's picture of mainstream thinking partly a caricature, but a
  thought-provoking one (p. 216). Short quote: "based in part on a caricature" (p. 216).
- Agrees that work without an explicit probabilistic base belongs in statistics (p. 216).
- Prefers to start from an issue, question or hypothesis rather than data (p. 216).
- Data quality: meaning of the data, ascertainment bias, measurement error, missingness,
  time dependence; hard to handle these without probabilistic modeling (p. 216).
- Accepts that a directly empirical approach is sometimes better: short-term economic
  forecasting, real-time flood forecasting; issues then are stability and recalibration
  (p. 216).
- But much prediction is under quite different conditions from the data: the v-CJD
  epidemic in the UK; effect on US cancer incidence of cutting medical X-rays by 10%;
  predicting consequences of something only indirectly addressed by the data (p. 216).
  Then prediction without understanding of process becomes more tentative (p. 217).
- Mainstream strategy cannot be read from journals, which illustrate technique (p. 217).
- His summary of good practice: formal models useful, often almost essential; inference
  tools as guides, often lower bounds to real uncertainty; automatic model/variable
  selection to be shunned; unfocused tests of model adequacy rarely helpful (p. 217).
- Limits of conclusions lie more in data quality and study design than in analysis
  (p. 217).
- Cox & Wermuth (1996) requirements for a model: link to background knowledge, pointer to
  a generating process, primary parameters with subject-matter meaning, haphazard aspects
  specified for precision, adequate fit; fit is not the primary basis for model choice
  (p. 217). Aim: data descriptions that are potentially causal (p. 218).
- Breiman's attitude to underlying processes "rather defeatist"; Beveridge (1952) on false
  hypotheses that still led to progress (p. 218).
- Aphorism (perhaps Kelvin): better a rough answer to the right question than an exact
  answer to the wrong one (p. 218).
- On Vapnik and complete separation: separation likely with many variables; bootstrap and
  cross-validation may give a misleading illusion of stability (p. 218).
- Disagrees with the main thrust but found it stimulating (p. 218).

## Brad Efron, comment (pp. 218–219)

- First glance, the paper argues against parsimony and for black boxes "with lots of
  knobs to twiddle"; at second glance still (p. 218).
- Twentieth century as "100 years of unbiasedness" (p. 218); the theory needs a high
  signal-to-noise and n-to-parameters ratio; good design enforces it (p. 218).
- New algorithms with more knobs than data points must rely on some form of biased
  estimation (regularization, pruning, SVMs) (p. 219).
- Rule 1: new methods always look better than old ones (honest simulation comparisons are
  hard). Rule 2: complicated methods are harder to criticize than simple ones (p. 219).
- Prediction culture at Stanford bigger than 2%; estimation and testing are a form of
  prediction; prediction alone only occasionally sufficient; Gregory's study also aimed
  at understanding hepatitis; most surveys aim at causal factors (p. 219).
- Hepatitis: first analysed in Gail Gong's 1982 thesis; fuller account in Efron and Gong
  (1983). Variables 12 or 17 (**13 or 18** in their numbering) appeared as important in
  **60%** of bootstrap simulations vs **59% for variable 19**, the most for any single
  variable (p. 219).
- Asks in what sense a variable is important; points to Efron & Tibshirani (1998)
  (p. 219).
- Algorithmic prediction is a positive development, not an argument against data
  modeling, as splines are not against polynomials; the point of science is to open black
  boxes (p. 219). Short quote: "born-again black boxist" (p. 219).

## Bruce Hoadley, comment (pp. 220–224)

- Fair, Isaac credit scoring since the 1960s as an algorithmic culture built by engineers
  and OR people (p. 220).
- Behaviour scoring toy: 12 months of bills and payments, x of dimension **24**; y =
  no severe delinquency in next 6 months (p. 220).
- Segmented scorecard: characteristics → segments (leaves of a shallow-to-medium tree) →
  scorecard per segment, linear in binned attribute dummies. **About 100
  characteristics × 10 attributes × 10 segments = 10,000 features** (p. 220).
- A scorecard is a GAM with bin smoothing plus score engineering (monotone constraints,
  shrinkage) (p. 221); fit by maximizing divergence, a two-group separation measure
  (p. 221); validation only on test samples (p. 221). Disputes "primitive" (p. 221).
- Large margin ↔ large divergence under his definition (p. 221).
- Overfitting the test sample: the test sample is the current population; things change;
  redevelopment every 18–24 months (pp. 221–222).
- **Fat Scorecard (p. 222)**: logistic regression on attribute dummies, **36**
  characteristics available; all models with **27 to 36** characteristics had the same
  test performance (he calls this Rashomon); chose **27**; **162** score weights with
  p-values **0.0001 to 0.984**, only one below 0.05; no interaction adjustments helped; a
  neural net with **2,000** connection weights was **slightly outperformed** by the Fat
  Scorecard; three further test samples **one, two and three years later**, the Fat
  Scorecard beat the thinner scorecard on all four.
- Discriminant analysis on dummy variables works as an algorithmic model despite violated
  assumptions (p. 222).
- Solution to the Occam dilemma: segmented scorecards are interpretable and accurate;
  "A++" for the shallow tree (p. 223). "Ping-Pong theorem" (p. 223).
- Tuning dilemma: peeking at the test sample biases test or CV results (p. 223).
- Generalizing generalization error: optimize dollars saved, not misclassification
  (fraud example, 0.5% of transactions flagged) (p. 223).
- Measuring importance: an equally accurate black box excluding x1 exists, in which x1 is
  unimportant (p. 223). Build models where no variable dominates.
- Summary: accuracy without the Occam dilemma via medium trees with GAMs in the leaves;
  data-modeling tools work if you ignore textbook advice, embrace dimensionality, use
  constraints, regularize, validate (p. 224).

## Emanuel Parzen, comment (pp. 224–226)

- Prediction vs information recast as **"management" vs "science"**: profit and short-run
  decisions vs truth and long-run understanding (p. 224). Student's t-test as a Guinness
  management tool.
- Wrong model = "the right answer to the wrong question", the error of the third kind
  (p. 224).
- Tests worth running routinely: bivariate dependence among inputs (proxies, unstable
  models) and two-sample conditional clustering (pp. 224–225).
- Statistics has many cultures, not two; "statistical methods mining"; SIEVE stages;
  MacKay & Oldford's PPDAC (p. 225).
- Quantile culture: quantile and comparison density functions, conditional quantiles,
  high-dimensional comparison densities (pp. 225–226).

## Breiman, rejoinder (pp. 226–231)

- Parzen and Hoadley "more or less in agreement", Efron "serious reservations", Cox in
  "downright disagreement" (p. 226).
- Cox: agrees on starting with a question, adding prediction of future events (p. 226).
- C1: data models useful when the mechanism's science is known, or when data are too few
  to answer complex questions; quotes Cox's v-CJD/X-ray passage and replies "I agree."
  (p. 227).
- C2: says Cox justifies data models to the exclusion of others; Cox & Wermuth's diabetes
  example (68 patients, 7 covariates, R² = 0.34) has no predictive accuracy computed
  (p. 227).
- C4: be pragmatic; 1990 census adjustment as error analysis no model could resolve;
  astronomy (two billion objects, 100+ terabytes), microarrays, fMRI (pp. 227–228).
- C5: "mainstream" is definitional; by numbers his version prevails **at least 100 to 1**
  (SAS licences, canned packages) (p. 228). CART and RF as general tools (p. 229).
- E1: science moves from simple to complex (relativity, quantum mechanics); extract
  information from the most accurate model (p. 229).
- E2: a model barely better than baseline cannot be trusted to identify causal factors;
  the more accurate, the more faith in the variables it fingers (p. 229).
- E3: importance has always been defined operationally; his definition is predictive;
  correlated pairs: deleting one does not hurt, deleting both may; RF spots both. No
  satisfactory theoretical definition; it depends on dependencies among inputs; "The
  problem begs for research." (p. 230)
- E4: RF has one tuning parameter, insensitive over a wide range; SVMs one or two
  (p. 230). NIPS 1992 conversion to neural nets (p. 230).
- Hoadley: agrees models must be updated with new-context data; Fair, Isaac an exception
  on interpretability; neural nets explained to clients by fitting trees to their inputs
  and outputs (pp. 230–231).
- Parzen: keeps two cultures because the line is clear: modeling the inside of the box
  or not; Bayesians belong to data modeling (p. 231).
- Closing: the danger is defining the field by familiar tools and problems (p. 231).

## Things the paper does not specify (reproduction gaps)

- Software, tree count, `mtry`, and missing-value handling for Table 2 and Section 11.
  (The RF paper of 2001 and the Fortran code are the best guide.)
- The seven-dimensional goodness-of-fit simulation: no setup given.
- The 30-variable Rashomon example: data not given; the three equations are
  illustrations, cited to Breiman 1996a.
- The neural net experiment: data, architecture and criterion for "distinct minima" not
  given.
- How missing values in hepatitis were handled for logistic regression.
- The lymphoma error rate.
