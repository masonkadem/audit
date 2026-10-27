# Pre-registration: time-of-night shortcut audit of wav2sleep on MESA

Committed before any intervention or control model is run. Any later change to this file is a
deviation and will be listed as such in `report.md`, with its commit.

## Question

Does wav2sleep (`hf://joncarter/wav2sleep`, 4-class: Wake / Light / Deep / REM) use *where in
the night an epoch sits* as a shortcut, instead of (or as well as) the physiology in and around
that epoch?

## Model facts that constrain the design (step 0, measured before anything else)

See `prereg_artifacts/rf.json` (produced by `python -m audit.rf`; summary copied here).

- **Sequence mixer (`SequenceCNN`)**, from the released checkpoint's own `config.yaml`
  (HF snapshot `8b6e46ab3ef6e3945282b81f631933c59db0b205`): 2 residual blocks × 5 dilated conv
  layers, kernel 5, dilations 1, 2, 4, 8, 16, symmetric padding. The repo's default
  `scripts/config/model/wav2sleep.yaml` (kernel 7, 6 dilations) is *not* the released model.
  Each layer reaches (k−1)/2 × d epochs to each side, so the theoretical receptive field is
  **±124 epochs (±62 min); 249 epochs (2.1 h) in total**, non-causal. On the loaded checkpoint,
  the gradient of mixer output at t0 with respect to mixer input is non-zero out to exactly ±124
  epochs and zero beyond. With random inputs, 95% of that gradient mass lies within ±100 epochs.
- **Signal encoders**: receptive field 1531 samples for ECG/PPG and 379 for ABD/THX
  (≈1.5 epochs), i.e. about one neighbouring epoch either side.
- **Global paths that bypass the mixer receptive field**:
  1. every recording is z-scored over the whole 10-h frame (`ParquetDataset._zscore_normalize`);
  2. every encoder conv layer uses `InstanceNorm1d`, whose statistics are pooled over the whole
     sequence.
  Both are invariant to reordering the same content, so they carry no position information under
  relocation, up to seam-local activations. They do change when content is added (offset) or
  truncated, and that change reaches every epoch, so it is a non-positional confound for offsets.
  Full-model check (float64, CPU, deterministic; Gaussian-noise inputs): replacing one epoch at
  lags 125–590, beyond the mixer RF, changes P(stage) at t0 by 0.03–0.06, the same order as
  within the RF. **The whole-model receptive field is therefore the whole night**, through
  InstanceNorm. That path carries no order information.
- **Fixed 10-h frame with visible edges**: preprocessing resamples every night onto 1200 epochs
  and zero-fills after the recording ends, and the mixer zero-pads at both sequence ends. The
  only direct cue to absolute position is therefore *edge exposure*: being within 124 epochs
  (62 min) of the recording start or of the start of the zero tail. Consequences:
  - An epoch that is ≥124 epochs from every seam, and ≥124 epochs from the edges in both the
    original and the altered night, sees identical mixer input after relocation. Its Δp can come
    only from InstanceNorm statistics, so it is expected to be ≈0 *by construction*. This is
    reported as a built-in sanity check ("interior"), not as evidence for P1.
  - Away from edges, a time-of-night shortcut can only be *content-mediated*: inferring "late
    night" from ±62 min of physiological context. Seam-distance curves measure exactly this.

## Data

- **Test**: only the 200 nights in wav2sleep's own MESA test split
  (`src/wav2sleep/data/splits/mesa/test.txt`), preprocessed with wav2sleep's own
  `preprocessing/1_ingest.py` and `2a_create_splits_paper.py`. `audit/run_model.py` refuses any
  file outside that split. MESA train and val nights were used to train wav2sleep and are never
  evaluated.
- **Position prior**: the 4-class labels of the nights in wav2sleep's MESA *train* split
  (1541 listed), read from the NSRR `annotations-events-nsrr` XMLs with wav2sleep's own
  `parse_xml_annotations` (the parser its ingest uses). Nights without an XML on NSRR are listed
  in the report.
- Labels are mapped with wav2sleep's `INTEGER_LABEL_MAPS[4]` (N1+N2 → Light, N3 → Deep).
  Unscored epochs (−1) move with content and are excluded from every metric.

## Definitions

- **Sleep onset**: first scored non-Wake epoch of the night.
- **Position prior (primary, onset-anchored)**: P(stage | ⌊(time since sleep onset)/30 min⌋),
  pooled over training epochs. Pre-onset epochs get negative bins. Bins with fewer than 1000
  training epochs are pooled into the nearest populated bin, and out-of-range lookups clamp.
- **Position prior (secondary, start-anchored)**: P(stage | ⌊(time since recording start)/30 min⌋).
- **Index map**: every intervention is a map `src` such that altered epoch *j* carries the
  signals *and label* of original epoch `src[j]`. Signals inside an epoch are never modified.
  The altered night is rebuilt in the 10-h frame (zero-filled tail, content beyond 10 h
  truncated) and z-scored exactly as wav2sleep's own loader does.
- **Seam**: altered index *j* with `src[j] ≠ src[j−1]+1`. **Seam distance** of an epoch: epochs to
  the nearest seam (0 = adjacent).
- **Δprior** of a content epoch *i* at new position *j*: prior(new bin, computed from the altered
  night's own onset) − prior(old bin), per stage.
- **Δp**: (p_intervention − p_unaltered) − (p_matched-null − p_unaltered) = p_intervention −
  p_matched-null, per stage, at the same content epoch. Softmax probabilities.

## Interventions (step 3)

| Name | Map | Matched null |
|---|---|---|
| `null_literal_{30,60}` | blocks of 30/60 min reassembled in order | — |
| `null_jitter_{30,60}` | in-order blocks, with a real discontinuity at every cut (see below) | — |
| `null_jitter_thirds` | same, cuts at n/3 and 2n/3 | — |
| `null_jitter_offset_{15,30,60}` | same, one cut at the offset length | — |
| `swap_thirds` | last third, middle third, first third | `null_jitter_thirds` |
| `reverse_{30,60}` | 30/60-min blocks in reverse order, each block forward | `null_jitter_{30,60}` |
| `offset_{15,30,60}` | night's own first 15/30/60 min prepended, then the full night | `null_jitter_offset_*` |

**Clarifications of the task specification (decided before any data were seen):**

1. *Literal null splice is the identity.* Cutting a contiguous recording into blocks and
   reassembling them in order gives a bit-identical input, so its Δ is exactly 0 and subtracting
   it removes nothing. It is kept as a determinism check (expected max |Δp| < 1e-5). The null that
   is subtracted is the **jittered null**: at alternating cuts, either the first epoch after the
   cut is deleted or the last epoch before it is duplicated. This puts a genuine waveform
   discontinuity at every cut while keeping cumulative position drift within one epoch. It
   controls for the *signal* seam artefact. It cannot control for the change of physiological
   *context* across a relocation seam; seam-distance curves handle that (point 3).
2. *Offset needs real wake.* `offset_k` is run only on nights whose first k minutes are all scored
   Wake. Excluded nights are counted and reported. Prepended copies are not analysed; the real
   night is analysed at its shifted position.
3. *The seam-exclusion rule removes the block-reversal conditions entirely.* With a one-sided
   RF of 124 epochs (62 min), every epoch of a 30- or 60-min block night is inside the RF of a
   seam (maximum seam distance 15 / 30 min). Block reversals therefore never enter the strict
   (≥124-epoch) analysis. They contribute to the primary analysis only through epochs at
   distance ≥ R, which exist only if R < 30 / 60 epochs.
   Swapped-thirds and offset nights of typical length do have epochs beyond the RF. Therefore:
   - all results are reported as a function of seam distance, in bins of
     [0,5), [5,15), [15,30), [30,62), [62,120), ≥120 min;
   - the **primary** analysis uses epochs with seam distance ≥ **R**, the *effective* receptive
     field: the median, over unaltered test nights with all four signals, of the radius holding
     95% of |∂ mixer-output[t0] / ∂ mixer-input[t]| (t0 = middle of the scored night).
     R is computed by `python -m audit.rf --data …` on unaltered nights *before* any intervention
     is run, and recorded in `results/rf_real.json`;
   - a strict sensitivity analysis uses seam distance ≥124 epochs (the theoretical RF), split
     into *edge-exposed* epochs (within 124 epochs of either night's edges) and *interior* epochs
     (the by-construction sanity check above).
4. *Offset cannot move the onset-anchored prior.* Prepending wake moves sleep onset with the
   content, so time since onset, and hence onset-anchored Δprior, is exactly 0 for every epoch.
   Offsets are therefore analysed against the **start-anchored** prior, as a separate
   pre-registered test of the lights-out anchor that the architecture most directly supports
   (see model facts above).

## Regression (step 4)

For each model × input subset × stage: OLS with intercept of Δp_stage on Δprior_stage, pooled
over epochs. 95% CI: percentile bootstrap over **participants** (2000 resamples). Reported for:

- **Primary**: relocations pooled (`swap_thirds`, `reverse_30`, `reverse_60`), onset-anchored
  prior, seam distance ≥ R.
- **Primary (offset)**: offsets pooled, start-anchored prior, seam distance ≥ R.
- **Secondary**: each condition × seam-distance bin × both anchors; the uncorrected Δ
  (p_intervention − p_unaltered); the mean |Δ| of every null.
- **Accuracy and Cohen's κ** of every condition (including unaltered and nulls) against the moved
  labels: pooled across epochs, plus mean per-night accuracy.

## Secondary robustness check: alternative preprocessing

A previously extracted copy of the same MESA nights exists on the analysis machine
(`mesa_ppg_with_labels.h5`, `mesa_real_ecg.h5`). It was band-pass filtered (PPG: Chebyshev II;
ECG: 4th-order Butterworth), clipped at ±3 SD and z-scored, which differs from wav2sleep's
ingest, and it has no ABD/THX. The PPG-only and ECG-only analyses are repeated on the same 200
test nights from this copy. This is reported separately and does not affect the P1/P2 verdict.
Its purpose is to show whether the slopes depend on preprocessing.

## Input subsets (step 5)

PPG only; ECG only; all signals (ECG, PPG, ABD, THX). Absent signals are passed as −inf, wav2sleep's
own missing-modality convention. Nights lacking a signal are skipped for subsets that need it.

## Controls (step 6), on the same test nights

The existing windowed sleep-staging features from the MESA actigraphy-overlap pipeline are used:
31 HRV features, plus RRV and EDR features over 150/210/270-s windows (151 features, no
actigraphy). They are available for 98 of the 200 test nights. Two LightGBM models are trained
on wav2sleep's MESA train split, with early stopping on its val split:

- **Negative control**: windowed features only (no time).
- **Positive control**: the same plus an explicit time-since-sleep-onset feature (minutes), which
  is recomputed from the altered night's own labels after each intervention.

Interventions permute feature rows with the same index maps. Windows that would straddle a seam
are not recomputed, so control epochs within 5 epochs of a seam are excluded (R = 5 for controls).
wav2sleep is also reported restricted to these 98 nights.

**Assay validity (checked before reading wav2sleep results):** the assay is valid only if the
positive control's primary 95% CI lower bound exceeds +0.10 for at least one stage, and the
negative control satisfies the equivalence criterion below. If the assay is invalid, wav2sleep's
results are reported but neither prediction is marked confirmed.

### Amendment 1 (2026-10-27), committed before any wav2sleep intervention was run

At this point the negative and onset-time positive controls had been run, and R had been
computed on unaltered nights. No wav2sleep intervention had been run.

The onset-time positive control cannot validate the offset test: an offset leaves time since
sleep onset unchanged, so that control's offset slope is ≈0 by construction (as observed). A
third control is added:

- **Positive control (offset)**: the same windowed features plus an explicit
  time-since-recording-start feature (minutes since the first epoch of the feature record),
  recomputed from each epoch's position in the altered night.

**Offset assay validity**: the offset test (start-anchored prior) is valid only if this
control's `primary_offset` 95% CI lower bound exceeds +0.10 for at least one stage. If it
does not, wav2sleep's offset results are reported but cannot be read as evidence of
faithfulness. The relocation verdict and its validity rule are unchanged.

## Predictions and decision rules

Evaluated on the **primary** relocation slopes, all signals; repeated per input subset.

- **P1 (faithful): slope = 0.** *Confirmed* if, for every stage, the 95% CI lies within
  [−0.10, +0.10] (equivalence margin: the model passes through less than 10% of the change in
  the position prior).
- **P2 (shortcut): slope > 0, largest for Deep and REM.** *Confirmed* if the 95% CI lower bound is
  > 0 for both Deep and REM, *and* Deep and REM have the two largest slope point estimates.
- Each prediction is otherwise marked **not confirmed**. Both can be not confirmed (inconclusive).
- The offset primary slopes (start anchor) are evaluated with the same two rules and reported as a
  separate line; they do not change the P1/P2 verdict.

## Outputs

`report.md`: verdict per prediction; primary and secondary slopes with CIs; controls; accuracy
and κ; a stage-by-position figure (training prior, and mean predicted vs observed stage
probability by time since onset on unaltered test nights); and a slope plot by seam distance.

Per-participant data and outputs are kept out of version control (NSRR data-use agreement).
