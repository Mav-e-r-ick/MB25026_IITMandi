# Dataset Audit — Glucose Forecasting Digital Twin (Phase 1)

**Audit date:** 14 October 2026 · **Status:** Draft for team decision · **Scope:** public / legitimately accessible CGM + clinical datasets for a 2-hour glucose-excursion Digital Twin.

Legend: **[V]** verified on the official page or primary publication during this audit · **[S]** from a secondary source (another paper citing the dataset) · **[U]** unknown / not verifiable — must be checked before relying on it.

---

## Update, 14 Oct 2026 (evening)

- **Official data rule read on the Unstop page:** "Teams are strictly required to use anonymized, open-source, or
  synthetic datasets for training their models." CGMacros and ShanghaiT2DM qualify. Submission = public GitHub
  repo; folder name "Team Name_College Name"; README must list team, title, problem, stack, ≥20-min video link,
  open-source licence, architecture diagram (PDF/PPT) and presentation (PDF/PPT). Deadline 20 Oct 2026, 7:00 PM IST.
- **CGMacros paper is now published:** Das A et al., *Sci Data* 12, 1557 (2025), doi:10.1038/s41597-025-05851-7.
- **Access route used:** PhysioNet was unreachable from the build environment, so the harmonised CGMacros Dexcom
  and ShanghaiT2DM files were taken from the Glucose-ML GitHub repository. These include age, sex, race/ethnicity,
  HbA1c and diabetes status, but **not meals, Fitbit or the laboratory panel**, which need the PhysioNet download.
- **Interpolation finding:** CGMacros Dexcom values are linearly interpolated to 1 minute. The real 5-min readings
  were recovered (125,819 readings, 45 participants, mean missingness 2.7%).

## 0. Three findings that change the plan

1. **Deadline is ~6 days away.** The Unstop listing shows the *Prototype & Code Submission* stage running 18 Sep → **20 Oct 2026, 13:30 UTC (= 19:00 IST)**, and a "participate before **18 Oct 2026, 19:00 IST**" cut-off for joining. [V] https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873
   The master plan (8 phases, deep learning, AI-READI, full ablation suite) is not achievable in 6 days. The scope below is cut to what is defensible by 20 Oct.

2. **The challenge's data rules must be read before any download.** The Unstop page renders client-side and could not be read programmatically. Several *other teams'* public repos describe the rules as "sandbox rules" requiring synthetic data, or list "only anonymised, open or synthetic data" as a checklist item. [S — competitor READMEs, not official]
   - https://github.com/Physics0070/Vicimus_Vishwakarma-Institute-of-Technology
   - https://github.com/Varshith2403315/TeamDrafts_IITGoa
   - https://github.com/shashi-bhushan-27/digitalTwinChallenge
   **Decision gate:** if the official rule allows "anonymised/open" data → the strategy below stands. If it is "synthetic only" → real open data is used *offline only* to calibrate and validate a synthetic generator, and every artefact in the repo/demo is synthetic.

3. **Name collision.** A public submission named **"GlucoTwin India"** (Team Vicimus, VIT) already exists with the same pitch — hybrid physiological + ML twin forecasting T2D glucose excursions 2 h ahead. [V — repo above] Recommend renaming (e.g. a name that signals the differentiator: *real open-data validation + uncertainty*).

---

## 1. Candidate datasets — detail

### 1.1 CGMacros (PhysioNet) — **RECOMMENDED PRIMARY**
- **URL:** https://physionet.org/content/cgmacros/1.0.0/ · Code: https://github.com/PSI-TAMU/CGMacros
- **Citation / DOI:** Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. PhysioNet v1.0.0 (2025). doi:10.13026/3z8q-x658 [V]. Companion paper listed as *Scientific Data (under review)* on the PhysioNet page [V] — check whether it has since been published.
- **Custodian:** Texas A&M (PSI lab) + Sansum Diabetes Research Institute, Santa Barbara, CA; hosted by PhysioNet / MIT LCP. IRB: Advarra Pro00049227; NCT04991142 [V]
- **Participants:** 45 — 15 no diabetes (HbA1c < 5.7%), 16 prediabetes (5.7–6.4%), 14 T2D (> 6.4%); ages 18–69; BMI 21–46; recruited 2021–2024 [V]
- **CGM:** Dexcom G6 Pro (5-min) on abdomen **and** Abbott Libre Pro (15-min) on arm, both blinded; ~10 days per person [V]. Main files are on a 1-minute grid [V] → CGM values are sparse on that grid and must be re-aligned.
- **Clinical profile (static):** age, gender, ethnicity, height, weight, BMI, HbA1c, fasting glucose, fasting insulin, triglycerides, cholesterol, HDL, non-HDL, LDL, VLDL, 3 finger-stick glucose values, gut microbiome (1,979 taxa indicators) and 22 Viome gut-health scores [V]
- **Dynamic inputs:** meal timestamps (from photos), calories, carbohydrate, protein, fat, fibre, meal type; Fitbit Sense readings [V]. Exact Fitbit columns (heart rate? METs? steps?) **[U — read DataDictionary_CGMacros-00X.csv]**
- **Medication / insulin:** not listed **[U]** · **Sleep:** not listed **[U]**
- **Paired static + dynamic for the same person:** **Yes** [V] — this is the strongest feature for a Digital Twin.
- **Real / synthetic:** Real, de-identified; dates shifted by 365–720 days [V]
- **Access:** Open — anyone, subject to licence [V]
- **Licence:** **CC BY-NC-SA 4.0** [V] → non-commercial; attribution; derivatives of the data must be shared under the same licence.
- **Size:** 627 MB ZIP (mostly meal photos) [V]
- **Feasible by deadline:** Yes.
- **Role:** Primary training + leave-participant-out evaluation; source of virtual patients for the replay demo (if rules allow real data).
- **Risks:** small n (14 T2D); US-only, not Indian; semi-standardised meals (protein-shake breakfasts with varied macros, fast-casual-restaurant lunches, free dinners) [V] — meal responses may not transfer to Indian diets; blinded CGM means no behavioural feedback (good for physiology, unlike real use); healthy participants will rarely cross 180 mg/dL → event class imbalance concentrated in T2D; participant folders numbered 1–49 for 45 people [V] → expect gaps.

### 1.2 ShanghaiT2DM (Figshare) — **RECOMMENDED BACKUP / EXTERNAL VALIDATION**
- **URL:** https://doi.org/10.6084/m9.figshare.20444397 · Paper: Zhao Q et al., *Sci Data* 10, 2023. doi:10.1038/s41597-023-01940-7 [V]
- **Custodian:** Shanghai East Hospital & Shanghai Fourth People's Hospital (Tongji University), DiaDRIL registry [V]
- **Participants:** 100 T2D (plus 12 T1D in the sibling ShanghaiT1DM set) [V]; 109 recording files, 112,475 CGM readings [V]; age 22–97, 44 M / 56 F [S — arXiv 2402.16230]
- **CGM:** FreeStyle Libre H, 15-min, 3–14 days [V/S]
- **Clinical profile:** clinical characteristics, laboratory measurements, medications [V]
- **Dynamic inputs:** dietary records (food names + weights, in Chinese) [S]; medication/insulin entries [S]; **no activity / exercise** [S]
- **Paired static + dynamic:** Yes
- **Access:** Open download
- **Licence:** **No explicit licence attached** — noted by the Glucose-ML authors [S — arXiv 2507.14077]. Legally, absence of a licence ≠ permission to redistribute. → Use for research evaluation with citation; **do not commit any rows or derived row-level files to GitHub**. (Not legal advice.)
- **Size:** small (spreadsheets) **[U — exact MB]**
- **Feasible by deadline:** Yes for a glucose-only external test; meal parsing (Chinese food logs → carbohydrate) is probably **not** feasible in time.
- **Role:** External T2D validation of the glucose-only model (different country, device, sampling); larger T2D n than CGMacros.
- **Risks:** 15-min sampling (vs 5-min) → must evaluate at 15-min resolution; different device; licence ambiguity; diet logs not machine-ready.

### 1.3 AI-READI (FAIRhub) — strongest in principle, **not feasible by 20 Oct**
- **URL:** https://fairhub.io/datasets/2 · https://aireadi.org/goals/data-sharing
- **Paper:** AI-READI Consortium, *Nat Metab* 2024. doi:10.1038/s42255-024-01165-x [S]
- **Participants:** 1,067 across three US sites; four groups (no diabetes, prediabetes/lifestyle, T2D oral meds, T2D insulin) [S]. After filtering, papers report ~895–896 usable: ~322 healthy, 207 prediabetes, 258 T2D-oral, 108 T2D-insulin [S — arXiv 2509.18457, 2606.12699]
- **Modalities:** Dexcom G6 (5-min, 10 days), Garmin Vivosmart 5 (HR, stress, activity, respiration), environmental sensor, surveys, clinical labs, retinal imaging [S]
- **Access:** Login via CILogon; attest T2D-related research; show research-ethics training. Public set strips PHI plus sex and race/ethnicity; controlled set needs extra steps [V/S]. Full release ~2 TB / 165,051 files [S]. Some papers describe v2.0.0 as controlled-access [S].
- **Licence:** custom AI-READI data licence, doi:10.5281/zenodo.10642459 [V]
- **Flag:** the AI-READI site footer states the repository is under review for potential modification in compliance with Administration directives [V] → access terms may change.
- **Verdict:** Best T2D sample size + medication groups + wearables, but approval/training/download cannot be guaranteed inside 6 days. Mention as the scale-up path in "Future work".

### 1.4 BIG IDEAs Lab Glycemic Variability & Wearable Device Data (PhysioNet)
- **URL:** https://physionet.org/content/big-ideas-glycemic-wearable/ (v1.1.3) [V]
- **Participants:** 16 (post-menopausal women 35–65, normoglycaemia/prediabetes, HbA1c 5.2–6.4%) [S]; 8–10 days [V]
- **Sensors:** Dexcom G6 (5-min) + Empatica E4 (BVP, EDA, skin temp, acceleration, HR, IBI) [V/S]; food logs; standardised breakfast every other day [V]
- **Licence:** Open Data Commons Attribution (ODC-By) [S]
- **Role:** Optional — tests whether wrist-worn HR/activity adds signal. **No T2D**, glucose rarely >180 → weak for the event target. Cut for the 6-day plan.

### 1.5 Glucose-ML collection — **tooling, not a dataset**
- **URL:** https://www.glucose-ml-project.com/ · https://github.com/Augmented-Health-Lab/Glucose-ML-Project [V]
- **Scale:** 44.9 M CGM samples, 20+ datasets, 4,300+ individuals, 6 countries; T1D 38%, T2D 25%, prediabetes 13%, no diabetes 24% [V]
- **Open vs controlled:** 14 open-access datasets with auto-download (incl. `cgmacros`, `shanghai`, `bigideas`); controlled ones (e.g. `ai-readi`, `ohiot1dm`, `t1dexi`, `diatrend`) need their own DUAs [V]
- **Code licence:** MIT [V]. Each dataset keeps its own licence.
- **Role:** Reuse its harmonisation scripts (`cgmacros_dexcom`, `shanghait2dm`) and participant-split utilities to save a day. 44.9 M samples ≠ 44.9 M independent examples — the unit of independence is the participant.

### 1.6 Indian sources

| Source | What it is | Participant-level data accessible? |
|---|---|---|
| **Ultrahuman M1 study** — Chaudhry M et al., *Sci Rep* 14:6490 (2024), doi:10.1038/s41598-024-56933-2 | 105 Indians (53 non-diabetic, 52 prediabetic), 9 clinics in 5 states, Libre-based UH-M1 + Fitbit Inspire 2, 14 days, OGTT, HbA1c, HOMA-IR, hs-CRP, cortisol; T2D **excluded**; CTRI/2022/08/044808; company-sponsored [V] | **No.** "Study outcomes data" available on reasonable request; platform code proprietary [V]. Not obtainable by 20 Oct. |
| **Pune-2021** — Majumdar S et al., *Front Endocrinol* 14 (2023), doi:10.3389/fendo.2023.1264072 | 50 CGM traces from SPPU Health Centre (Jul–Sep 2021) + Joshi Hospital set; 128 traces with HbA1c total, 112 usable; mixed diabetic / prediabetic / non-diabetic [V] | **No public repository located [U].** Data availability statement could not be retrieved (PMC blocked automated access). Treat as unavailable. |
| **Khan SA et al.** retrospective Indian CGM study | 499 adults (487 T2D, 12 T1D), Libre Pro or Gluconova, 5–14 days, age/sex/BMI/duration/HbA1c/PPBS [V — abstract] | **[U]** No repository identified. |
| **ICMR-INDIAB** (NHRDR) | Cross-sectional household survey, adults ≥20, 31 states/UTs, 2008–2020 [V] | **No.** NHRDR states the raw dataset will not be shared — only query outcomes [V]. No CGM. |

**Conclusion: no publicly accessible participant-level Indian CGM dataset was found.** Indian studies are used to establish local relevance (prevalence, Indian CGM reference values, HbA1c–CGM calibration differences), not to train or validate. The model must not be described as validated on Indian patients.

Not exhaustively searched in this pass (do if time permits after submission): CTRI registry trawl, India Data Portal, Zenodo keyword search for Indian CGM uploads.

---

## 2. Ranking — five strongest candidates

| Rank | Dataset | Why | Main weakness |
|---|---|---|---|
| 1 | **CGMacros** | Only open dataset with T2D + prediabetes, 5-min CGM, timed meal macros, wearable, *and* a lab panel for the same people | n = 45 (14 T2D); US population; NC-SA licence |
| 2 | **ShanghaiT2DM** | 100 T2D, open, clinical + meds | 15-min, no activity, no licence, Chinese diet logs |
| 3 | **AI-READI** | ~1,000 participants, T2D by treatment, wearables | Access approval + 2 TB; not feasible this week |
| 4 | **BIG IDEAs** | Clean Dexcom + research-grade wristband | No T2D; 16 people |
| 5 | **Ultrahuman M1 (India)** | Most relevant Indian cohort | Data not publicly accessible; no T2D |

---

## 3. Recommended data strategy

| Layer | Data | Purpose |
|---|---|---|
| Train + internal test | CGMacros, Dexcom 5-min stream + meals + Fitbit + bio.csv | Leave-participant-out CV (grouped K-fold, K = 5, stratified by diabetes status). Report every metric separately for T2D, prediabetes and healthy. |
| External test | ShanghaiT2DM, glucose-only model at 15-min resolution | Does the glucose-only model transfer to a different T2D population and device? |
| Demo stream (hosted app) | **Synthetic** virtual patients, clearly labelled, generated from the fitted mechanistic model with parameters sampled from the CGMacros posterior range | Satisfies "no private data on the public server" and any synthetic-only rule; real participants appear only in local runs if rules permit |
| Context | ICMR-INDIAB publications, Ultrahuman & Pune papers | Problem framing + Indian limitations section |

**Why this beats the alternatives:** it is the only combination that is (a) downloadable today without approval, (b) has paired static + dynamic data for the same person, (c) includes T2D, (d) supports an external-population check, and (e) fits the deadline. Choosing AI-READI would risk ending the week with no data; choosing synthetic-only (as many competing teams have) cannot answer the research question honestly — real-data evaluation is the differentiator.

**Combination rule:** CGMacros and Shanghai are **not** pooled for training (different devices, sampling, populations, meal encoding). Shanghai is a held-out external test only.

---

## 4. Variables that can actually be used (CGMacros)

| Group | Variables | Status |
|---|---|---|
| Target | Dexcom G6 glucose, 5-min | [V] |
| Alternate sensor | Libre Pro, 15-min (sensor-disagreement / noise ablation) | [V] |
| Meals | timestamp, kcal, carbs, protein, fat, fibre, meal type | [V] |
| Wearable | Fitbit fields | [U] — confirm from data dictionary |
| Static | age, gender, BMI, HbA1c, fasting glucose, fasting insulin, lipid panel | [V] |
| Derived static | HOMA-IR proxy from fasting glucose × insulin (proxy, single time point) | modelling assumption |
| Not used | microbiome, meal photos (out of scope for 6 days) | decision |

---

## 5. Licensing assessment for the public repo (not legal advice)

| Asset | Can it go on GitHub? | Notes |
|---|---|---|
| Our code | Yes | MIT or Apache-2.0 |
| CGMacros raw data | Technically allowed under NC-SA, but **don't** (627 MB, and the challenge may restrict real data) | Provide download instructions + citation instead |
| Small CGMacros-derived demo slice | Only if challenge rules allow real data; must be labelled **CC BY-NC-SA 4.0** with attribution | |
| Models trained on CGMacros | Treat conservatively as NC-SA derivative; label accordingly | |
| ShanghaiT2DM rows / derived rows | **No** | No licence attached |
| Synthetic demo stream | Yes | Our own licence; document the generator |

Project licence suggestion: **code under MIT; a `DATA_LICENSES.md` file listing each dataset's terms; model artefacts under CC BY-NC-SA 4.0.** Confirm the challenge has no conflicting IP clause.

---

## 6. Open items to verify (in order)

1. **Challenge data rules** on the Unstop page (decision gate — today).
2. Registration status before **18 Oct 19:00 IST**.
3. CGMacros data dictionary: Fitbit columns, missingness, which participants have Dexcom data.
4. Whether the CGMacros paper is now published (update citation).
5. ShanghaiT2DM file size and column structure.
