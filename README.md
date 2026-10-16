# DSA4213 Assignment 1 - Language models, embeddings and representations

Corpus: nine 2025 corporate reports (eight sustainability / responsibility reports and one annual report) of Singapore-based
and regional companies. Parts I-II train language models on it; Part III uses the frozen models for an ESG
sentence-classification task.

## 1. Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # exact versions used
```
Developed on an Apple M1 MacBook Air with 8 GB RAM (PyTorch MPS backend); everything also runs on CPU or CUDA.
**Run all commands from this folder (`Assignment 1/`).**

## 2. Data

### 2.1 Reports (Parts I-II)
| id | company | document | source URL |
|---|---|---|---|
| 01 | Sembcorp | Sustainability Report 2025 | https://www.sembcorp.com/media/z4ohu5lz/sembcorp-ar25_sustainability-report.pdf |
| 02 | Keppel | Sustainability Report 2025 | https://www.keppel.com/file/sustainability/sustainability-reports/keppel-sustainability-report-2025.pdf |
| 03 | SP Group | Sustainability Report FY2025/26 | https://www.spgroup.com.sg/dam/spgroup/pdf/about-us/our-sustainability-commitment/sp-group-sustainability-report-fy2025-26 |
| 04 | ST Engineering | Sustainability Report 2025 | https://www.stengg.com/Getcontentasset/Ad7f7fb3-5623-4Eb1-829B-44340Bfd8c28/Dfc3d011-8F63-43F6-9Ed8-4B444333a1d0/St-Engineering-Sustainability-Report-2025.Pdf?Language=En |
| 05 | CBRE | Corporate Responsibility Report 2025 | https://mediaassets.cbre.com/-/media/project/cbre/dotcom/global/about/corporate-responsibility/corporate-responsibility-report-2025/cbre-2025-corporate-responsibility-report.pdf |
| 06 | Singtel | Sustainability Report 2025 | https://cdn1.singteldigital.com/content/dam/singtel/investorRelations/annualReports/2025/SR2025.pdf |
| 07 | DBS | Sustainability Report 2025 | https://www.dbs.com/iwov-resources/images/sustainability/reporting/pdf/web/DBS_SR2025.pdf |
| 08 | ComfortDelGro | Sustainability Report 2025 | https://www.comfortdelgro.com/wp-content/uploads/2026/03/ComfortDelGro-Sustainability-Report-2025-2.pdf |
| 09 | Singapore Airlines | Annual Report FY2025/26 | https://www.singaporeair.com/content/dam/sia/web-assets/pdfs/about-us/information-for-investors/annual-report/annualreport2526.pdf |

The PDFs are **not** included (size and redistribution). Download them from the URLs above.

### 2.2 How `data/sustainability_reports_corpus.txt` was built
1. Text was extracted from each PDF: **TODO - state the tool/command used** -> `data/<company>_..._cleaned.txt`.
2. The nine cleaned files were **merged into one file with the help of ChatGPT** (an LLM), using one
   `<<<REPORT_START id=.. company=".." year="..">>>` marker per report. The merge was not purely a
   concatenation. Comparing the final file with the cleaned inputs paragraph by paragraph shows that it fixed ligatures
   (e.g. "ﬁned" -> "fined"), split run-together words ("Asiacentric" -> "Asia centric"), removed index / infographic
   fragments in the Keppel and SP Group reports, and re-flowed the Sembcorp report into short paragraphs. This step is
   not scripted and cannot be reproduced exactly, so the final corpus file is the reference input for all experiments.
   SHA-256 of the file used: `bde5dce576a0b38802cc36ce56a2f925abc86b1816c505f0139d7d1e3ab18b80`.
3. The final corpus file is included in this repository (`data/sustainability_reports_corpus.txt`). The reports are public
   documents; the per-report cleaned files are available on request.

### 2.3 Preprocessing (`src/data.py`)
Lower-casing; word-level tokens + punctuation; paragraph end -> `<eos>`; paragraphs under 3 tokens dropped;
`scope 1/2/3` -> `scope_1/2/3`; years 2010-2069 -> `year_YYYY`; every other standalone number -> `<num>`;
vocabulary = words with >=3 occurrences in the **training** split (cap 10,000), the rest -> `<unk>`.

**Split:** each report is cut into 1,024-token blocks; per report, 10% of blocks go to validation and 10% to test
(rotating phase so every section type can appear in every split), the rest to training; windows never cross a block
boundary; the trailing <1,024 tokens of each report are discarded (3,946 tokens, ~1%). Deterministic, no randomness.
Evaluation therefore measures generalisation to held-out sections of familiar reports, not to unseen companies.
(An earlier version took the last 10% of each report as test; that put appendices / assurance statements in val/test and
inflated perplexity, so it was replaced.)

### 2.4 Part III data
`ESGBERT/environmental_2k` (Hugging Face, Apache-2.0; label `env`): downloaded automatically by `src/classify.py`.
I could not find documentation of how the labels were made. I drop exact duplicate sentences and build my own
stratified 60/20/20 split (3 split seeds).

## 3. Recreating the findings, step by step

Everything in the report comes from the commands below. Training is resumable: a run whose `results/<tag>_result.json`
exists is skipped, so an interrupted run can simply be started again.

**Step 1 - environment.** Create the environment and install the packages (section 1). Run all commands from `Assignment 1/`.

**Step 2 - data.** `data/sustainability_reports_corpus.txt` is included; check it with
`shasum -a 256 data/sustainability_reports_corpus.txt` (the hash is in section 2.2). Download GloVe for Part II:
```
mkdir -p data/glove
curl -L -o data/glove/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip -o data/glove/glove.6B.zip glove.6B.100d.txt -d data/glove
```
The ESGBERT dataset for Part III downloads automatically from Hugging Face (internet needed).

**Step 3 - quick test (a few minutes).** `SMOKE=1 bash src/run_all.sh` runs the whole pipeline with 1 epoch per run, so
numbers will differ; it only confirms that everything runs. Archive its outputs before the real run:
`mv results results_smoke && mv data/processed data/processed_smoke`.

**Step 4 - full run (about 4-6 hours on an M-series laptop).** The repository already contains the published `results/`,
which `run_all.sh` would skip as finished. To retrain from scratch, move them aside first:
```
mv results results_published && mv data/processed data/processed_published 2>/dev/null
bash src/run_all.sh
```
`run_all.sh` can also run one stage at a time, in this order:

| stage | what it does | main outputs |
|---|---|---|
| `prep` | tokenise, split, build vocabulary; trigram | `results/logs/data_stats.txt`, `results/ngram_result.json` |
| `grid` | RNN / LSTM / Transformer at lr 1e-3 and 3e-3 (seed 42); picks the best lr on validation perplexity | `results/*_result.json`, `*_log.csv`, `chosen.json` |
| `seeds` | seeds 43 and 44 for the chosen configurations | `results/*_s43_*`, `*_s44_*` |
| `part2` | own Skip-gram, GloVe coverage, and the five embedding settings x 3 seeds | `results/sgns_cost.json`, `glove_coverage.json`, `lstm_{sgns,glove}_{frozen,init}*` |
| `eval` | correctness tests, parameter counts, loss by position, Part III classification, generation, throughput | `results/logs/tests.txt`, `param_breakdown.csv`, `classify_*.csv`, `errors_*.md`, `generations_*.md`, `benchmark_*.json` |

**Step 5 - extra checks and report files.**
```
python src/leakage_check.py         # Part III sentences vs. training text -> results/leakage_check.json
python src/make_report_assets.py    # tables and figures used in the report
```
`make_report_assets.py` only reads the files in `results/`, so it can also be run on its own (without retraining) to
regenerate `table_part1/2/3` and the `fig_*.png` files from the stored results. It needs `matplotlib` and `tabulate`.

**Step 6 - compare with the report.** Training on MPS is not bit-wise deterministic, so expect small differences
(about 1 perplexity point or 0.01 macro-F1); the ordering of the models should be the same.

| report item | produced by | file |
|---|---|---|
| Table 1 (Part I results) | `train.py`, `benchmark.py`, `make_report_assets.py` | `results/table_part1.md` |
| Figure 1 (training curves) | `make_report_assets.py` | `results/fig_part1_curves.png` |
| Figure A2 (loss by position) | `extra_checks.py position` | `results/fig_loss_by_position.png`, `loss_by_position.csv` |
| Figure A1 (correctness tests) | `extra_checks.py tests` | `results/logs/tests.txt` |
| Table A2 and repetition rates (Part I generation) | `generate.py --set part1` | `results/generations_part1.md`, `generation_stats_part1.json` |
| Table 2 (Part II) | `train.py`, `embeddings.py`, `make_report_assets.py` | `results/table_part2.md`, `sgns_cost.json`, `glove_coverage.json` |
| Figure B1 and Table B1 | `make_report_assets.py`, `generate.py --set part2` | `results/fig_part2_curves.png`, `generations_part2.md` |
| Table 3 and Figure C1 (Part III) | `classify.py`, `make_report_assets.py` | `results/table_part3.md`, `fig_part3_f1.png`, `classify_summary.csv`, `classify_all_runs.csv` |
| Table C1 (errors) | `classify.py` | `results/errors_LSTM_mean.md`, `errors_Transformer_mean.md` |
| Leakage check (Section 5.1) | `leakage_check.py` | `results/leakage_check.json` |

Headline numbers to check (mean over 3 seeds): test perplexity trigram 127.4, RNN 108.0, LSTM 97.8, Transformer 94.0;
Part II LSTM with frozen Skip-gram 100.2, frozen GloVe 102.5, initialised Skip-gram 84.0, initialised GloVe 85.1;
Part III mean-pooled macro-F1 of the GloVe-frozen LSTM 0.842, TF-IDF baseline 0.877.

Model checkpoints (`*.pt`) are not included; `run_all.sh` regenerates them.

| script | purpose |
|---|---|
| `src/data.py` | cleaning, tokenisation, split, vocabulary, windows |
| `src/ngram.py` | interpolated trigram (unigram add-one + bigram + trigram; weights tuned on val) |
| `src/models.py` | RNN / LSTM / causal-Transformer LMs (untied input/output embeddings) |
| `src/train.py` | training (AdamW, grad-clip 1.0, early stopping on val loss, padding excluded from loss) |
| `src/select_best.py` | picks the learning rate per architecture on **validation** perplexity |
| `src/embeddings.py` | Skip-gram with negative sampling (own embeddings); GloVe 6B-100d loading + coverage |
| `src/classify.py` | Part III: frozen-LM features -> logistic regression |
| `src/generate.py`, `src/benchmark.py` | matched-decoding generation; inference throughput |
| `src/extra_checks.py` | causality / padding / loss-masking tests; parameter breakdown; loss-by-position |
| `src/leakage_check.py` | overlap between the Part III sentences and the reports |
| `src/make_report_assets.py` | tables and figures for the report |

## 4. Configuration and seeds
Embedding dim 100 (matches GloVe-100d); RNN/LSTM: 1 layer, hidden 256; Transformer: 2 layers, 4 heads, d_model 100, FFN 256,
learned positions, pre-norm, causal mask; dropout 0.3; AdamW (weight decay 0.01); batch 64; context 64 tokens;
epoch cap 100 (250 for the Transformer) with patience 8. Learning rate in {1e-3, 3e-3} chosen per architecture on validation
perplexity (seed 42), then seeds 43 and 44 are run for the chosen setting. Seeds control initialisation, dropout **and**
training-batch order. Skip-gram: window 5, 5 negatives, 10 epochs, sub-sampling 1e-3, seed 42 (single table). Both embedding
tables are rescaled by one scalar to unit std; missing GloVe words ~ N(mean, std) of found vectors; `<pad>` = 0.
Part III: logistic regression, `class_weight=balanced`, C in {0.001, 0.01, 0.1, 1} chosen on validation macro-F1;
LM seeds 42-44 x split seeds 42-44. MPS kernels are not bit-wise deterministic, so reruns can differ slightly.

## 5. Outputs (`results/`)
`table_part1/2/3.md|csv` (report tables), `fig_*.png` (report figures), `*_result.json` and `*_log.csv` (per-run results and
training curves), `chosen.json` (selected learning rates), `benchmark_*.json` (throughput), `generations_part*.md`
(samples), `errors_*.md` (Part III errors), `leakage_check.json`, `logs/` (run logs and correctness tests).

## 6. Tools, sources and citations
- Code was developed with substantial assistance from **Claude (Anthropic)**, which wrote the first versions of the scripts,
  helped debug them and helped draft parts of the report text; I ran, checked and modified the code, revised the text and wrote
  the analysis myself. **ChatGPT (OpenAI)** helped merge and clean the corpus text (section 2.2).
- GloVe: Pennington, Socher, Manning (2014), *GloVe: Global Vectors for Word Representation*, EMNLP. Vectors: glove.6B (Wikipedia 2014 + Gigaword 5).
- Dataset: `ESGBERT/environmental_2k` (Hugging Face).
- Libraries: PyTorch, NumPy, pandas, scikit-learn, Hugging Face `datasets`, matplotlib.
