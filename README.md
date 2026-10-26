# ChronoABOR

Analysis code and figure source data for the article

Temporal multiomics features outperform frontier large language models (LLMs) for Alzheimer's disease blood biomarker discovery.

ChronoABOR ranks candidate Alzheimer's disease (AD) blood biomarker genes from multi-omics evidence frozen at a historical cutoff year. The evidence comes from the Alzheimer's Disease Biomarker Online Resource (ABOR), freely available at [http://gepa.org.cn/ABOR/](http://gepa.org.cn/ABOR/).

## Repository structure

```
code/
├── analysis/       # analysis scripts for the results reported in the article
└── figures/        # scripts for Figures 2 and 3 and Supplementary Figures 1 and 2
source_data/
└── Supplementary_Data_1.xlsx   # source data for Figures 2 and 3 (identical to Supplementary Data 1 of the article)
pseudocode/         # key pseudocode of the ChronoABOR framework
```

`code/analysis/` contains:

- `rev_exp_common.py`: loading of the ABOR evidence tables, temporal feature construction and shared evaluation utilities
- `rev_exp_annotations.py`: static annotations (UniProt, KEGG, DisGeNET)
- `rev_exp_e*.py`: analyses of the revised manuscript
- `nm_exp*.py`, `experiment_01_baselines.py`: analyses of the initial submission, including the gene-blind LLM scoring; several revision analyses reuse their outputs
- `rev_exp_fetch_years*.py`, `pub_year_external.py`: resolution of publication years for ABOR source studies

Each script writes its output to `code/analysis/results/<script name>.json`.

## Main scripts and results

| Result in the article | Script(s) |
|---|---|
| External-corpus Folds C and D and the within-secreted evaluation under repeated nested cross-validation (Table 1, Table 2, Fig. 2a, Fig. 3b–d) | `code/analysis/rev_exp_e12_strict_windows.py` |
| Top-k precision with bootstrap confidence intervals (Fig. 2b) | `code/analysis/rev_exp_e12_strict_windows.py`, `code/figures/strict_fold_c_topk_bootstrap.py` |
| Retrospective recovery of established biomarker genes (Fig. 2c) | `code/analysis/rev_exp_e8_spec_model.py` |
| Feature-block ablation (Fig. 2d) | `code/analysis/rev_exp_e21_deployed_block_ablation.py` |
| SHAP feature contributions (Fig. 3a) | `code/analysis/rev_exp_e13_strict_spec_ablation.py` |
| Gene-blind LLM scores | `code/analysis/nm_exp5_llm_vs_ml_benchmark.py` (GPT models), `code/analysis/nm_exp_gemini_*.py` (Gemini-2.5 models), `code/analysis/rev_exp_e17_gemini31_strict.py` (Gemini-3.1-Pro) |
| Nested-CV hyperparameter selection | `code/analysis/rev_exp_e9_nested_tuning.py` |
| Permutation null on Fold C | `code/analysis/rev_exp_e14_strict_permutation.py` |
| Robustness and rolling-origin within-secreted evaluation (Supplementary Fig. 1) | `code/analysis/nm_exp_p9_rolling_origin.py`, `code/analysis/rev_exp_e8_spec_model.py` |
| Ranked and covariate-matched proteomic enrichment (Supplementary Fig. 2) | `code/analysis/nm_exp_p7_ranked_enrichment.py`, `code/analysis/nm_exp_p7b_matched_enrichment.py` |
| Second proteomic cohort and non-canonical enrichment | `code/analysis/nm_exp_p10_second_validation.py`, `nm_exp_p11_noncanonical_enrichment.py`, `nm_exp_p12_noncanonical_matched_null.py`, `nm_exp_p13_noncanon_thresholds.py` |
| Corpus-size subsampling, validation tiers, study list, analyte translation | `code/analysis/rev_exp_e2_corpus_saturation.py`, `rev_exp_e4_validation_strata.py`, `rev_exp_e5_study_list.py`, `rev_exp_e6_analyte_translation.py` |

## Requirements

Python 3.11. The package versions of the environment used for the reported analyses are listed in `requirements.txt`:

```
pip install -r requirements.txt
```

## Input data

The source data of the analyses are available from ABOR at [http://gepa.org.cn/ABOR/](http://gepa.org.cn/ABOR/).

- **ABOR evidence tables.** Download the seven layer tables (`degs.csv`, `snp.csv`, `cpg.csv`, `hubgene.csv`, `miRNA.csv`, `signature.csv`, `target.csv`) from ABOR and place them in `code/analysis/`. ABOR is updated continuously.
- **External evaluation corpus.** The gene list with provenance publications and fold assignments is given in Supplementary Table 3 of the article. The scripts read it as a LaTeX table from `code/analysis/paper/supplementary_table_s1.tex`.
- **Annotations and comparison datasets** in `code/analysis/external_data/`, obtained from the original sources under their own terms of use:
  - `annotations/`: reviewed human UniProt entries (release 2019-11), KEGG pathway hsa05010 gene links and DisGeNET v7.0 status (`uniprot_reviewed_human_raw.tsv`, `uniprot_secreted_human.tsv`, `uniprot_reviewed_human_manifest.json`, `kegg_hsa05010_link.tsv`, `kegg_hsa05010_genes.tsv`, `kegg_hsa_list.tsv`, `disgenet_status.json`)
  - `Frick2024_SupplementaryTables.xlsx`: supplementary tables of Frick *et al.*, *Nat. Aging* **4**, 1446–1464 (2024), including the SomaScan 7k analyte list
  - `olink_explore3072_assay_list.csv`: Olink Explore 3072 assay list
  - Indiana ADRC and Knight ADRC plasma proteomics hit lists: Huang *et al.*, *Alzheimers Dement.* **21**, e70579 (2025) and Heo *et al.*, *Nat. Aging* **5**, 1114–1131 (2025); the hit lists used are written in `nm_exp_p5_external_validation.py`
- **LLM access.** Scoring with LLMs requires API access. Set `OPENAI_API_KEY` (optionally `OPENAI_BASE_URL`) for the GPT models and `GEMINI_API_KEY` for the Gemini models.

## Running

Run the analysis scripts from `code/analysis/`, for example:

```
cd code/analysis
python rev_exp_e12_strict_windows.py
```

Then run the figure scripts from `code/figures/`; figures are written to `code/figures/output/`.
