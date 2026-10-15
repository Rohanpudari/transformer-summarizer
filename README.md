# Transformer Summarization Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohanpudari/transformer-summarizer/blob/main/notebooks/transformer_summarizer.ipynb)

Compares three pre-trained Hugging Face summarizers on CNN/DailyMail, tests whether
map-reduce chunking helps on long documents (BillSum), and audits the outputs for
unsupported content. Everything runs in one Colab notebook on a free T4 GPU.
This is a notebook-based toolkit; it has no web UI.

## What is in the notebook
- Preprocessing and postprocessing, token-aware chunking, and map-reduce summarization
  for documents beyond the 1,024-token input limit
- A controlled benchmark of DistilBART, BART-large and PEGASUS (same 100 articles, same decoding settings)
- ROUGE-1/2/L, BERTScore, compression, latency, a Lead-3 baseline, and paired-bootstrap confidence intervals
- A long-document experiment (50 BillSum bills): truncation vs chunking
- A faithfulness audit (automatic checks plus a small manual review)

## Results: CNN/DailyMail (100 random test articles, seed 42)

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Avg words | Latency/article |
|---|---|---|---|---|---|---|
| Lead-3 baseline | 39.6 | 16.7 | 24.9 | 0.248 | - | - |
| DistilBART-cnn-12-6 | 44.4 | 20.9 | 29.7 | 0.307 | 67.1 | 1.04 s |
| BART-large-cnn | 44.0 | 21.3 | 30.6 | 0.302 | 63.0 | 1.46 s |
| PEGASUS-cnn_dailymail | 44.3 | 20.2 | 30.6 | 0.314 | 50.8 | 1.71 s |

Human reference summaries average 53.2 words. Latency is from a single T4 run.

1. **No significant quality difference between the three models.** All 12 paired
   bootstrap comparisons (10,000 resamples, 95% CI) include 0. With N=100 the
   intervals are about +/-2.3 ROUGE-1 points, so smaller gaps cannot be detected.
2. **All three beat Lead-3.** ROUGE-1 gains are +4.3 to +4.7 and BERTScore gains
   +0.054 to +0.066; all 12 intervals exclude 0.
3. **Speed differs.** DistilBART was about 1.4x faster than BART-large and 1.6x
   faster than PEGASUS in this run.
4. **The models mostly copy.** Novel bigrams are 10.6-21.5% for the models vs 52.7%
   for the human references.

## Results: long documents (50 BillSum bills of 1,500-6,000 tokens, BART-large-cnn)

| Metric | Truncated (first 1024 tokens) | Map-reduce chunked | Difference (95% CI) |
|---|---|---|---|
| ROUGE-1 | 34.8 | 34.1 | -0.6 (-2.9, +1.5) |
| ROUGE-2 | 16.1 | 13.8 | -2.3 (-4.7, -0.1) |
| ROUGE-L | 23.8 | 22.5 | -1.4 (-3.7, +0.8) |
| BERTScore F1 | 0.095 | 0.080 | -0.016 (-0.045, +0.012) |

Chunking did not improve quality and was about 2.7x slower (5.2 s vs 2.0 s per bill).
Only the ROUGE-2 drop is significant, and only marginally; four metrics were tested
without correction.

## Error analysis
- In a small manual review (8 random BART-large summaries plus 10 flagged cases),
  every claim I checked in the random sample was supported by the source. The
  weaknesses were heavy copying, dropped key facts and sentence fragments.
- In 2 of 100 articles (apparent opinion pieces), 5 of 6 model outputs credited the
  text to an author who does not appear in the source.
- Entity flags (summaries with a name absent from the source): 5% DistilBART,
  5% BART-large, 8% PEGASUS. Only PEGASUS had unsupported numbers (2%).
- On one BillSum bill, the truncated output invented a legislative history with
  names absent from the source.
- These checks are crude proxies; they find candidates to read, not hallucination rates.

## Limitations
- Small samples (100 and 50 documents), one seed, one GPU run, one reviewer.
- ROUGE-L here is the whole-text variant, about 10 points below the ROUGE-Lsum
  figures in model papers.
- BERTScore is baseline-rescaled; values are not comparable to unscaled ones.
- BillSum references (161.5 words on average) are about twice as long as the outputs
  (about 75 words), which lowers recall for both strategies. The models are
  news-trained and were not fine-tuned on legal text.
- No dedicated factuality metric.

## Reproducibility note
The saved notebook outputs come from a single Colab session. The notebook was
developed incrementally and was not re-run top to bottom after the final edits, so
a fresh run may need small fixes. Library versions are not pinned (Google Colab
default runtime, October 2026).

## Repo layout
`notebooks/` the notebook with saved outputs, `results/` CSVs (article text removed),
`figures/` charts, `requirements.txt`.

## Data and models
CNN/DailyMail (abisee/cnn_dailymail), BillSum (FiscalNote/billsum), and the
checkpoints sshleifer/distilbart-cnn-12-6, facebook/bart-large-cnn and
google/pegasus-cnn_dailymail. Check each dataset and model card for its license.