# Migration Record

## Upstream

```text
path: C:/Users/Administrator/Documents/阅读代码/language_decoding_text_decoding_5396c4c
commit: 5396c4ca8b9620cfdfba6a680e084fe003c58832
```

## Migrated unchanged

- `data/evaluation/littleprince_hf_v1/littleprince_hf_lexicon_v1.csv`
- `data/evaluation/littleprince_hf_v1/littleprince_sentence_keyword_labels_v1.csv`
- `data/evaluation/littleprince_hf_v1/README.md`
- `data/manifests/littleprince_pl_all_clean_manifest.csv`
- `chineseeeg2_littleprince/io/brainvision.py`

The source README is stored as `SOURCE_README.md`. Exact source hashes are recorded in `PROVENANCE.json`.

## Deliberately not migrated

- embedding-SHA canonical `target_id`;
- legacy sentence retrieval metrics;
- legacy `train.py`;
- incomplete MEG/speech sequence code;
- checkpoints, outputs, temporary screenshots and model caches.

New code uses `text_embedding_idx` as the sentence occurrence join key and will use a versioned normalized-text group for cross-validation.

