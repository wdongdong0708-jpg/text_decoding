# Stage 3 Variable Dual-Sequence Data Contract

## 1. Scientific access boundary

`ContextEEGDataset` accepts exactly one frozen outer fold and one role.

| role | EEG | keyword truth | contextual word targets |
|---|---:|---:|---:|
| train | yes | yes | optional, normally yes |
| validation | yes | yes | forbidden |
| test | yes | yes | forbidden |

For `validation` or `test`, either requesting
`include_context_targets=True` or wiring a `context_store_root` raises
`ContextTargetAccessError`. Their batches contain `context_words=None`; the
pipeline never substitutes zero vectors for unavailable text targets.

Word IDs, surfaces and keyword indices in evaluation samples are label
metadata for metrics and offline error analysis. They are not model inputs.

## 2. Sample unit

One `ContextEEGSample` represents one subject view of one sentence
occurrence. It contains:

- EEG identity: `eeg_view_id`, subject/session/task/run and local/global row;
- split identity: `text_embedding_idx`, `sentence_group_id`, `outer_fold`,
  `role`;
- window metadata: sampling rate, start/stop samples and EEG length;
- raw variable EEG tensor `[C,T] float32`;
- frozen word occurrence IDs, positions, surfaces and character spans;
- stable per-word context-token-group and lexical-surface-type indices;
- stable integer `sentence_group_index`;
- stable per-word Master indices (`-1` for a non-Master word);
- unique present-keyword truth in the same Master index space;
- optional train-only contextual word tensor;
- backend, cache metadata SHA256 and context-vector SHA256.

The Dataset does not pool subjects, construct prototypes, project features,
run Sinkhorn, normalize EEG, crop, resample or augment.

## 3. Batch contract

`ContextEEGCollator` pads only to maxima inside the current batch:

```text
eeg                       [B,C,T_max] float32
eeg_mask                  [B,T_max] bool
eeg_lengths               [B] int64
context_words             [B,N_max,...] float32 or None
word_mask                 [B,N_max] bool
word_lengths              [B] int64
context_token_group_indices [B,N_max] int64, padding=-1
surface_type_indices      [B,N_max] int64, padding=-1
word_keyword_indices      [B,N_max] int64, padding=-1
word_positions            [B,N_max] int64, padding=-1
word_char_spans           [B,N_max,2] int64, padding=-1
subject_indices           [B] int64
text_embedding_indices    [B] int64
sentence_group_indices    [B] int64
```

MacBERT retains feature tail `[4,768]`; BGE-M3 retains `[1024]`. Collate does
not hard-code either shape and rejects mixed tail shapes or backends.
`eeg_mask.sum(1) == eeg_lengths` and
`word_mask.sum(1) == word_lengths` are enforced.

Variable present-keyword truth and string metadata remain per-sample tuples.
`ContextEEGBatch.to()` moves tensors only and preserves strings.
`ContextEEGBatch.pin_memory()` is available for explicit pinned-memory use.

## 4. Stable indices

- Subjects use ascending Unicode code-point order over the complete manifest,
  producing `sub-01`–`sub-08 -> 0`–`7`.
- Master keywords use ascending frozen lexicon rank.
- Core, Main, Extended and Master share the same 247-entry index space.
- Indices never change by fold or fold-specific eligibility.
- Surface types use ascending Unicode order over all 14,034 frozen word
  occurrences; non-Master words retain distinct lexical identities.
- Context-token groups use ascending
  `(sentence_group_id, word_position, surface_form)` tuples, so normalized
  duplicate sentences share targets while different word positions do not.
- The lexical mapping records both source SHA256 values and its own canonical
  mapping SHA256.

## 5. Worker resource lifecycle

Dataset initialization validates frozen folds, word truth and train context
cache hashes. EEG recordings are not loaded at initialization.

Each process lazily caches one `BrainVisionReader` per recording and one
read-only `ContextWordStore` per Dataset. Dataset pickling removes memmaps and
reader objects, so Windows spawn workers reopen their own read-only resources
instead of serializing open handles or large arrays.
