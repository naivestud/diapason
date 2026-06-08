# Manifest format

A manifest is a JSONL file — one JSON object per line, one evaluation example
per object. Manifests are the only data files diapason reads; everything else
is configuration. A manifest may be gzip-compressed: name it `*.jsonl.gz` and
it is decompressed transparently.

## Common fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | string | yes | Unique identifier for the example. Duplicates are rejected. |
| `audio` | string | no | Path to the audio file. Passed through to adapters; diapason itself never reads the audio. |
| `reference` | string or list | task dependent | The ground-truth annotation the metrics score against. |
| `prediction` | string | no | A stored prediction. When absent, the task's adapter generates one at run time. |
| `prompt` / `question` | string | AQA only | The question or instruction given to the model. `question` is an alias for `prompt`. |

Any field not listed above is collected into the example's `meta` mapping, so
you can carry speaker ids, durations, or splits through the harness.

## ASR manifests

`reference` is the transcription; `prediction`, when present, is the model
transcription. The reference may also be a list of acceptable transcriptions —
the first entry is the primary reference and the rest are alternatives; WER
and CER report the best match across them.

```jsonl
{"id": "asr-001", "audio": "audio/asr-001.wav", "reference": "the quick brown fox"}
{"id": "asr-002", "audio": "audio/asr-002.wav", "reference": ["the seashore", "the sea shore"]}
```

## SED manifests

`reference` and `prediction` are lists of events. Each event carries `onset`
and `offset` in seconds plus a string `label`; predictions may add a
confidence `score` in `[0, 1]` used by threshold-sweeping metrics such as
`psds_lite`. SED tasks never call an adapter — both sides come from the file.

```jsonl
{"id": "sed-001", "reference": [{"onset": 0.5, "offset": 2.1, "label": "dog"}], "prediction": [{"onset": 0.6, "offset": 2.0, "label": "dog", "score": 0.9}]}
```

## AQA manifests

`reference` is the answer string, and every example needs a `question` (or
`prompt`). Predictions stored in the file are scored directly.

```jsonl
{"id": "aqa-001", "question": "What animal can be heard?", "reference": "a dog", "prediction": "a dog"}
```

## Validation rules

Loading a manifest fails fast with the file path and line number when:

- a line is not valid JSON or not a JSON object;
- an example is missing `id`;
- two examples share an `id`;
- a reference required by the task type is missing (`reference` for ASR and
  AQA, reference events for SED);
- an AQA example has no question;
- the file contains no examples at all.
