# Historical SimLLM source review

The historical implementation is identified and does not need to be rediscovered:

- Legacy Ascend PR #66: `f52ee0d301dc`, `eda5cbbcf64e`, `4db869c13ce1`
- Legacy Ascend PR #70: `c7e261573814`, `df6359d16276`, `dde9657ae3f0`,
  `9765e20897fc`, `d466c26dacce`, `c73365e4afde`, `0bcd745144a0`,
  `9b90d9553d9f`
- Legacy Ascend PR #80: `a66f1f15011b`, `0387940de05d`
- Legacy Ascend PR #157: `fa29895b0de3`, `7303b70da467`, `c35a1d554a84`,
  `e75a2b6301ef`, `0f4da0a33535`, `44c40343d8d1`, `aeed44dfea1d`,
  `bb7b901ec771`

This note records the source review state before the owner-directed migration.
The repository now contains the legacy Ascend implementation as an installable
plugin. The earlier hold in this note is superseded by that migration.
