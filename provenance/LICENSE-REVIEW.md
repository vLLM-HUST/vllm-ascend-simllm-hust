# SimLLM source migration hold

The historical implementation is identified and does not need to be rediscovered:

- Legacy Ascend PR #66: `f52ee0d301dc`, `eda5cbbcf64e`, `4db869c13ce1`
- Legacy Ascend PR #70: `c7e261573814`, `df6359d16276`, `dde9657ae3f0`,
  `9765e20897fc`, `d466c26dacce`, `c73365e4afde`, `0bcd745144a0`,
  `9b90d9553d9f`
- Legacy Ascend PR #80: `a66f1f15011b`, `0387940de05d`
- Legacy Ascend PR #157: `fa29895b0de3`, `7303b70da467`, `c35a1d554a84`,
  `e75a2b6301ef`, `0f4da0a33535`, `44c40343d8d1`, `aeed44dfea1d`,
  `bb7b901ec771`

The implementation appears related to the public `CGCL-codes/SimLLM` project,
which did not expose a reusable license when this migration was reviewed. Copying
implementation patches into this Apache-2.0 repository is therefore on hold.

Release requires either a license declaration from the copyright holder or a
clean-room implementation against a documented behavior and host-hook contract.
Metadata, manifest work, and interface design may continue meanwhile.
