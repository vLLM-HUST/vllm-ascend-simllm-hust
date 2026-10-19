# Issue #2 runtime guard verification, 2026-10-19

These are engineering checks, with no model-inference or performance claim.
See [the acceptance audit](../../../docs/issue2-acceptance.md) for scope and remaining work.

- `unit-tests.log`: 227 passed, 2 skipped, 4 model tests deselected. The manifest
  test was excluded because the remote environment lacks vllm_hust_ext.
- `clean-wheel.log`: isolated Python 3.13 install, inert entry point, rejection
  of unqualified enablement, restart disable and uninstall.
- `host-lifecycle-initial-failure.log`: retained initial probe failure from
  importing the host runner before the Ascend operators were initialized.
- `host-lifecycle.log`: corrected operator import order; real runner states
  disabled/enabled/disabled and overlay uninstall all passed on Python 3.11.16.
  The installed test wheel SHA-256 is
  `12dc24c9992b12e19afd4a4bd1265d86c7990e262f641599489992d6053ab608`.
  It predates the final missing-runner rejection, which the final unit suite
  covers. No installed inference regression was run for this branch.

Remote commands (after copying this tree to the host):

```bash
source /data/jxd/Ascend-9.1.0/cann-9.1.0/set_env.sh
source /data/jxd/Ascend-9.1.0/nnal/atb/set_env.sh
PYTHONPATH=src /data/jxd/envs/vllm-hust-v1-cann91/bin/python -m pytest \
  -q -m 'not model' --ignore=tests/test_manifest.py
/data/jxd/envs/vllm-hust-v1-cann91/bin/python benchmarks/verify_host_lifecycle.py
```

The host probe inherits existing runtime dependencies in a disposable virtual
environment. It does not uninstall or replace the original serving package.
Logs are checksummed in `SHA256SUMS`.
