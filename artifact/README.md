# Beam24 reproduction interface

The maintained entry point is `./reproduce.sh`; `./artifact/reproduce.sh`
remains compatible. Start with the [main README](../README.md) and follow the
[reproduction guide](../docs/REPRODUCIBILITY.md) for environment setup, data,
measurement contracts, and the complete target map.

```bash
./reproduce.sh smoke
./reproduce.sh evidence
./reproduce.sh external-hierarchy --dry-run
```

The primary external comparison is `external-hierarchy`. `system` and
`hierarchy` are internal controls and must not be labeled as external results.
`manifest.json` defines target ownership and scope;
`expected/reference_anchors.json` is the sole source of reference values.

Fresh outputs live in a new `artifact/runs/` directory. Failed, incomplete,
and low results remain available; default discovery does not scan historical
experiments. Data and upstream source caches live outside the tracked tree.
