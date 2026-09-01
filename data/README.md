# Data

This directory contains the two public input files used by the original OPBench
workflow:

- `locomo10.json`: the ten-conversation LoCoMo subset used to build the benchmark
  and to initialize the LDAgent/SimpleRAGAgent memories.
- `locomo10_overpersonalized.json`: the generated OPBench task file containing
  profiles, observations, and the four task groups.

The repository does not include generated answers, judge outputs, vector stores,
or model caches. Those files belong under `results/` or a local cache directory
and are ignored by Git.

