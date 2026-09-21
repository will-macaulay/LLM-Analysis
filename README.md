# Control Protocol Frontier

Measuring the **safety–usefulness frontier** of AI control protocols: given an
untrusted model (U) that may sabotage and a weaker trusted model (T), how much
useful work can we extract while keeping successful sabotage rare?

Novel piece: an attack policy derived as an **optimal stopping problem** (a
time-varying threshold rule), compared against a heuristic attacker.

See [`CLAUDE.md`](CLAUDE.md) for the full project brief and [`SETUP.md`](SETUP.md)
to get running.

## Repo map
```
config.py     all knobs in one place (models, seed, filters, audit budget)
data/         APPS loading + reliability filtering + train/held-out split
sandbox/      isolated execution of model code (Sandbox interface; Docker + local backends)
models/       inference wrappers (Ollama) + content-addressed call cache
protocols/    blue-team implementations        (upcoming)
attacks/      red-team policies                 (upcoming)
analysis/     metrics, bootstrap CIs, plotting  (upcoming)
results/      raw JSON experiment outputs
writeup/
```

## Key design decisions
- **Local inference on an AMD 7900 XTX via Ollama.** 24 GB VRAM fits a 4-bit 32B
  model; we avoid the CUDA-only HF stack. `models.generate()` hides where
  inference runs behind one function.
- **Every model call is cached** by (model, prompt, params, seed) so each unique
  call runs once and results reproduce exactly.
- **Untrusted code only ever runs in Docker.** A `Sandbox` interface separates
  *how we isolate* from *how we grade*; a hard guard refuses untrusted code on
  the non-isolated local backend.
- **Reliable test cases are verified, not assumed:** a problem is kept only if a
  ground-truth APPS solution actually passes all its tests in the sandbox.

## Week-1 pipeline
```powershell
python data\load_apps.py           # download APPS, structural filter -> candidates
python data\validate_and_split.py  # verify reliable tests, sample, 50/50 split
python models\smoke_test.py        # (after Ollama setup) check U/T + cache
```

## Status
Week 1 in progress: data pipeline + sandbox + inference cache built. Next:
measure U vs T solve rates (capability-gap GATE), then check U can produce
verified backdoors.
