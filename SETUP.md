# Setup

## 1. Python environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
(`datasets` is pinned `<4.0` because APPS is a script-based dataset.)

## 2. Local models via Ollama (runs on the AMD 7900 XTX)

We run both models locally through [Ollama](https://ollama.com), which serves
them on `http://localhost:11434`. The 7900 XTX (gfx1100) is supported by
Ollama's ROCm backend, so no CUDA is involved.

1. **Install Ollama for Windows:** download from <https://ollama.com/download>
   and run the installer. It starts a background service automatically.
2. **Pull the two models** (this is a big one-time download: ~19 GB + ~4.7 GB):
   ```powershell
   ollama pull qwen2.5-coder:32b
   ollama pull qwen2.5-coder:7b
   ```
3. **Confirm they're there:**
   ```powershell
   ollama list
   ```
4. **Confirm the GPU is being used** (first token should be fast, and Task
   Manager's GPU memory should jump ~18 GB when the 32B model loads):
   ```powershell
   ollama run qwen2.5-coder:32b "print hello world in python"
   ```

### Model roles
- `qwen2.5-coder:32b` = **U** (untrusted, capable). ~18-20 GB in 4-bit, fits 24 GB VRAM.
- `qwen2.5-coder:7b`  = **T** (trusted, weaker). ~4.7 GB.

Ollama loads one model at a time and swaps U <-> T on demand (a few seconds);
with the call cache each unique call happens only once, so swapping is a
one-time cost.

## 3. Docker (sandbox for model-written code)

Any code produced by U (or T) is untrusted and must run isolated -- never on the
host. We use Docker for this. The solve-rate GATE and everything after it need it.

1. **Install Docker Desktop for Windows** from
   <https://www.docker.com/products/docker-desktop/> (uses the WSL2 backend;
   the installer sets this up). Start it once so the engine is running.
2. **Pre-pull the runner image** so the first sandboxed run isn't slow:
   ```powershell
   docker pull python:3.11-slim
   ```
3. **Confirm it works:**
   ```powershell
   docker run --rm --network none python:3.11-slim python -c "print('sandbox ok')"
   ```
   Should print `sandbox ok`. The `--network none` proves isolation from the network.

## 4. Smoke test
Once the models are pulled:
```powershell
.\.venv\Scripts\python.exe models\smoke_test.py
```
This checks Ollama is up, both models respond, and the cache works.

## 5. The Week-1 GATE
With Ollama + Docker both up:
```powershell
.\.venv\Scripts\python.exe analysis\measure_solve_rate.py --limit 30   # quick check
.\.venv\Scripts\python.exe analysis\measure_solve_rate.py              # full train split
```
This reports U's and T's solve rates and the capability gap between them.
