# Chapter 0 — PyTorch Installation and Environment



---

## 1. Overview

- How to install PyTorch on different operating systems: Linux, Windows, macOS.
- Difference between CPU-only and CUDA-enabled PyTorch builds.
- Using virtual environments (`venv`, `conda`) to isolate PyTorch projects.
- Installing compatible versions of `torch`, `torchvision`, and `torchaudio`.
- Verifying installation: checking version, CUDA availability, cuDNN, and device properties.
- Setting up a reproducible development environment for PyTorch projects.
- Common installation errors and how to resolve them.

---

## 2. Core Concepts

**Python Environment:** An isolated Python installation with its own packages. Prevents dependency conflicts between projects.

**pip:** Python's default package manager. Installs packages from PyPI or custom indexes.

**conda:** A cross-platform package and environment manager. Often used in scientific computing.

**Virtual Environment (`venv`):** Built-in Python module for creating lightweight isolated environments.

**Conda Environment:** A conda-managed isolated environment that can include non-Python dependencies.

**CUDA:** NVIDIA's parallel computing platform. Required for GPU acceleration with NVIDIA GPUs.

**cuDNN:** NVIDIA's deep neural network library. PyTorch uses it for optimized convolution and RNN operations.

**NVIDIA Driver:** System-level software that enables the OS to communicate with the GPU. Must be compatible with the CUDA version PyTorch expects.

**CUDA Toolkit:** Development tools and libraries for building CUDA applications. PyTorch wheels bundle the necessary runtime libraries, so a full local CUDA toolkit is often not required.

**ROCm:** AMD's open-source GPU computing platform. PyTorch supports ROCm on supported AMD GPUs.

**MPS (Metal Performance Shaders):** Apple's GPU acceleration backend for Apple Silicon Macs.

**PyTorch Wheel:** A precompiled binary package distributed via pip. Contains PyTorch C++/CUDA extensions.

**CPU-only Build:** PyTorch compiled without CUDA support. Runs only on CPU.

**CUDA Build:** PyTorch compiled with CUDA support. Can run on NVIDIA GPUs.

**Nightly Build:** Development version with latest features but less stability. Not recommended for production.

**Stable Build:** Officially released version recommended for most users.

---

## 3. Important PyTorch APIs

### `torch.__version__`

- **Name:** `torch.__version__`
- **Purpose:** Returns the installed PyTorch version string.
- **Syntax:** `torch.__version__`
- **Parameters:** None.
- **Return value:** `str` — e.g., `'2.4.0+cu121'`.
- **Important behavior:** The suffix indicates CUDA version if GPU-enabled.
- **Example:**

```python
import torch
print(torch.__version__)
# 2.4.0+cu121
```

### `torch.cuda.is_available()`

- **Name:** `torch.cuda.is_available`
- **Purpose:** Checks whether CUDA is available and usable.
- **Syntax:** `torch.cuda.is_available()`
- **Parameters:** None.
- **Return value:** `bool` — `True` if CUDA is available.
- **Important behavior:** Returns `False` for CPU-only builds or if no compatible GPU/driver is found.
- **Example:**

```python
print(torch.cuda.is_available())
# True
```

### `torch.cuda.device_count()`

- **Name:** `torch.cuda.device_count`
- **Purpose:** Returns the number of available CUDA devices.
- **Syntax:** `torch.cuda.device_count()`
- **Parameters:** None.
- **Return value:** `int`.
- **Example:**

```python
print(torch.cuda.device_count())
# 1
```

### `torch.cuda.get_device_name(index)`

- **Name:** `torch.cuda.get_device_name`
- **Purpose:** Returns the name of the specified CUDA device.
- **Syntax:** `torch.cuda.get_device_name(0)`
- **Parameters:**
  - `index` (int): Device index. Default `0`.
- **Return value:** `str`.
- **Example:**

```python
print(torch.cuda.get_device_name(0))
# NVIDIA GeForce RTX 4090
```

### `torch.version.cuda`

- **Name:** `torch.version.cuda`
- **Purpose:** Returns the CUDA version PyTorch was compiled with.
- **Syntax:** `torch.version.cuda`
- **Return value:** `str` or `None`.
- **Example:**

```python
print(torch.version.cuda)
# 12.1
```

### `torch.backends.cudnn.version()`

- **Name:** `torch.backends.cudnn.version`
- **Purpose:** Returns the cuDNN version used by PyTorch.
- **Syntax:** `torch.backends.cudnn.version()`
- **Return value:** `int` or `None`.
- **Example:**

```python
print(torch.backends.cudnn.version())
# 8907
```

### `torch.backends.mps.is_available()`

- **Name:** `torch.backends.mps.is_available`
- **Purpose:** Checks if Apple Metal Performance Shaders backend is available.
- **Syntax:** `torch.backends.mps.is_available()`
- **Return value:** `bool`.
- **Example:**

```python
print(torch.backends.mps.is_available())
# True on Apple Silicon with MPS support
```

### `torch.device`

- **Name:** `torch.device`
- **Purpose:** Represents the device on which a tensor or model is allocated.
- **Syntax:** `torch.device('cuda:0')`, `torch.device('cpu')`, `torch.device('mps')`
- **Parameters:**
  - `device` (str or int): Device string or ordinal.
- **Return value:** `torch.device` object.
- **Example:**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 4. Code Examples

### Example 1: Verifying PyTorch Installation

```python
import torch
import torchvision
import torchaudio

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Number of CUDA devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device capability:", torch.cuda.get_device_capability(0))
```

**What the code does:**
- Imports PyTorch and companion libraries.
- Prints version information.
- Checks CUDA availability and device details.

**Important lines:**
- `torch.__version__` shows the PyTorch version and CUDA suffix.
- `torch.cuda.is_available()` confirms GPU support.
- `torch.cuda.get_device_capability(0)` returns compute capability (e.g., `(8, 9)`).

**Expected output (GPU example):**
```
PyTorch version: 2.4.0+cu121
Torchvision version: 0.19.0+cu121
Torchaudio version: 2.4.0+cu121
CUDA available: True
CUDA version: 12.1
cuDNN version: 8907
Number of CUDA devices: 1
Device name: NVIDIA GeForce RTX 4090
Device capability: (8, 9)
```

### Example 2: Creating a Virtual Environment with `venv`

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

**What the code does:**
- Creates an isolated Python environment in `.venv`.
- Activates it so `pip` installs packages locally.

**Important lines:**
- `python -m venv .venv` creates the environment.
- `source .venv/bin/activate` activates on Linux/macOS.
- `.venv\Scripts\activate` activates on Windows.

### Example 3: Installing PyTorch with pip

```bash
# CPU-only (Linux/Windows)
pip install torch torchvision torchaudio

# CUDA 12.1 (Linux/Windows) — check pytorch.org for current command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# macOS (CPU or MPS)
pip install torch torchvision torchaudio
```

**What the code does:**
- Installs PyTorch and companion libraries.
- `--index-url` selects the CUDA-enabled wheel index.

**Important lines:**
- Always use the official command from `pytorch.org` for your CUDA version.
- The `cu121` suffix indicates CUDA 12.1.

### Example 4: Installing PyTorch with conda

```bash
conda create -n torch-env python=3.11
conda activate torch-env

# CPU-only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**What the code does:**
- Creates a conda environment named `torch-env` with Python 3.11.
- Installs PyTorch with or without CUDA support.

**Important lines:**
- `-c pytorch` and `-c nvidia` specify conda channels.
- `pytorch-cuda=12.1` selects the CUDA version.

### Example 5: Device Selection and Simple GPU Operation

```python
import torch

# Select best available device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("Using device:", device)

# Create tensors on device
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Matrix multiplication on device
z = x @ y
print("Result shape:", z.shape)
print("Result device:", z.device)
```

**What the code does:**
- Selects CUDA, MPS, or CPU automatically.
- Creates large tensors on the selected device.
- Performs matrix multiplication.

**Important lines:**
- `torch.device(...)` creates a device object.
- `device=device` places tensors directly on the device.

---

## 5. Important Parameters

### `pip install` Options

| Parameter | Description |
|-----------|-------------|
| `--index-url URL` | Base URL of the package index. Used for PyTorch CUDA wheels. |
| `--extra-index-url URL` | Additional index to search. |
| `--upgrade` / `-U` | Upgrade package to latest version. |
| `--no-cache-dir` | Disable pip cache. Useful for clean installs. |
| `--force-reinstall` | Reinstall package even if already installed. |
| `--pre` | Install pre-release/nightly versions. |

### `conda create` Options

| Parameter | Description |
|-----------|-------------|
| `-n NAME` | Name of the new environment. |
| `python=X.Y` | Python version. |
| `-c CHANNEL` | Conda channel to use. |
| `--yes` / `-y` | Answer yes to prompts. |

### `conda install` Options

| Parameter | Description |
|-----------|-------------|
| `pytorch` | PyTorch package. |
| `torchvision` | Computer vision library. |
| `torchaudio` | Audio library. |
| `pytorch-cuda=X.Y` | CUDA version for PyTorch. |
| `cpuonly` | CPU-only build. |
| `-c pytorch` | Official PyTorch channel. |
| `-c nvidia` | NVIDIA channel for CUDA packages. |

### `torch.cuda` Functions

| Function | Description |
|----------|-------------|
| `torch.cuda.is_available()` | Check CUDA availability. |
| `torch.cuda.device_count()` | Number of GPUs. |
| `torch.cuda.current_device()` | Current device index. |
| `torch.cuda.get_device_name(i)` | Name of GPU `i`. |
| `torch.cuda.get_device_capability(i)` | Compute capability tuple. |
| `torch.cuda.get_device_properties(i)` | Detailed properties. |
| `torch.cuda.memory_allocated()` | Current GPU memory usage. |
| `torch.cuda.memory_reserved()` | Cached GPU memory. |
| `torch.cuda.empty_cache()` | Release cached memory. |

---

## 6. Internal Working

### What Happens During PyTorch Installation

1. **Package Resolution:**
   - `pip` or `conda` resolves dependencies: Python version, NumPy, typing extensions, etc.
   - For CUDA builds, the wheel depends on specific CUDA runtime libraries.

2. **Wheel Download:**
   - PyTorch wheels are large (hundreds of MB to several GB).
   - CUDA wheels bundle CUDA runtime, cuDNN, and other libraries.

3. **Installation:**
   - Files are extracted to `site-packages/torch/`.
   - Native shared libraries (`.so`, `.dll`, `.dylib`) are placed alongside Python modules.

4. **Runtime Linking:**
   - When `import torch` runs, Python loads the native library.
   - PyTorch initializes CUDA context if GPU support is compiled in.
   - It checks the NVIDIA driver version and CUDA runtime compatibility.

5. **CUDA Context Creation:**
   - On first GPU operation, PyTorch creates a CUDA context on the current device.
   - Context initialization may take time and memory.

6. **cuDNN Initialization:**
   - cuDNN is loaded lazily when convolution or RNN operations are first used.

### Driver vs CUDA Runtime vs CUDA Toolkit

| Component | Role | Provided By |
|-----------|------|-------------|
| NVIDIA Driver | OS-level GPU communication | System installation |
| CUDA Runtime | High-level CUDA API | PyTorch wheel or system |
| CUDA Toolkit | Compiler, libraries, tools | Optional system installation |
| cuDNN | Deep learning primitives | PyTorch wheel or system |

**Key fact:** PyTorch wheels include the CUDA runtime and cuDNN. You only need a compatible NVIDIA driver on the system. You do **not** need to install the full CUDA Toolkit unless compiling custom CUDA extensions.

### Device Selection Flow

```
import torch
    ↓
Check torch.cuda.is_available()
    ↓
If True → CUDA device available
If False → Check torch.backends.mps.is_available()
    ↓
If True → MPS device available
If False → CPU device
```

---

## 7. Common Mistakes

**Mistake:** Installing PyTorch with `pip install torch` from default PyPI when GPU support is needed.

**Why it happens:** The default PyPI package may be CPU-only or link to a different CUDA version than the system driver supports.

**Correct approach:** Use the official command from `pytorch.org` with the correct `--index-url` for your CUDA version.

---

**Mistake:** Mismatched CUDA version between PyTorch and NVIDIA driver.

**Why it happens:** PyTorch wheel compiled for CUDA 12.1 requires a driver supporting CUDA 12.1 or newer.

**Correct approach:** Check driver version with `nvidia-smi`. Update driver if necessary. Install PyTorch wheel matching the driver's supported CUDA version.

---

**Mistake:** Mixing `conda` and `pip` installs in the same environment.

**Why it happens:** Conda and pip manage dependencies differently, leading to conflicts.

**Correct approach:** Prefer one package manager. If mixing, install conda packages first, then pip packages. Avoid installing PyTorch with both.

---

**Mistake:** Installing PyTorch in the system Python or base conda environment.

**Why it happens:** Convenience, but causes dependency conflicts across projects.

**Correct approach:** Always create and activate a dedicated virtual environment.

---

**Mistake:** Forgetting to activate the virtual environment before installing or running code.

**Why it happens:** The shell uses the system Python instead of the environment's Python.

**Correct approach:** Activate the environment and verify with `which python` (Linux/macOS) or `where python` (Windows).

---

**Mistake:** Assuming `torch.cuda.is_available()` returns `True` after installing CPU-only PyTorch.

**Why it happens:** CPU-only builds do not include CUDA support.

**Correct approach:** Install the CUDA-enabled wheel and verify with `torch.version.cuda`.

---

**Mistake:** Using incompatible versions of `torch`, `torchvision`, and `torchaudio`.

**Why it happens:** These libraries are version-locked. Mismatched versions cause import errors or runtime crashes.

**Correct approach:** Install them together from the same index. Check compatibility table on PyTorch website.

---

**Mistake:** Ignoring Python version compatibility.

**Why it happens:** PyTorch wheels are built for specific Python versions (e.g., 3.9–3.12).

**Correct approach:** Use a supported Python version. Check PyTorch release notes.

---

**Mistake:** Not setting `CUDA_VISIBLE_DEVICES` when multiple GPUs exist.

**Why it happens:** PyTorch uses GPU 0 by default, which may be busy or unsuitable.

**Correct approach:** Set `CUDA_VISIBLE_DEVICES=0` or use `torch.cuda.set_device(1)`.

---

**Mistake:** Running out of GPU memory due to cached memory not released.

**Why it happens:** PyTorch caches GPU memory for reuse. `nvidia-smi` may show high usage even after tensors are deleted.

**Correct approach:** Use `torch.cuda.empty_cache()` when needed, but avoid excessive calls as it slows performance.

---

## 8. Important Differences

| Concept | Difference |
|---------|-----------|
| `pip` vs `conda` | pip installs Python packages; conda manages environments and non-Python dependencies. |
| CPU-only vs CUDA build | CPU-only runs on CPU; CUDA build can use NVIDIA GPUs. |
| CUDA vs cuDNN | CUDA is the parallel computing platform; cuDNN is a library of optimized deep learning primitives. |
| NVIDIA Driver vs CUDA Toolkit | Driver enables GPU communication; Toolkit provides development tools. PyTorch wheels bundle runtime. |
| `venv` vs `conda` env | `venv` is Python-only; conda can manage Python and non-Python packages. |
| Stable vs Nightly | Stable is recommended; Nightly has latest features but may be unstable. |
| CUDA vs ROCm | CUDA is for NVIDIA GPUs; ROCm is for AMD GPUs. |
| CUDA vs MPS | CUDA is NVIDIA GPU backend; MPS is Apple Silicon GPU backend. |
| `torch.cuda.is_available()` vs `torch.backends.mps.is_available()` | Check for different GPU backends. |
| `torch.version.cuda` vs `nvidia-smi` CUDA version | `torch.version.cuda` is PyTorch's compiled CUDA version; `nvidia-smi` shows driver-supported CUDA version. |

---

## 9. Important Rules / Facts

- PyTorch wheels bundle CUDA runtime and cuDNN. Only a compatible NVIDIA driver is required.
- Always use a dedicated virtual environment per project.
- Use the official installation command from `pytorch.org` for your OS, package manager, and CUDA version.
- `torch.cuda.is_available()` must return `True` for GPU training.
- `torch.version.cuda` shows the CUDA version PyTorch was compiled with.
- `nvidia-smi` shows the driver version and the maximum CUDA version supported by the driver.
- PyTorch version must be compatible with `torchvision` and `torchaudio` versions.
- Python version must be supported by the PyTorch wheel.
- On Apple Silicon, use MPS backend for GPU acceleration.
- On AMD GPUs, use ROCm-enabled PyTorch builds.
- Nightly builds are for testing new features, not production.
- Conda and pip should not be mixed carelessly in the same environment.
- `CUDA_VISIBLE_DEVICES` controls which GPUs are visible to PyTorch.
- GPU memory is cached by PyTorch; `nvidia-smi` may show higher usage than actual tensor memory.
- Reproducibility requires pinning versions in `requirements.txt` or `environment.yml`.

---

## 10. Practical Example

### Complete Environment Setup and Verification

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.1 (check pytorch.org for current command)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Verify installation
python -c "
import torch
import torchvision
import torchaudio

print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('Torchaudio:', torchaudio.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('cuDNN version:', torch.backends.cudnn.version())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
"
```

**What the code does:**
- Creates an isolated Python environment.
- Installs PyTorch with CUDA support.
- Verifies all versions and GPU availability.

**Important lines:**
- `source .venv/bin/activate` activates the environment.
- `--index-url` selects the CUDA wheel index.
- The verification script prints all critical information.

**Expected output (GPU example):**
```
PyTorch: 2.4.0+cu121
Torchvision: 0.19.0+cu121
Torchaudio: 2.4.0+cu121
CUDA available: True
CUDA version: 12.1
cuDNN version: 8907
Device count: 1
Device name: NVIDIA GeForce RTX 4090
```

### Reproducible Environment File

```txt
# requirements.txt
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
numpy==1.26.4
```

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

**What the code does:**
- Pins exact versions for reproducibility.
- Installs all dependencies in one command.

---

## 11. Chapter Summary

- PyTorch can be installed via pip or conda.
- Use virtual environments to isolate projects.
- CPU-only builds are smaller; CUDA builds require compatible NVIDIA drivers.
- PyTorch wheels bundle CUDA runtime and cuDNN.
- Verify installation with `torch.__version__`, `torch.cuda.is_available()`, and related APIs.
- Device selection depends on available hardware: CUDA, MPS, or CPU.
- Mismatched versions of `torch`, `torchvision`, and `torchaudio` cause errors.
- `nvidia-smi` shows driver and GPU information.
- `torch.version.cuda` shows PyTorch's compiled CUDA version.
- `CUDA_VISIBLE_DEVICES` controls GPU visibility.
- Reproducibility requires pinned versions in environment files.
- Common mistakes include mixing package managers, using CPU-only builds for GPU work, and ignoring driver compatibility.

---

## 12. Important APIs to Remember

| API | Purpose |
|-----|---------|
| `torch.__version__` | PyTorch version string |
| `torch.cuda.is_available()` | Check CUDA availability |
| `torch.cuda.device_count()` | Number of GPUs |
| `torch.cuda.get_device_name(i)` | GPU name |
| `torch.cuda.get_device_capability(i)` | Compute capability |
| `torch.cuda.get_device_properties(i)` | Detailed GPU properties |
| `torch.version.cuda` | PyTorch compiled CUDA version |
| `torch.backends.cudnn.version()` | cuDNN version |
| `torch.backends.mps.is_available()` | Check Apple MPS availability |
| `torch.device(...)` | Device object |
| `tensor.to(device)` | Move tensor to device |
| `model.to(device)` | Move model to device |
| `torch.cuda.empty_cache()` | Release cached GPU memory |
| `torch.cuda.memory_allocated()` | Current GPU memory usage |
| `torch.cuda.memory_reserved()` | Cached GPU memory |
| `torch.cuda.set_device(i)` | Set current GPU |
| `torch.cuda.current_device()` | Get current GPU index |

---

## 13. Key Takeaways

1. Always use a virtual environment for PyTorch projects.
2. Use the official installation command from `pytorch.org` for your CUDA version.
3. PyTorch wheels bundle CUDA runtime and cuDNN; only a compatible NVIDIA driver is needed.
4. `torch.cuda.is_available()` must return `True` for GPU training.
5. `torch.version.cuda` and `nvidia-smi` report different CUDA versions — understand the difference.
6. `torch`, `torchvision`, and `torchaudio` versions must be compatible.
7. Use `torch.device` to manage CPU/GPU/MPS placement.
8. `CUDA_VISIBLE_DEVICES` controls which GPUs PyTorch can see.
9. `torch.cuda.empty_cache()` releases cached memory but should be used sparingly.
10. Reproducibility requires pinning exact versions in `requirements.txt` or `environment.yml`.
11. CPU-only builds are sufficient for small experiments but not for large-scale training.
12. Apple Silicon users should use the MPS backend for GPU acceleration.
13. AMD GPU users need ROCm-enabled PyTorch builds.
14. Verify installation before starting any project.
15. A correct environment setup prevents most runtime errors in later chapters.
