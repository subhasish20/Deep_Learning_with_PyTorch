# Introduction to PyTorch



## 1. What is PyTorch?

### Simple Definition
PyTorch is an **open-source Python library** for building and training machine learning and deep learning models. It provides tools to work with data, build neural networks, and compute gradients automatically.

### What Problem Does PyTorch Solve?
Without PyTorch (or similar libraries), you would have to:
- Manually write complex math for neural networks
- Manually calculate gradients (which is extremely error-prone)
- Write your own GPU-accelerated code

PyTorch solves all of this by giving you:
- Ready-made building blocks for neural networks
- Automatic gradient calculation
- Easy GPU support
- A Python-friendly interface

### Why Was PyTorch Created?
Before PyTorch, frameworks like Theano and early TensorFlow were powerful but **hard to debug** and **not very Pythonic**. You had to define a graph first, then run it — making debugging difficult. PyTorch was created to provide:
- **Dynamic computation graphs** (define-as-you-go)
- **Python-native feel** (works like normal Python)
- **Easy debugging** (use standard Python tools)

### Who Developed It?
PyTorch was developed by **Meta AI (Facebook's AI Research lab)** and was released in **2016**. It is now maintained by Meta and a large open-source community under the **PyTorch Foundation** (part of the Linux Foundation).

### Where Is It Commonly Used?
- **Computer Vision** (image classification, object detection)
- **Natural Language Processing** (chatbots, translation, LLMs)
- **Reinforcement Learning**
- **Research** (most AI research papers use PyTorch)
- **Industry** (Tesla, Microsoft, OpenAI, and many more)

### Key Points
- ⭐ PyTorch is an open-source deep learning library for Python.
- ⭐ It was created by Meta AI in 2016.
- ⭐ It solves the problem of building and training neural networks easily.
- ⭐ It is widely used in research and industry.

---

## 2. Why PyTorch?

### Main Features
- **Tensor computation** (like NumPy but with GPU support)
- **Automatic differentiation** (autograd)
- **Dynamic computation graphs**
- **Neural network library** (`torch.nn`)
- **Optimizers** (`torch.optim`)
- **Data loading utilities** (`torch.utils.data`)
- **GPU acceleration** (CUDA support)

### Advantages
- **Pythonic** — feels like writing normal Python
- **Easy to debug** — use `print()`, `pdb`, etc.
- **Dynamic graphs** — change model behavior on the fly
- **Strong community** — huge ecosystem and tutorials
- **Research-friendly** — most new papers release PyTorch code

### Why Researchers and Developers Use It
- Researchers love the **flexibility** and **ease of experimentation**
- Developers love the **production tools** (TorchScript, ONNX, etc.)
- Both love the **large community** and **abundant resources**

### PyTorch vs Traditional Python/NumPy for ML

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| GPU support | ❌ No | ✅ Yes |
| Automatic gradients | ❌ No | ✅ Yes |
| Neural network tools | ❌ No | ✅ Yes |
| Dynamic graphs | ❌ No | ✅ Yes |
| Syntax similarity | ✅ Yes | ✅ Yes (similar to NumPy) |
| Speed on CPU | ✅ Fast | ✅ Fast |
| Speed on GPU | ❌ N/A | ✅ Very fast |

### Key Points
- ⭐ PyTorch combines NumPy-like syntax with GPU and autograd support.
- ⭐ It is preferred over NumPy for deep learning because of GPU and gradient support.
- ⭐ Dynamic graphs make debugging and experimentation easier.

---

## 3. PyTorch Ecosystem

| Component | Purpose |
|-----------|---------|
| **PyTorch (torch)** | Core library: tensors, autograd, neural networks |
| **TorchVision** | Image datasets, models, and transforms |
| **TorchAudio** | Audio datasets, models, and transforms |
| **TorchText** | Text datasets and utilities (⚠️ no longer actively maintained; use HuggingFace or torchtext alternatives) |
| **TorchServe** | Model deployment (advanced — not covered here) |
| **TorchScript** | Convert models for production (advanced) |

### Brief Explanations
- **PyTorch**: The main library you import as `torch`.
- **TorchVision**: Provides popular datasets (CIFAR-10, MNIST), pre-trained models (ResNet, VGG), and image transformations.
- **TorchAudio**: Similar to TorchVision but for audio — datasets, transforms, and pre-trained models.
- **TorchText**: Was used for text data. Its development has slowed; many practitioners now use HuggingFace's `datasets` and `transformers` instead.

### Key Points
- ⭐ PyTorch is the core; TorchVision, TorchAudio, and TorchText are domain-specific extensions.
- ⭐ TorchVision is the most commonly used extension for beginners.
- ⭐ TorchText is no longer actively maintained.

---

## 4. Installing PyTorch

### Basic Installation Using pip

```bash
# CPU-only version (simplest)
pip install torch torchvision torchaudio
```

```bash
# GPU version (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> 💡 Always check the official PyTorch website (pytorch.org) for the exact command for your system.

### CPU vs GPU/CUDA Installation (High Level)
- **CPU version**: Works on any computer. Slower for deep learning but fine for learning.
- **GPU/CUDA version**: Requires an NVIDIA GPU and CUDA drivers. Much faster for training.
- If you don't have an NVIDIA GPU, install the CPU version.

### How to Verify Installation

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

**Explanation:**
- `torch.__version__` prints the installed PyTorch version.
- `torch.cuda.is_available()` returns `True` if a CUDA-capable GPU is available, else `False`.

### Key Points
- ⭐ Install with `pip install torch torchvision torchaudio`.
- ⭐ Use the CPU version if you don't have an NVIDIA GPU.
- ⭐ Verify with `torch.__version__` and `torch.cuda.is_available()`.

---

## 5. Importing PyTorch

```python
import torch
```

### What Does `torch` Provide?
- **Tensor creation**: `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.rand()`
- **Math operations**: `torch.add()`, `torch.matmul()`, etc.
- **Autograd**: `torch.autograd`
- **Neural networks**: `torch.nn`
- **Optimizers**: `torch.optim`
- **Data utilities**: `torch.utils.data`
- **GPU support**: `torch.cuda`

### Key Points
- ⭐ `import torch` is the standard way to use PyTorch.
- ⭐ The `torch` module contains everything from tensors to neural networks.

---

## 6. Tensors — Introduction Only

### What Is a Tensor?
A **tensor** is a multi-dimensional array — the fundamental data structure in PyTorch. Think of it as a generalization of numbers, vectors, and matrices.

### Why Tensors Are Important in PyTorch
- All data (images, text, audio) is converted into tensors.
- Models operate on tensors.
- Gradients are computed on tensors.
- Tensors can live on CPU or GPU.

### Scalar, Vector, Matrix, and Higher-Dimensional Tensors

| Name | Dimensions | Example |
|------|------------|---------|
| Scalar | 0-D | `torch.tensor(5)` |
| Vector | 1-D | `torch.tensor([1, 2, 3])` |
| Matrix | 2-D | `torch.tensor([[1, 2], [3, 4]])` |
| Higher-D | 3-D+ | `torch.tensor([[[1, 2], [3, 4]]])` |

### Creating Simple Tensors

```python
import torch

# From a Python list
a = torch.tensor([1, 2, 3])

# Zeros and ones
b = torch.zeros(2, 3)
c = torch.ones(2, 3)

# Random values
d = torch.rand(2, 3)

# Range of values
e = torch.arange(0, 10, 2)
```

**Explanation:**
- `torch.tensor([1, 2, 3])` creates a 1-D tensor from a list.
- `torch.zeros(2, 3)` creates a 2×3 tensor filled with zeros.
- `torch.ones(2, 3)` creates a 2×3 tensor filled with ones.
- `torch.rand(2, 3)` creates a 2×3 tensor with random values between 0 and 1.
- `torch.arange(0, 10, 2)` creates a tensor `[0, 2, 4, 6, 8]`.

### Basic Tensor Properties

```python
x = torch.rand(2, 3)

print(x.shape)    # torch.Size([2, 3])
print(x.dtype)    # torch.float32
print(x.device)   # cpu
```

**Explanation:**
- `.shape` tells you the dimensions.
- `.dtype` tells you the data type (e.g., `float32`, `int64`).
- `.device` tells you where the tensor lives (`cpu` or `cuda`).

### Very Basic Tensor Operations

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)        # tensor([5, 7, 9])
print(a * b)        # tensor([4, 10, 18])
print(a.sum())      # tensor(6)
print(a.mean())     # tensor(2.)
```

**Explanation:**
- `a + b` adds element-wise.
- `a * b` multiplies element-wise.
- `.sum()` adds all elements.
- `.mean()` computes the average.

### Key Points
- ⭐ Tensors are multi-dimensional arrays — the core data structure of PyTorch.
- ⭐ Key properties: `shape`, `dtype`, `device`.
- ⭐ Basic operations work like NumPy.

---

## 7. NumPy vs PyTorch

### Similarities
- Both use multi-dimensional arrays.
- Both support similar operations (add, multiply, reshape, etc.).
- Both are fast and written in C/C++ under the hood.

### Differences

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| GPU support | ❌ | ✅ |
| Automatic gradients | ❌ | ✅ |
| Deep learning tools | ❌ | ✅ |
| Dynamic graphs | ❌ | ✅ |
| Syntax | Similar | Similar |

### When PyTorch Tensors Are Preferred
- When you need **GPU acceleration**
- When you need **automatic differentiation**
- When building **neural networks**
- When you want to **train models**

### Basic Conversion Between NumPy and Tensors

```python
import numpy as np
import torch

# NumPy → PyTorch
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# PyTorch → NumPy
tensor2 = torch.tensor([4, 5, 6])
np_array2 = tensor2.numpy()
```

**Explanation:**
- `torch.from_numpy()` converts a NumPy array to a tensor (shares memory).
- `.numpy()` converts a tensor to a NumPy array (shares memory on CPU).

### Key Points
- ⭐ NumPy and PyTorch are similar, but PyTorch adds GPU and autograd.
- ⭐ Use `torch.from_numpy()` and `.numpy()` for conversion.
- ⭐ PyTorch tensors are preferred for deep learning.

---

## 8. CPU and GPU

### What CPU and GPU Mean in PyTorch
- **CPU**: Central Processing Unit — good for general tasks, but slower for deep learning.
- **GPU**: Graphics Processing Unit — has thousands of cores, great for parallel math.

### Why GPUs Are Useful for Deep Learning
Deep learning involves **huge matrix multiplications**. GPUs can perform thousands of these operations in parallel, making training **10–100x faster** than CPU.

### What "Device" Means
A **device** is where a tensor or model lives — either `cpu` or `cuda` (GPU).

### Basic Example of Moving a Tensor to a Device

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.tensor([1, 2, 3])
x = x.to(device)
print(x.device)
```

**Explanation:**
- `torch.device(...)` picks GPU if available, else CPU.
- `.to(device)` moves the tensor to that device.
- `x.device` shows where the tensor currently lives.

### Key Points
- ⭐ GPU is much faster than CPU for deep learning.
- ⭐ "Device" means CPU or GPU.
- ⭐ Use `.to(device)` to move tensors.

---

## 9. Autograd — Introduction

### What Automatic Differentiation Means
Automatic differentiation (autograd) is PyTorch's ability to **automatically compute gradients** for you. You don't need to do calculus by hand.

### Why Gradients Are Needed in Machine Learning
Gradients tell us **how to change model parameters** to reduce error. They are the foundation of training.

### What `requires_grad` Means
If `requires_grad=True`, PyTorch tracks all operations on that tensor so it can compute gradients later.

### Very Small Example

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()
print(x.grad)
```

**Explanation:**
- `x` is a tensor with `requires_grad=True`.
- `y = x² + 3x + 1`.
- `y.backward()` computes the gradient of `y` with respect to `x`.
- `x.grad` is `2x + 3 = 7` when `x = 2`.

### Key Points
- ⭐ Autograd automatically computes gradients.
- ⭐ Gradients are needed to train models.
- ⭐ `requires_grad=True` enables tracking.

---

## 10. PyTorch and Deep Learning

### How PyTorch Fits into a Typical Deep Learning Workflow

```
Dataset → Model → Prediction → Loss → Gradient → Optimization
```

### Conceptual Overview
1. **Dataset**: Your data (images, text, etc.).
2. **Model**: A neural network that makes predictions.
3. **Prediction**: The model's output.
4. **Loss**: A number measuring how wrong the prediction is.
5. **Gradient**: How to change parameters to reduce loss.
6. **Optimization**: Update parameters using gradients.

This loop repeats many times until the model is good.

### Key Points
- ⭐ The workflow is: data → model → prediction → loss → gradient → update.
- ⭐ PyTorch handles gradients and optimization for you.

---

## 11. Basic PyTorch Workflow

### Very Small Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Create data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. Create a simple model
model = nn.Linear(1, 1)

# 3. Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training loop
for epoch in range(100):
    # Prediction
    y_pred = model(x)
    
    # Loss
    loss = loss_fn(y_pred, y)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()

# 5. Check result
print(model(torch.tensor([[5.0]])))
```

### Step-by-Step Explanation
1. **Create data**: `x` is input, `y` is target (y = 2x).
2. **Model**: `nn.Linear(1, 1)` is a simple linear layer (y = wx + b).
3. **Loss**: `MSELoss` measures mean squared error.
4. **Optimizer**: `SGD` updates parameters using gradients.
5. **Training loop**:
   - Predict
   - Compute loss
   - Zero old gradients
   - Compute new gradients (`backward`)
   - Update parameters (`step`)
6. **Check**: After training, `model(5)` should be close to `10`.

### Key Points
- ⭐ The training loop repeats: predict → loss → backward → step.
- ⭐ `optimizer.zero_grad()` prevents gradient accumulation.
- ⭐ `loss.backward()` computes gradients; `optimizer.step()` updates weights.

---

## 12. Important PyTorch Terminology

| Term | Simple Meaning |
|------|----------------|
| **Tensor** | Multi-dimensional array (core data structure) |
| **Model** | A neural network that makes predictions |
| **Dataset** | A collection of data samples |
| **DataLoader** | Loads data in batches during training |
| **Loss function** | Measures how wrong the model's predictions are |
| **Optimizer** | Updates model parameters using gradients |
| **Gradient** | Direction and amount to change parameters |
| **Autograd** | PyTorch's automatic gradient system |
| **Epoch** | One full pass through the training data |
| **Batch** | A small group of samples processed together |
| **Parameter** | Learnable value in a model (weights, biases) |
| **Device** | Where tensors live (CPU or GPU) |

### Key Points
- ⭐ These terms appear in almost every PyTorch project.
- ⭐ Understanding them makes reading PyTorch code much easier.

---

## 13. PyTorch vs TensorFlow — Introduction

### What PyTorch Is
A deep learning framework known for its **Pythonic feel**, **dynamic graphs**, and **ease of debugging**.

### What TensorFlow Is
A deep learning framework developed by **Google**, known for **production deployment** and **TensorBoard** visualization.

### Basic Differences

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| Developer | Meta | Google |
| Graph type | Dynamic | Static (and dynamic via Eager) |
| Debugging | Easy (Pythonic) | Historically harder |
| Research popularity | Very high | Moderate |
| Production tools | Growing | Mature |
| Learning curve | Gentle | Moderate |

> ⚠️ Neither is universally better. Choose based on your needs.

### Key Points
- ⭐ PyTorch is preferred in research; TensorFlow is strong in production.
- ⭐ Both are powerful; choice depends on context.

---

## 14. Common Beginner Mistakes

- Forgetting to call `optimizer.zero_grad()`
- Forgetting `loss.backward()` before `optimizer.step()`
- Mixing CPU and GPU tensors
- Using `torch.tensor()` when `torch.Tensor()` is meant (they differ)
- Not setting `requires_grad=True` when needed
- Confusing `model.train()` and `model.eval()`
- Forgetting to move model and data to the same device
- Using Python lists instead of tensors
- Not reshaping inputs correctly
- Ignoring `dtype` mismatches (e.g., `float32` vs `float64`)
- Overcomplicating the first model
- Not checking `torch.cuda.is_available()`
- Forgetting `with torch.no_grad()` during evaluation
- Using too high a learning rate
- Not normalizing input data

### Key Points
- ⭐ Most beginner errors come from forgetting `zero_grad`, `backward`, or device placement.
- ⭐ Always check shapes, dtypes, and devices.

---

## 15. What to Learn Next (Roadmap)

1. **Tensors** — master creation, indexing, reshaping, broadcasting
2. **Datasets and DataLoaders** — load and batch data efficiently
3. **Neural Networks** — understand layers and activations
4. **`nn.Module`** — build custom models
5. **Loss Functions** — MSE, CrossEntropy, etc.
6. **Optimizers** — SGD, Adam, etc.
7. **Training Loops** — write full training pipelines
8. **Validation/Testing** — evaluate model performance
9. **CNNs** — convolutional neural networks for images
10. **Transfer Learning** — use pre-trained models
11. **Saving/Loading Models** — persist trained models

### Key Points
- ⭐ Follow this order for a smooth learning path.
- ⭐ Don't rush — master each step before moving on.

---

## Quick Revision

- **PyTorch** = open-source deep learning library by Meta (2016)
- **Tensor** = multi-dimensional array (core data structure)
- **Autograd** = automatic gradient computation
- **Device** = CPU or GPU
- **Training loop** = predict → loss → backward → step
- **Ecosystem** = torch, torchvision, torchaudio, torchtext
- **Install** = `pip install torch torchvision torchaudio`
- **Verify** = `torch.__version__`, `torch.cuda.is_available()`
- **NumPy ↔ Tensor** = `torch.from_numpy()`, `.numpy()`
- **requires_grad** = track gradients for a tensor

---

## 15 Important Beginner-Level Questions with Short Answers

1. **What is PyTorch?**  
   An open-source deep learning library for Python.

2. **Who developed PyTorch?**  
   Meta AI (Facebook).

3. **What is a tensor?**  
   A multi-dimensional array — PyTorch's core data structure.

4. **What does `requires_grad=True` do?**  
   Tells PyTorch to track operations for gradient computation.

5. **What is autograd?**  
   PyTorch's automatic differentiation system.

6. **What is a device?**  
   Where a tensor lives — CPU or GPU.

7. **How do you move a tensor to GPU?**  
   `tensor.to("cuda")`.

8. **What is an epoch?**  
   One full pass through the training dataset.

9. **What is a batch?**  
   A small group of samples processed together.

10. **What does `optimizer.step()` do?**  
    Updates model parameters using gradients.

11. **Why call `optimizer.zero_grad()`?**  
    To clear old gradients before computing new ones.

12. **What is a loss function?**  
    Measures how wrong the model's predictions are.

13. **What is TorchVision used for?**  
    Image datasets, models, and transforms.

14. **How do you convert a NumPy array to a tensor?**  
    `torch.from_numpy(array)`.

15. **What is the typical training loop order?**  
    Predict → Loss → zero_grad → backward → step.

---

## 15 MCQs with Answers

1. **PyTorch was developed by:**  
   a) Google  b) Meta  c) Microsoft  d) Amazon  
   **Answer: b**

2. **Which is the core data structure in PyTorch?**  
   a) List  b) Array  c) Tensor  d) DataFrame  
   **Answer: c**

3. **Which command installs PyTorch?**  
   a) `pip install pytorch`  b) `pip install torch`  c) `install torch`  d) `apt install torch`  
   **Answer: b**

4. **What does `torch.cuda.is_available()` return?**  
   a) True/False  b) GPU name  c) Version  d) None  
   **Answer: a**

5. **What is autograd used for?**  
   a) Data loading  b) Gradients  c) Plotting  d) Saving models  
   **Answer: b**

6. **Which is NOT a PyTorch ecosystem component?**  
   a) TorchVision  b) TorchAudio  c) TorchText  d) TorchSQL  
   **Answer: d**

7. **What does `.shape` return?**  
   a) Data type  b) Device  c) Dimensions  d) Value  
   **Answer: c**

8. **Which converts a tensor to NumPy?**  
   a) `.to_numpy()`  b) `.numpy()`  c) `.array()`  d) `.list()`  
   **Answer: b**

9. **What does `requires_grad=True` enable?**  
   a) GPU use  b) Gradient tracking  c) Faster speed  d) Data loading  
   **Answer: b**

10. **Which is a loss function?**  
    a) SGD  b) MSELoss  c) Linear  d) ReLU  
    **Answer: b**

11. **What does `optimizer.step()` do?**  
    a) Loads data  b) Updates weights  c) Computes loss  d) Clears gradients  
    **Answer: b**

12. **What is an epoch?**  
    a) One batch  b) One full pass  c) One gradient  d) One layer  
    **Answer: b**

13. **Which device is faster for deep learning?**  
    a) CPU  b) GPU  c) RAM  d) Disk  
    **Answer: b**

14. **What does `torch.zeros(2,3)` create?**  
    a) 2×3 ones  b) 2×3 zeros  c) 3×2 zeros  d) Random  
    **Answer: b**

15. **Which is NOT a common beginner mistake?**  
    a) Forgetting zero_grad  b) Mixing CPU/GPU  c) Using tensors  d) Wrong dtype  
    **Answer: c**

---

## PyTorch in One Page

- **PyTorch** = open-source deep learning library by Meta (2016)
- **Core structure** = Tensor (multi-dimensional array)
- **Key features** = GPU support, autograd, dynamic graphs
- **Ecosystem** = torch, torchvision, torchaudio, torchtext
- **Install** = `pip install torch torchvision torchaudio`
- **Verify** = `torch.__version__`, `torch.cuda.is_available()`
- **Device** = CPU or GPU (`tensor.to(device)`)
- **Autograd** = automatic gradients (`requires_grad=True`)
- **Training loop** = predict → loss → zero_grad → backward → step
- **NumPy ↔ Tensor** = `torch.from_numpy()`, `.numpy()`
- **Workflow** = Dataset → Model → Prediction → Loss → Gradient → Optimization
- **Learn next** = Tensors → DataLoaders → nn.Module → Training loops → CNNs → Transfer learning
- **Common mistakes** = forgetting zero_grad, mixing devices, wrong dtype
- **PyTorch vs TensorFlow** = PyTorch for research, TF for production
- **Goal** = Understand what PyTorch is, why it's used, and how to start building models
