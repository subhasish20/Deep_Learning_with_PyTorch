# PyTorch Tensors 

---

## 1. WHAT IS A TENSOR?

### What Is a Tensor?
A **tensor** is a multi-dimensional array — a container for numbers arranged in a grid-like structure. It is the fundamental data structure in PyTorch.

Think of it as a generalization of:
- A single number (scalar)
- A list of numbers (vector)
- A table of numbers (matrix)
- A cube of numbers (3D tensor)
- And beyond...

### Why Tensors Are Important in PyTorch
- **All data** in PyTorch is represented as tensors (images, text, audio, etc.)
- **All operations** (math, reshaping, slicing) work on tensors
- **Gradients** (for learning) are computed on tensors
- **GPU acceleration** works on tensors
- **Neural networks** take tensors as input and produce tensors as output

### Tensor vs Normal Python Numbers/Lists

| Feature | Python List | PyTorch Tensor |
|---------|-------------|----------------|
| Math operations | Manual loops | Built-in, fast |
| GPU support | ❌ No | ✅ Yes |
| Gradient tracking | ❌ No | ✅ Yes |
| Shape information | ❌ No | ✅ Yes |
| Memory efficiency | Low | High |
| Speed | Slow | Fast |

### Tensor vs NumPy Array

| Feature | NumPy Array | PyTorch Tensor |
|---------|-------------|----------------|
| GPU support | ❌ No | ✅ Yes |
| Gradient tracking | ❌ No | ✅ Yes |
| Deep learning tools | ❌ No | ✅ Yes |
| Syntax | Similar | Similar |
| Speed (CPU) | Fast | Fast |

### Why Deep Learning Frameworks Use Tensors
- **Efficient**: Optimized C/C++ backend
- **Parallel**: Can use GPU for massive speedup
- **Automatic differentiation**: Tracks operations for gradients
- **Flexible**: Handles any dimensionality
- **Standardized**: All frameworks use tensors

### The Relationship: Scalar → Vector → Matrix → Higher-Dimensional Tensor

```
Scalar (0D)     Vector (1D)     Matrix (2D)      3D Tensor
   ●            [●, ●, ●]      [●, ●, ●]      [ [●, ●], [●, ●] ]
                               [●, ●, ●]      [ [●, ●], [●, ●] ]
```

| Name | Dimensions | Example | PyTorch Code |
|------|------------|---------|--------------|
| Scalar | 0D | A single number | `torch.tensor(5)` |
| Vector | 1D | A list of numbers | `torch.tensor([1, 2, 3])` |
| Matrix | 2D | A table | `torch.tensor([[1, 2], [3, 4]])` |
| 3D Tensor | 3D | A cube of numbers | `torch.tensor([[[1, 2], [3, 4]]])` |
| N-D Tensor | N-D | Any dimension | `torch.rand(2, 3, 4, 5)` |

### ⭐ Key Points
- ⭐ A tensor is a multi-dimensional array — the core data structure of PyTorch.
- ⭐ Tensors support GPU, gradients, and fast math.
- ⭐ Scalar (0D) → Vector (1D) → Matrix (2D) → Higher-D (N-D).
- ⭐ All PyTorch data is tensors.

---

## 2. CREATING TENSORS

### 1. `torch.tensor()`
**What it does:** Creates a tensor from Python data (list, tuple, number).

**Syntax:**
```python
torch.tensor(data, dtype=None, device=None, requires_grad=False)
```

**Example:**
```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
c = torch.tensor(5)
```

**Expected Output:**
```
tensor([1, 2, 3])
tensor([[1., 2.],
        [3., 4.]])
tensor(5)
```

**Explanation:**
- `a` is a 1D tensor of integers.
- `b` is a 2D tensor of floats.
- `c` is a 0D tensor (scalar).

**When useful:** When you have existing Python data and want to convert it.

---

### 2. `torch.zeros()`
**What it does:** Creates a tensor filled with zeros.

**Syntax:**
```python
torch.zeros(*size, dtype=None, device=None)
```

**Example:**
```python
x = torch.zeros(2, 3)
```

**Expected Output:**
```
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

**Explanation:** Creates a 2×3 tensor of zeros.

**When useful:** Initializing weights, masks, or placeholders.

---

### 3. `torch.ones()`
**What it does:** Creates a tensor filled with ones.

**Syntax:**
```python
torch.ones(*size, dtype=None, device=None)
```

**Example:**
```python
x = torch.ones(2, 3)
```

**Expected Output:**
```
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

**When useful:** Creating masks, initializing values.

---

### 4. `torch.empty()`
**What it does:** Creates a tensor with **uninitialized** (garbage) values.

**Syntax:**
```python
torch.empty(*size, dtype=None, device=None)
```

**Example:**
```python
x = torch.empty(2, 3)
```

**Expected Output:** (values are random garbage)
```
tensor([[1.2e-10, 3.4e-20, 5.6e-30],
        [7.8e-40, 9.0e-50, 1.1e-60]])
```

**Explanation:** The values are whatever was in memory. You must fill them later.

**When useful:** When you will overwrite all values immediately (faster than zeros).

---

### 5. `torch.full()`
**What it does:** Creates a tensor filled with a specific value.

**Syntax:**
```python
torch.full(size, fill_value, dtype=None, device=None)
```

**Example:**
```python
x = torch.full((2, 3), 7)
```

**Expected Output:**
```
tensor([[7, 7, 7],
        [7, 7, 7]])
```

**When useful:** When you need a constant tensor (e.g., all 0.5).

---

### 6. `torch.arange()`
**What it does:** Creates a tensor with a range of values.

**Syntax:**
```python
torch.arange(start, end, step, dtype=None)
```

**Example:**
```python
x = torch.arange(0, 10, 2)
```

**Expected Output:**
```
tensor([0, 2, 4, 6, 8])
```

**Explanation:** Starts at 0, ends before 10, step 2.

**When useful:** Creating sequences, indices.

---

### 7. `torch.linspace()`
**What it does:** Creates a tensor with evenly spaced values between start and end (inclusive).

**Syntax:**
```python
torch.linspace(start, end, steps)
```

**Example:**
```python
x = torch.linspace(0, 1, 5)
```

**Expected Output:**
```
tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

**Explanation:** 5 evenly spaced values from 0 to 1 (inclusive).

**When useful:** Plotting, sampling, creating smooth ranges.

---

### 8. `torch.eye()`
**What it does:** Creates an identity matrix (ones on diagonal, zeros elsewhere).

**Syntax:**
```python
torch.eye(n, m=None)
```

**Example:**
```python
x = torch.eye(3)
```

**Expected Output:**
```
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
```

**When useful:** Linear algebra, initialization.

---

### 9. `torch.rand()`
**What it does:** Creates a tensor with random values from a **uniform distribution** [0, 1).

**Syntax:**
```python
torch.rand(*size)
```

**Example:**
```python
x = torch.rand(2, 3)
```

**Expected Output:**
```
tensor([[0.1234, 0.5678, 0.9012],
        [0.3456, 0.7890, 0.2345]])
```

**When useful:** Random initialization, sampling.

---

### 10. `torch.randn()`
**What it does:** Creates a tensor with random values from a **standard normal distribution** (mean 0, std 1).

**Syntax:**
```python
torch.randn(*size)
```

**Example:**
```python
x = torch.randn(2, 3)
```

**Expected Output:**
```
tensor([[ 0.1234, -0.5678,  1.9012],
        [-0.3456,  0.7890, -1.2345]])
```

**When useful:** Weight initialization in neural networks.

---

### 11. `torch.randint()`
**What it does:** Creates a tensor with random **integers** in a given range.

**Syntax:**
```python
torch.randint(low, high, size)
```

**Example:**
```python
x = torch.randint(0, 10, (2, 3))
```

**Expected Output:**
```
tensor([[3, 7, 1],
        [9, 0, 5]])
```

**When useful:** Random indices, discrete data.

---

### Difference Between `rand()`, `randn()`, and `randint()`

| Function | Distribution | Range | Type |
|----------|-------------|-------|------|
| `torch.rand()` | Uniform | [0, 1) | Float |
| `torch.randn()` | Normal (Gaussian) | Any real | Float |
| `torch.randint()` | Uniform | [low, high) | Integer |

### ⭐ Key Points
- ⭐ Use `torch.tensor()` for existing data.
- ⭐ Use `zeros()`, `ones()`, `full()` for constant tensors.
- ⭐ Use `rand()`, `randn()`, `randint()` for random tensors.
- ⭐ `arange()` for sequences; `linspace()` for evenly spaced.

---

## 3. TENSOR ATTRIBUTES

### Shape (`.shape`)
**What it means:** The dimensions of the tensor.

```python
x = torch.rand(2, 3, 4)
print(x.shape)  # torch.Size([2, 3, 4])
```

**Explanation:** 2 blocks, 3 rows, 4 columns.

---

### Number of Dimensions (`.ndim`)
**What it means:** How many dimensions the tensor has.

```python
x = torch.rand(2, 3, 4)
print(x.ndim)  # 3
```

---

### Data Type (`.dtype`)
**What it means:** The type of numbers stored (float32, int64, etc.).

```python
x = torch.tensor([1, 2, 3])
print(x.dtype)  # torch.int64

y = torch.tensor([1.0, 2.0])
print(y.dtype)  # torch.float32
```

---

### Device (`.device`)
**What it means:** Where the tensor lives (CPU or GPU).

```python
x = torch.tensor([1, 2, 3])
print(x.device)  # cpu

# If GPU available:
# x = x.to("cuda")
# print(x.device)  # cuda:0
```

---

### Requires Gradient (`.requires_grad`)
**What it means:** Whether PyTorch tracks operations for gradient computation.

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
print(x.requires_grad)  # True
```

### ⭐ Key Points
- ⭐ `.shape` = dimensions; `.ndim` = number of dimensions.
- ⭐ `.dtype` = data type; `.device` = CPU/GPU.
- ⭐ `.requires_grad` = gradient tracking flag.

---

## 4. TENSOR DATA TYPES

### Common PyTorch dtypes

| dtype | Description | Typical Use |
|-------|-------------|-------------|
| `torch.float32` | 32-bit float | Default for neural networks |
| `torch.float64` | 64-bit float | High precision |
| `torch.float16` | 16-bit float | Mixed precision (advanced) |
| `torch.bfloat16` | 16-bit brain float | Modern GPUs |
| `torch.int8` | 8-bit integer | Quantization |
| `torch.int16` | 16-bit integer | Rare |
| `torch.int32` | 32-bit integer | Indices |
| `torch.int64` | 64-bit integer | Default for integers |
| `torch.bool` | Boolean | Masks |

### Why dtype Matters
- **Memory**: Smaller dtypes use less memory.
- **Speed**: Some dtypes are faster on certain hardware.
- **Precision**: Float64 is more precise than Float32.
- **Compatibility**: Some operations require specific dtypes.

### How to Specify dtype

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.zeros(2, 3, dtype=torch.int64)
```

### How to Change dtype

```python
x = torch.tensor([1.5, 2.5, 3.5])

# Convert to int (truncates)
x_int = x.int()
print(x_int)  # tensor([1, 2, 3])

# Convert to float64
x_double = x.double()
print(x_double.dtype)  # torch.float64

# Convert to bool
x_bool = x.bool()
print(x_bool)  # tensor([True, True, True])

# Generic conversion
x_long = x.to(torch.int64)
print(x_long.dtype)  # torch.int64
```

### Common dtype Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Using int for division | Integer division truncates | Convert to float first |
| Mixing float32 and float64 | Error or unexpected results | Use `.to(torch.float32)` |
| Using float for indices | Index must be int | Use `.long()` |
| Forgetting dtype for labels | Loss function expects long | Use `.long()` |

### Why Neural Networks Use Floating-Point Tensors
- Weights and biases are **continuous values**.
- Gradients require **decimal precision**.
- Float32 is the standard balance of speed and precision.

### ⭐ Key Points
- ⭐ `float32` is the default for neural networks.
- ⭐ `int64` is the default for integers.
- ⭐ Use `.float()`, `.int()`, `.long()`, `.bool()` to convert.
- ⭐ Mismatched dtypes cause errors.

---

## 5. TENSOR SHAPE

### 0D Tensor (Scalar)
```python
x = torch.tensor(5)
print(x.shape)  # torch.Size([])
```
**Meaning:** A single number.

---

### 1D Tensor (Vector)
```python
x = torch.tensor([1, 2, 3])
print(x.shape)  # torch.Size([3])
```
**Meaning:** A list of 3 numbers.

---

### 2D Tensor (Matrix)
```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape)  # torch.Size([2, 3])
```
**Meaning:** 2 rows, 3 columns.

---

### 3D Tensor
```python
x = torch.rand(2, 3, 4)
print(x.shape)  # torch.Size([2, 3, 4])
```
**Meaning:** 2 blocks, each 3×4.

---

### Higher-Dimensional Tensors

| Shape | Meaning | Example |
|-------|---------|---------|
| `[3]` | Vector of 3 | A point in 3D space |
| `[2, 3]` | 2×3 matrix | A small table |
| `[2, 3, 4]` | 3D tensor | 2 images of 3×4 pixels |
| `[32, 3, 224, 224]` | 4D tensor | Batch of 32 RGB images (224×224) |
| `[32, 10]` | 2D tensor | Batch of 32 samples, 10 classes |

### How to Read Shapes
- **Rightmost dimension** = fastest changing (columns).
- **Leftmost dimension** = slowest changing (batch).
- For images: `[batch, channels, height, width]`.

### ⭐ Key Points
- ⭐ Shape tells you the size of each dimension.
- ⭐ `[32, 3, 224, 224]` = batch of 32 RGB images.
- ⭐ Always check shape before operations.

---

## 6. INDEXING AND SLICING

### Single Element Access
```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x[0, 1])  # tensor(2)
```

### Row Access
```python
print(x[0])  # tensor([1, 2, 3])
print(x[0].shape)  # torch.Size([3])
```

### Column Access
```python
print(x[:, 1])  # tensor([2, 5])
print(x[:, 1].shape)  # torch.Size([2])
```

### Multiple Dimensions
```python
x = torch.rand(2, 3, 4)
print(x[0, 1, 2])  # scalar
print(x[0, 1].shape)  # torch.Size([4])
```

### Slicing
```python
x = torch.tensor([0, 1, 2, 3, 4, 5])
print(x[1:4])   # tensor([1, 2, 3])
print(x[:3])    # tensor([0, 1, 2])
print(x[3:])    # tensor([3, 4, 5])
```

### Negative Indexing
```python
print(x[-1])    # tensor(5)
print(x[-3:])   # tensor([3, 4, 5])
```

### Step Slicing
```python
print(x[::2])   # tensor([0, 2, 4])
print(x[1::2])  # tensor([1, 3, 5])
```

### Shape After Indexing/Slicing

| Operation | Original Shape | Result Shape |
|-----------|---------------|--------------|
| `x[0]` | `[2, 3]` | `[3]` |
| `x[:, 1]` | `[2, 3]` | `[2]` |
| `x[0:1]` | `[2, 3]` | `[1, 3]` |
| `x[0, 1]` | `[2, 3]` | `[]` (scalar) |

### ⭐ Key Points
- ⭐ Use `x[i, j]` for 2D indexing.
- ⭐ Use `:` for all elements in a dimension.
- ⭐ Slicing returns a view (shares memory).
- ⭐ Negative indices count from the end.

---

## 7. BASIC TENSOR OPERATIONS

### Element-wise Operations

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
print(a + b)  # tensor([5, 7, 9])

# Subtraction
print(a - b)  # tensor([-3, -3, -3])

# Multiplication (element-wise)
print(a * b)  # tensor([4, 10, 18])

# Division
print(a / b)  # tensor([0.2500, 0.4000, 0.5000])

# Power
print(a ** 2)  # tensor([1, 4, 9])

# Modulo
print(b % a)  # tensor([0, 1, 0])
```

### Element-wise Multiplication vs Matrix Multiplication

| Operation | Symbol | Meaning |
|-----------|--------|---------|
| Element-wise | `*` | Multiply corresponding elements |
| Matrix | `@` or `torch.matmul()` | Dot product of rows and columns |

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Element-wise
print(a * b)
# tensor([[ 5, 12],
#         [21, 32]])

# Matrix multiplication
print(a @ b)
# tensor([[19, 22],
#         [43, 50]])
```

### ⭐ Key Points
- ⭐ `+`, `-`, `*`, `/` are element-wise.
- ⭐ `*` is NOT matrix multiplication.
- ⭐ Use `@` or `torch.matmul()` for matrix multiplication.

---

## 8. MATRIX MULTIPLICATION

### `torch.matmul()` and `@`

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape [2, 3]
b = torch.tensor([[1, 2], [3, 4], [5, 6]])  # shape [3, 2]

c = torch.matmul(a, b)
print(c)
# tensor([[22, 28],
#         [49, 64]])
print(c.shape)  # torch.Size([2, 2])
```

### `torch.mm()` (2D only)

```python
c = torch.mm(a, b)  # Same as matmul for 2D
```

### Shape Rule

```
(m × n) @ (n × p) → (m × p)
```

The **inner dimensions must match**.

### Incompatible Shapes Example

```python
a = torch.rand(2, 3)
b = torch.rand(4, 5)

# torch.matmul(a, b)  # ERROR: 3 != 4
```

**Error:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 4x5)`

**Fix:** Ensure inner dimensions match. Transpose if needed:
```python
b = torch.rand(3, 5)
c = torch.matmul(a, b)  # Works: [2, 3] @ [3, 5] → [2, 5]
```

### ⭐ Key Points
- ⭐ `@` and `torch.matmul()` do matrix multiplication.
- ⭐ Shape rule: `(m, n) @ (n, p) → (m, p)`.
- ⭐ Inner dimensions must match.
- ⭐ `torch.mm()` is for 2D only.

---

## 9. COMMON TENSOR FUNCTIONS

### `torch.sum()`
```python
x = torch.tensor([[1, 2], [3, 4]])
print(torch.sum(x))  # tensor(10)
```

### `torch.mean()`
```python
print(torch.mean(x.float()))  # tensor(2.5000)
```

### `torch.max()` and `torch.min()`
```python
print(torch.max(x))  # tensor(4)
print(torch.min(x))  # tensor(1)
```

### `torch.argmax()` and `torch.argmin()`
```python
print(torch.argmax(x))  # tensor(3) — index of max
print(torch.argmin(x))  # tensor(0) — index of min
```

### `torch.abs()`
```python
x = torch.tensor([-1, -2, 3])
print(torch.abs(x))  # tensor([1, 2, 3])
```

### `torch.sqrt()`
```python
x = torch.tensor([4.0, 9.0, 16.0])
print(torch.sqrt(x))  # tensor([2., 3., 4.])
```

### `torch.exp()`
```python
x = torch.tensor([0.0, 1.0, 2.0])
print(torch.exp(x))  # tensor([1.0000, 2.7183, 7.3891])
```

### `torch.log()`
```python
x = torch.tensor([1.0, 2.7183, 7.3891])
print(torch.log(x))  # tensor([0.0000, 1.0000, 2.0000])
```

### Important Parameters: `dim`
```python
x = torch.tensor([[1, 2], [3, 4]])
print(torch.sum(x, dim=0))  # tensor([4, 6]) — sum over rows
print(torch.sum(x, dim=1))  # tensor([3, 7]) — sum over columns
```

### ⭐ Key Points
- ⭐ `sum()`, `mean()`, `max()`, `min()` reduce tensors.
- ⭐ `argmax()`, `argmin()` return indices.
- ⭐ `abs()`, `sqrt()`, `exp()`, `log()` are element-wise.
- ⭐ `dim` controls which axis to reduce.

---

## 10. DIMENSIONS AND `dim`

### The Concept of `dim`
`dim` specifies **which dimension** an operation acts on.

### Simple Matrix Example

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(torch.sum(x))        # tensor(21) — sum all
print(torch.sum(x, dim=0)) # tensor([5, 7, 9]) — sum over rows
print(torch.sum(x, dim=1)) # tensor([6, 15]) — sum over columns
```

### Visual Explanation

```
x = [[1, 2, 3],    dim=0 is the row dimension (down)
     [4, 5, 6]]    dim=1 is the column dimension (across)

dim=0: sum down each column → [1+4, 2+5, 3+6] = [5, 7, 9]
dim=1: sum across each row  → [1+2+3, 4+5+6] = [6, 15]
```

### Higher-Dimensional Example

```python
x = torch.rand(2, 3, 4)  # shape [2, 3, 4]

print(x.sum(dim=0).shape)  # [3, 4]
print(x.sum(dim=1).shape)  # [2, 4]
print(x.sum(dim=2).shape)  # [2, 3]
```

**Rule:** The dimension you sum over is **removed** from the shape.

### ⭐ Key Points
- ⭐ `dim=0` operates on rows; `dim=1` operates on columns.
- ⭐ The specified dimension is removed in the result.
- ⭐ Essential for batch operations.

---

## 11. RESHAPING TENSORS

### `reshape()`
**What it does:** Changes the shape without changing data.

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # [2, 3]
y = x.reshape(3, 2)
print(y)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
```

### `view()`
**What it does:** Similar to `reshape()` but requires contiguous memory.

```python
y = x.view(3, 2)  # Works if x is contiguous
```

### `flatten()`
**What it does:** Flattens to 1D.

```python
x = torch.tensor([[1, 2], [3, 4]])
print(x.flatten())  # tensor([1, 2, 3, 4])
```

### `squeeze()`
**What it does:** Removes dimensions of size 1.

```python
x = torch.rand(1, 3, 1)
print(x.shape)          # [1, 3, 1]
print(x.squeeze().shape) # [3]
```

### `unsqueeze()`
**What it does:** Adds a dimension of size 1.

```python
x = torch.tensor([1, 2, 3])  # [3]
print(x.unsqueeze(0).shape)  # [1, 3]
print(x.unsqueeze(1).shape)  # [3, 1]
```

### `reshape()` vs `view()`

| Feature | `reshape()` | `view()` |
|---------|-------------|----------|
| Works on non-contiguous | ✅ Yes | ❌ No |
| Returns view when possible | ✅ Yes | ✅ Always |
| Safer | ✅ Yes | ❌ Can error |

**Beginner advice:** Use `reshape()` unless you know the tensor is contiguous.

### ⭐ Key Points
- ⭐ `reshape()` and `view()` change shape.
- ⭐ `flatten()` → 1D.
- ⭐ `squeeze()` removes size-1 dims; `unsqueeze()` adds them.
- ⭐ `reshape()` is safer than `view()`.

---

## 12. TRANSPOSE AND PERMUTE

### `transpose()`
**What it does:** Swaps two dimensions.

```python
x = torch.rand(2, 3)
print(x.transpose(0, 1).shape)  # [3, 2]
```

### `permute()`
**What it does:** Rearranges all dimensions.

```python
x = torch.rand(2, 3, 4)
print(x.permute(2, 0, 1).shape)  # [4, 2, 3]
```

### Difference

| Function | Purpose |
|----------|---------|
| `transpose()` | Swap exactly two dimensions |
| `permute()` | Reorder all dimensions |

### Why Change Dimensions?
- Align shapes for matrix multiplication.
- Convert image formats: `[H, W, C]` → `[C, H, W]`.
- Prepare data for specific layers.

### ⭐ Key Points
- ⭐ `transpose()` swaps two dims.
- ⭐ `permute()` reorders all dims.
- ⭐ Useful for shape alignment.

---

## 13. CONCATENATION AND STACKING

### `torch.cat()`
**What it does:** Joins tensors along an existing dimension.

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

print(torch.cat([a, b], dim=0).shape)  # [4, 2]
print(torch.cat([a, b], dim=1).shape)  # [2, 4]
```

### `torch.stack()`
**What it does:** Joins tensors along a **new** dimension.

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(torch.stack([a, b]).shape)       # [2, 3]
print(torch.stack([a, b], dim=1).shape) # [3, 2]
```

### Difference

| Function | Dimension | Shape Change |
|----------|-----------|--------------|
| `cat()` | Existing | Sum along that dim |
| `stack()` | New | Adds a dimension |

### ⭐ Key Points
- ⭐ `cat()` joins along existing dim.
- ⭐ `stack()` creates a new dim.
- ⭐ Shapes must match (except along concat dim).

---

## 14. BROADCASTING

### What Is Broadcasting?
Broadcasting allows PyTorch to perform operations on tensors of **different shapes** by automatically expanding the smaller one.

### Why PyTorch Uses Broadcasting
- Avoids unnecessary memory copies.
- Makes code cleaner.
- Enables efficient operations.

### Simple Examples

```python
a = torch.tensor([[1, 2], [3, 4]])  # [2, 2]
b = torch.tensor([10, 20])          # [2]

print(a + b)
# tensor([[11, 22],
#         [13, 24]])
```

**Explanation:** `b` is broadcast to `[[10, 20], [10, 20]]`.

### When Broadcasting Works
- Dimensions are equal, OR
- One dimension is 1, OR
- One tensor has fewer dimensions.

### When Broadcasting Fails

```python
a = torch.rand(2, 3)
b = torch.rand(2, 4)

# a + b  # ERROR: 3 != 4 and neither is 1
```

### ⭐ Key Points
- ⭐ Broadcasting expands smaller tensors automatically.
- ⭐ Dimensions must be compatible (equal or 1).
- ⭐ Saves memory and simplifies code.

---

## 15. COPYING TENSORS

### Assignment
```python
a = torch.tensor([1, 2, 3])
b = a  # b points to same data
b[0] = 99
print(a)  # tensor([99, 2, 3]) — a changed!
```

### `clone()`
```python
a = torch.tensor([1, 2, 3])
b = a.clone()  # independent copy
b[0] = 99
print(a)  # tensor([1, 2, 3]) — unchanged
```

### `detach()`
```python
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = a.detach()  # shares data but no gradient tracking
print(b.requires_grad)  # False
```

**Note:** `detach()` becomes important with autograd. For now, know it exists.

### ⭐ Key Points
- ⭐ `b = a` does NOT copy — both point to same data.
- ⭐ `clone()` creates an independent copy.
- ⭐ `detach()` removes gradient tracking.

---

## 16. TENSOR ↔ NUMPY

### NumPy → Tensor

```python
import numpy as np
import torch

np_array = np.array([1, 2, 3])

# Method 1: shares memory
t1 = torch.from_numpy(np_array)

# Method 2: copies data
t2 = torch.tensor(np_array)
```

### Tensor → NumPy

```python
tensor = torch.tensor([4, 5, 6])
np_array = tensor.numpy()
```

### Shared Memory Behavior
- `torch.from_numpy()` and `.numpy()` **share memory**.
- Changing one changes the other.
- Use `.clone()` to avoid this.

### Why dtype Matters
- NumPy default float is `float64`.
- PyTorch default float is `float32`.
- Convert explicitly if needed.

### ⭐ Key Points
- ⭐ `torch.from_numpy()` and `.numpy()` convert.
- ⭐ They share memory (be careful!).
- ⭐ Use `.clone()` for independent copies.

---

## 17. CPU AND GPU TENSORS

### CPU vs GPU

| CPU | GPU |
|-----|-----|
| General purpose | Parallel processing |
| Few cores | Thousands of cores |
| Slower for math | Faster for math |
| Always available | Needs NVIDIA GPU |

### What Is a Device?
Device = where the tensor lives (`cpu` or `cuda`).

### Moving Tensors

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1, 2, 3])
x = x.to(device)
print(x.device)
```

### Line-by-Line Explanation
1. `torch.cuda.is_available()` → `True` if GPU exists.
2. `torch.device("cuda" if ... else "cpu")` → picks GPU or CPU.
3. `x.to(device)` → moves tensor to that device.

### Checking GPU

```python
print(torch.cuda.is_available())  # True or False
```

### ⭐ Key Points
- ⭐ GPU is faster for tensor math.
- ⭐ `.to(device)` moves tensors.
- ⭐ Always check `torch.cuda.is_available()`.

---

## 18. TENSOR DEVICE RULE

### The Rule
> Operations require tensors to be on **compatible devices**.

### What Happens When Devices Differ

```python
a = torch.tensor([1, 2, 3])  # CPU
b = torch.tensor([4, 5, 6]).to("cuda")  # GPU

# a + b  # ERROR: Expected all tensors on same device
```

**Error:** `RuntimeError: Expected all tensors to be on the same device`

### How to Fix

```python
a = a.to("cuda")
c = a + b  # Works
```

### ⭐ Key Points
- ⭐ All tensors in an operation must be on same device.
- ⭐ Use `.to(device)` to fix.
- ⭐ Common beginner error.

---

## 19. RANDOM TENSORS AND RANDOM SEEDS

### Random Tensor Generation
```python
x = torch.rand(2, 3)  # Different every run
```

### `torch.manual_seed()`
```python
torch.manual_seed(42)
x = torch.rand(2, 3)

torch.manual_seed(42)
y = torch.rand(2, 3)

print(torch.equal(x, y))  # True
```

### Why Reproducibility Matters
- Debugging
- Comparing experiments
- Sharing results

### ⭐ Key Points
- ⭐ Random tensors differ each run.
- ⭐ `manual_seed()` makes results reproducible.
- ⭐ Essential for research and debugging.

---

## 20. TENSOR COMPARISONS

### Comparison Operators

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([3, 2, 1])

print(a == b)  # tensor([False, True, False])
print(a != b)  # tensor([True, False, True])
print(a > b)   # tensor([False, False, True])
print(a < b)   # tensor([True, False, False])
print(a >= b)  # tensor([False, True, True])
print(a <= b)  # tensor([True, True, False])
```

### `torch.equal()`
```python
print(torch.equal(a, b))  # False — exact equality
```

### `torch.allclose()`
```python
x = torch.tensor([1.0, 2.0])
y = torch.tensor([1.0001, 2.0001])

print(torch.allclose(x, y))  # True — close enough
```

### When to Use Each

| Function | Use |
|----------|-----|
| `==` | Element-wise comparison |
| `torch.equal()` | Exact tensor equality |
| `torch.allclose()` | Approximate equality (floats) |

### ⭐ Key Points
- ⭐ Comparisons return boolean tensors.
- ⭐ `torch.equal()` for exact match.
- ⭐ `torch.allclose()` for float tolerance.

---

## 21. COMMON TENSOR ERRORS

### 1. Shape Mismatch
```python
a = torch.rand(2, 3)
b = torch.rand(2, 4)
# a + b  # ERROR
```
**Fix:** Ensure shapes match or broadcast.

### 2. Wrong dtype
```python
a = torch.tensor([1, 2, 3])  # int64
b = torch.tensor([1.0, 2.0])  # float32
# a + b  # Works but result is float
```
**Fix:** Convert explicitly with `.float()`.

### 3. Wrong Device
```python
a = torch.tensor([1, 2])
b = torch.tensor([3, 4]).to("cuda")
# a + b  # ERROR
```
**Fix:** `a = a.to(b.device)`.

### 4. Matrix Multiplication Dimensions
```python
a = torch.rand(2, 3)
b = torch.rand(4, 5)
# a @ b  # ERROR
```
**Fix:** Match inner dimensions.

### 5. Index Out of Range
```python
x = torch.tensor([1, 2, 3])
# x[5]  # ERROR
```
**Fix:** Use valid indices.

### 6. Incorrect `dim`
```python
x = torch.rand(2, 3)
# torch.sum(x, dim=5)  # ERROR
```
**Fix:** Use valid dim (0 or 1 for 2D).

### 7. CPU/GPU Mismatch
```python
model = model.to("cuda")
data = data  # still on CPU
# model(data)  # ERROR
```
**Fix:** Move data to same device.

### ⭐ Key Points
- ⭐ Most errors are shape, dtype, or device mismatches.
- ⭐ Always check `.shape`, `.dtype`, `.device`.
- ⭐ Fix with `reshape`, `.float()`, `.to(device)`.

---

## 22. PRACTICAL EXAMPLE

```python
import torch

# 1. Create a tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)

# 2. Inspect properties
print("Shape:", x.shape)      # torch.Size([2, 3])
print("dtype:", x.dtype)      # torch.float32
print("Device:", x.device)    # cpu

# 3. Index it
print("First row:", x[0])     # tensor([1., 2., 3.])
print("Element [1,2]:", x[1, 2])  # tensor(6.)

# 4. Arithmetic
y = x + 10
print("After +10:\n", y)

# 5. Reshape
z = x.reshape(3, 2)
print("Reshaped:\n", z)

# 6. Statistics
print("Sum:", x.sum())        # tensor(21.)
print("Mean:", x.mean())      # tensor(3.5)

# 7. Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
print("Device after move:", x.device)
```

### Step-by-Step Explanation
1. **Create**: A 2×3 float tensor.
2. **Inspect**: Check shape, dtype, device.
3. **Index**: Access rows and elements.
4. **Arithmetic**: Add 10 to all elements.
5. **Reshape**: Change to 3×2.
6. **Statistics**: Sum and mean.
7. **Device**: Move to GPU if available.

### ⭐ Key Points
- ⭐ This example covers creation, inspection, indexing, math, reshape, stats, device.
- ⭐ Practice this until comfortable.

---

## 23. MINI PROJECT: TENSOR PLAYGROUND

### Problem Statement
Create a Python script that demonstrates your understanding of PyTorch tensors.

### Requirements
1. Create a tensor from a Python list.
2. Create a tensor of zeros and one of ones.
3. Create a random tensor with `torch.rand()`.
4. Print shape, dtype, and device of each.
5. Perform addition, subtraction, multiplication, division on two tensors.
6. Slice a tensor to get a sub-tensor.
7. Reshape a tensor from `[2, 6]` to `[3, 4]`.
8. Perform matrix multiplication on compatible tensors.
9. Use `dim=0` and `dim=1` with `torch.sum()`.
10. Convert a NumPy array to a tensor and back.
11. Move a tensor to GPU if available, else CPU.

### Expected Behavior
- All operations run without error.
- Printed shapes match expectations.
- Device is correctly reported.

### Hints
- Use `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.rand()`.
- Use `.shape`, `.dtype`, `.device`.
- Use `+`, `-`, `*`, `/`, `@`.
- Use slicing `x[0:2]`.
- Use `.reshape()`.
- Use `torch.sum(x, dim=...)`.
- Use `torch.from_numpy()` and `.numpy()`.
- Use `.to(device)`.

### Do NOT Look at the Solution Yet — Try It Yourself!

---

## 24. IMPORTANT TENSOR TERMINOLOGY

| Term | Simple Meaning | Example |
|------|----------------|---------|
| **Tensor** | Multi-dimensional array | `torch.rand(2, 3)` |
| **Scalar** | 0D tensor | `torch.tensor(5)` |
| **Vector** | 1D tensor | `torch.tensor([1, 2, 3])` |
| **Matrix** | 2D tensor | `torch.tensor([[1, 2], [3, 4]])` |
| **Dimension** | Axis of a tensor | `dim=0`, `dim=1` |
| **Shape** | Size of each dimension | `[2, 3]` |
| **dtype** | Data type | `torch.float32` |
| **device** | CPU or GPU | `cpu`, `cuda:0` |
| **Element** | Single value | `x[0, 1]` |
| **Batch** | Group of samples | `[32, 3, 224, 224]` |
| **Broadcasting** | Auto-expanding shapes | `[2,2] + [2]` |
| **Reshape** | Change shape | `x.reshape(3, 2)` |
| **View** | Reshape (contiguous) | `x.view(3, 2)` |
| **Flatten** | To 1D | `x.flatten()` |
| **Indexing** | Access element | `x[0]` |
| **Slicing** | Access sub-tensor | `x[1:3]` |
| **Concatenation** | Join along dim | `torch.cat([a, b])` |
| **Stack** | Join along new dim | `torch.stack([a, b])` |

---

## 25. TENSOR CHEAT SHEET

### Creating
```python
torch.tensor(data)
torch.zeros(2, 3)
torch.ones(2, 3)
torch.arange(0, 10, 2)
torch.linspace(0, 1, 5)
torch.rand(2, 3)
torch.randn(2, 3)
torch.randint(0, 10, (2, 3))
```

### Properties
```python
x.shape
x.ndim
x.dtype
x.device
```

### Operations
```python
a + b
a - b
a * b
a / b
a @ b
torch.matmul(a, b)
```

### Manipulation
```python
x.reshape(3, 2)
x.view(3, 2)
x.flatten()
x.squeeze()
x.unsqueeze(0)
x.transpose(0, 1)
x.permute(2, 0, 1)
```

### Combination
```python
torch.cat([a, b], dim=0)
torch.stack([a, b], dim=0)
```

### Statistics
```python
x.sum()
x.mean()
x.max()
x.min()
x.argmax()
x.argmin()
```

### Device
```python
x.to(device)
torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
```

### NumPy
```python
torch.from_numpy(np_array)
tensor.numpy()
```

---

## 26. PRACTICE QUESTIONS

### 20 Conceptual Questions

1. What is a tensor?
2. What is the difference between a scalar, vector, and matrix?
3. Why does PyTorch use tensors instead of Python lists?
4. What does `.shape` tell you?
5. What does `.dtype` mean?
6. What does `.device` mean?
7. What is the difference between `torch.rand()` and `torch.randn()`?
8. What is the difference between `torch.arange()` and `torch.linspace()`?
9. What does `requires_grad=True` do?
10. What is broadcasting?
11. What is the difference between `reshape()` and `view()`?
12. What does `squeeze()` do?
13. What does `unsqueeze()` do?
14. What is the difference between `torch.cat()` and `torch.stack()`?
15. What is the difference between `transpose()` and `permute()`?
16. What does `dim=0` mean in `torch.sum()`?
17. Why do we need to move tensors to GPU?
18. What happens if tensors are on different devices?
19. What is the purpose of `torch.manual_seed()`?
20. What is the difference between `torch.equal()` and `torch.allclose()`?

### 15 Code-Based Questions

1. Create a 3×4 tensor of zeros.
2. Create a tensor from `[1, 2, 3]` with dtype `float32`.
3. Create a 5×5 identity matrix.
4. Create a tensor of 10 evenly spaced values from 0 to 1.
5. Create a random tensor of shape `[2, 3, 4]`.
6. Print the shape, dtype, and device of a tensor.
7. Add two tensors of shape `[2, 3]`.
8. Multiply two tensors element-wise.
9. Perform matrix multiplication on `[2, 3]` and `[3, 4]`.
10. Slice a tensor to get the first two rows.
11. Reshape a `[2, 6]` tensor to `[3, 4]`.
12. Compute the sum of a tensor along `dim=1`.
13. Concatenate two tensors along `dim=0`.
14. Convert a NumPy array to a tensor.
15. Move a tensor to GPU if available.

### 10 Debugging Questions

1. Why does `torch.rand(2, 3) + torch.rand(3, 2)` fail?
2. Why does `torch.tensor([1, 2]) + torch.tensor([1.0, 2.0])` work but change dtype?
3. Why does `x[10]` fail for a tensor of length 5?
4. Why does `torch.sum(x, dim=5)` fail for a 2D tensor?
5. Why does `a @ b` fail for `[2, 3]` and `[4, 5]`?
6. Why does CPU tensor + GPU tensor fail?
7. Why does `x.view(3, 2)` fail for a non-contiguous tensor?
8. Why does `torch.tensor([1, 2, 3]).float()` work but `.int()` on a float tensor truncates?
9. Why does `torch.cat([a, b], dim=1)` fail if shapes don't match?
10. Why does `torch.stack([a, b])` require same shapes?

### 15 MCQs

1. What is the default dtype for `torch.tensor([1.0, 2.0])`?
   a) float64  b) float32  c) int64  d) float16
   **Answer: b**

2. What does `.ndim` return?
   a) Shape  b) Number of dimensions  c) dtype  d) Device
   **Answer: b**

3. Which function creates a tensor with random normal values?
   a) `torch.rand()`  b) `torch.randn()`  c) `torch.randint()`  d) `torch.zeros()`
   **Answer: b**

4. What does `torch.eye(3)` create?
   a) Zeros  b) Ones  c) Identity  d) Random
   **Answer: c**

5. What does `x[0]` return for a 2D tensor?
   a) First column  b) First row  c) First element  d) Error
   **Answer: b**

6. What is the shape of `torch.rand(2, 3, 4)`?
   a) [3, 2, 4]  b) [2, 3, 4]  c) [4, 3, 2]  d) [2, 4, 3]
   **Answer: b**

7. What does `torch.sum(x, dim=0)` do?
   a) Sum columns  b) Sum rows  c) Sum all  d) Error
   **Answer: b**

8. Which is NOT a valid dtype?
   a) `torch.float32`  b) `torch.int64`  c) `torch.string`  d) `torch.bool`
   **Answer: c**

9. What does `.to("cuda")` do?
   a) Converts dtype  b) Moves to GPU  c) Reshapes  d) Clones
   **Answer: b**

10. What does `torch.cat([a, b], dim=0)` do?
    a) Stacks  b) Concatenates  c) Multiplies  d) Reshapes
    **Answer: b**

11. What does `squeeze()` remove?
    a) All dims  b) Size-1 dims  c) Last dim  d) First dim
    **Answer: b**

12. What does `unsqueeze(0)` do?
    a) Removes dim  b) Adds dim at 0  c) Adds dim at 1  d) Flattens
    **Answer: b**

13. What is `torch.matmul(a, b)` for?
    a) Element-wise  b) Matrix multiplication  c) Concatenation  d) Reshape
    **Answer: b**

14. What does `torch.manual_seed(42)` do?
    a) Sets device  b) Sets random seed  c) Sets dtype  d) Sets shape
    **Answer: b**

15. What does `torch.allclose(a, b)` check?
    a) Exact equality  b) Approximate equality  c) Shape  d) dtype
    **Answer: b**

---

## 27. QUICK REVISION

- **Tensor**: Multi-dimensional array (core data structure)
- **Creation**: `tensor()`, `zeros()`, `ones()`, `rand()`, `randn()`, `arange()`, `linspace()`
- **Shape**: `.shape` — dimensions of tensor
- **dtype**: `.dtype` — data type (float32, int64, etc.)
- **device**: `.device` — CPU or GPU
- **Indexing**: `x[0]`, `x[:, 1]`, `x[1:3]`
- **Slicing**: `x[1:4]`, `x[::2]`
- **Arithmetic**: `+`, `-`, `*`, `/` (element-wise)
- **Matrix multiplication**: `@` or `torch.matmul()`
- **dim**: Axis for operations (`dim=0`, `dim=1`)
- **Reshaping**: `reshape()`, `view()`, `flatten()`, `squeeze()`, `unsqueeze()`
- **Broadcasting**: Auto-expanding shapes
- **NumPy conversion**: `torch.from_numpy()`, `.numpy()`
- **CPU/GPU**: `.to(device)`, `torch.device()`, `torch.cuda.is_available()`

---

## 28. WHAT TO LEARN NEXT

After mastering tensors, follow this roadmap:

1. **Autograd and Gradients** — how PyTorch computes gradients
2. **Computational Graphs** — how operations are tracked
3. **Datasets and DataLoaders** — loading and batching data
4. **nn.Module** — building neural networks
5. **Neural Network Layers** — Linear, Conv2d, etc.
6. **Loss Functions** — MSE, CrossEntropy
7. **Optimizers** — SGD, Adam
8. **Training Loops** — putting it all together


---
