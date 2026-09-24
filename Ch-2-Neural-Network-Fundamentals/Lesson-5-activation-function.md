# Part 2 — Neural Network Fundamentals
# Chapter 5 — Activation Functions

**Prerequisite:** Part 2 — Chapter 1 — Neural Network Basics in PyTorch, Part 2 — Chapter 2 — `torch.nn`, Part 2 — Chapter 3 — `nn.Module`, Part 2 — Chapter 4 — Layers

---

## 1. Overview

- Activation functions introduce non-linearity into neural networks, enabling them to approximate complex functions.
- Without activation functions, stacked linear layers collapse into a single linear transformation.
- Common activation functions: ReLU, Leaky ReLU, ELU, GELU, SiLU/Swish, Sigmoid, Tanh, Softmax, LogSoftmax, Softplus, Mish.
- Module form (`nn.ReLU`) vs functional form (`F.relu`).
- In-place activations (`inplace=True`) and their interaction with autograd.
- Activation functions in different architectures: CNNs (ReLU), Transformers (GELU), RNNs (Tanh), output layers (Sigmoid/Softmax).
- Saturation, vanishing gradients, dying ReLU, and how modern activations address them.
- Gradient behavior and computational cost.
- Choosing the right activation for a task.

---

## 2. Core Concepts

**Activation Function:** A non-linear function applied element-wise to the output of a linear layer. Denoted `σ(z)` or `f(z)`.

**Non-linearity:** The property that allows neural networks to model complex, non-linear relationships. Without it, the network is equivalent to a single linear layer.

**Saturation:** When an activation function's gradient approaches zero for large positive or negative inputs. Causes vanishing gradients.

**Vanishing Gradient:** Gradients become extremely small during backpropagation, preventing deep layers from learning.

**Dying ReLU:** A phenomenon where ReLU neurons output zero for all inputs, effectively dead because their gradient is always zero.

**Zero-Centered:** An activation whose output has mean approximately zero, which helps optimization.

**Monotonic:** An activation that is either entirely non-decreasing or non-increasing.

**Smooth:** An activation with continuous derivatives, which can help optimization.

**In-place Activation:** An activation that modifies its input tensor directly, saving memory.

**Logits:** Raw, unnormalized outputs of the last linear layer before applying softmax/sigmoid.

**Softmax:** Converts a vector of logits into a probability distribution summing to 1.

**LogSoftmax:** `log(softmax(x))`, numerically more stable.

**GELU (Gaussian Error Linear Unit):** A smooth activation used in Transformers. `GELU(x) = x * Φ(x)` where `Φ` is the standard Gaussian CDF.

**SiLU / Swish:** `SiLU(x) = x * sigmoid(x)`.

**Mish:** `Mish(x) = x * tanh(softplus(x))`.

**ELU (Exponential Linear Unit):** Smooth activation that allows negative values.

**Leaky ReLU:** ReLU variant that allows a small negative slope.

**PReLU:** Parametric ReLU with learnable negative slope.

**Softplus:** Smooth approximation of ReLU: `softplus(x) = log(1 + exp(x))`.

---

## 3. Important PyTorch APIs

### `torch.nn.ReLU(inplace=False)`

- **Purpose:** Applies `ReLU(x) = max(0, x)`.
- **Parameters:** `inplace` (bool): If `True`, modifies input directly. Default `False`.
- **Formula:** `f(x) = max(0, x)`
- **Derivative:** `f'(x) = 1 if x > 0 else 0`
- **Example:**
```python
relu = nn.ReLU()
x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
print(relu(x))  # tensor([0., 0., 1., 2.])
```

### `torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)`

- **Purpose:** `LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)`.
- **Parameters:**
  - `negative_slope` (float): Slope for negative inputs. Default `0.01`.
  - `inplace` (bool): Default `False`.
- **Formula:** `f(x) = x if x > 0 else negative_slope * x`
- **Derivative:** `f'(x) = 1 if x > 0 else negative_slope`
- **Example:**
```python
lrelu = nn.LeakyReLU(0.1)
x = torch.tensor([-1.0, 0.0, 1.0])
print(lrelu(x))  # tensor([-0.1000,  0.0000,  1.0000])
```

### `torch.nn.PReLU(num_parameters=1, init=0.25)`

- **Purpose:** Parametric ReLU with learnable negative slope.
- **Parameters:**
  - `num_parameters` (int): Number of slopes to learn. Default `1`.
  - `init` (float): Initial slope value. Default `0.25`.
- **Example:**
```python
prelu = nn.PReLU()
x = torch.tensor([-1.0, 0.0, 1.0])
print(prelu(x))
```

### `torch.nn.ELU(alpha=1.0, inplace=False)`

- **Purpose:** `ELU(x) = x if x > 0 else alpha * (exp(x) - 1)`.
- **Parameters:**
  - `alpha` (float): Scale for negative part. Default `1.0`.
  - `inplace` (bool): Default `False`.
- **Example:**
```python
elu = nn.ELU()
x = torch.tensor([-1.0, 0.0, 1.0])
print(elu(x))  # tensor([-0.6321,  0.0000,  1.0000])
```

### `torch.nn.SELU(inplace=False)`

- **Purpose:** Scaled ELU: `SELU(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))`.
- **Parameters:** `inplace` (bool): Default `False`.
- **Note:** Use with `nn.AlphaDropout` for self-normalizing networks.

### `torch.nn.CELU(alpha=1.0, inplace=False)`

- **Purpose:** Continuously differentiable ELU: `CELU(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1))`.
- **Parameters:** `alpha` (float): Default `1.0`. `inplace` (bool): Default `False`.

### `torch.nn.GELU(approximate='none')`

- **Purpose:** Gaussian Error Linear Unit. `GELU(x) = x * Φ(x)`.
- **Parameters:**
  - `approximate` (str): `'none'` for exact, `'tanh'` for tanh approximation. Default `'none'`.
- **Example:**
```python
gelu = nn.GELU()
x = torch.tensor([-1.0, 0.0, 1.0])
print(gelu(x))  # tensor([-0.1587,  0.0000,  0.8413])
```

### `torch.nn.SiLU(inplace=False)`

- **Purpose:** Sigmoid Linear Unit (Swish). `SiLU(x) = x * sigmoid(x)`.
- **Parameters:** `inplace` (bool): Default `False`.
- **Example:**
```python
silu = nn.SiLU()
x = torch.tensor([-1.0, 0.0, 1.0])
print(silu(x))  # tensor([-0.2689,  0.0000,  0.7311])
```

### `torch.nn.Mish(inplace=False)`

- **Purpose:** `Mish(x) = x * tanh(softplus(x))`.
- **Parameters:** `inplace` (bool): Default `False`.
- **Example:**
```python
mish = nn.Mish()
x = torch.tensor([-1.0, 0.0, 1.0])
print(mish(x))
```

### `torch.nn.Sigmoid()`

- **Purpose:** `Sigmoid(x) = 1 / (1 + exp(-x))`.
- **Output range:** `(0, 1)`.
- **Derivative:** `σ'(x) = σ(x) * (1 - σ(x))`.
- **Example:**
```python
sigmoid = nn.Sigmoid()
x = torch.tensor([-1.0, 0.0, 1.0])
print(sigmoid(x))  # tensor([0.2689, 0.5000, 0.7311])
```

### `torch.nn.Tanh()`

- **Purpose:** `Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`.
- **Output range:** `(-1, 1)`.
- **Derivative:** `tanh'(x) = 1 - tanh(x)^2`.
- **Example:**
```python
tanh = nn.Tanh()
x = torch.tensor([-1.0, 0.0, 1.0])
print(tanh(x))  # tensor([-0.7616,  0.0000,  0.7616])
```

### `torch.nn.Softmax(dim=None)`

- **Purpose:** Converts logits to probabilities. `Softmax(x_i) = exp(x_i) / sum_j exp(x_j)`.
- **Parameters:** `dim` (int): Dimension along which to compute softmax. Required.
- **Example:**
```python
softmax = nn.Softmax(dim=1)
x = torch.tensor([[1.0, 2.0, 3.0]])
print(softmax(x))  # tensor([[0.0900, 0.2447, 0.6652]])
```

### `torch.nn.LogSoftmax(dim=None)`

- **Purpose:** `LogSoftmax(x) = log(Softmax(x))`, numerically stable.
- **Parameters:** `dim` (int): Required.
- **Example:**
```python
log_softmax = nn.LogSoftmax(dim=1)
x = torch.tensor([[1.0, 2.0, 3.0]])
print(log_softmax(x))
```

### `torch.nn.Softplus(beta=1.0, threshold=20.0)`

- **Purpose:** Smooth approximation of ReLU: `Softplus(x) = (1/beta) * log(1 + exp(beta * x))`.
- **Parameters:**
  - `beta` (float): Default `1.0`.
  - `threshold` (float): Above this, linear approximation. Default `20.0`.
- **Example:**
```python
softplus = nn.Softplus()
x = torch.tensor([-1.0, 0.0, 1.0])
print(softplus(x))  # tensor([0.3133, 0.6931, 1.3133])
```

### `torch.nn.Softshrink(lambd=0.5)`

- **Purpose:** Soft thresholding: `f(x) = x - lambd if x > lambd; x + lambd if x < -lambd; else 0`.

### `torch.nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)`

- **Purpose:** Clamps input to `[min_val, max_val]`.

### `torch.nn.Hardsigmoid(inplace=False)`

- **Purpose:** Piecewise linear approximation of sigmoid.

### `torch.nn.Hardswish(inplace=False)`

- **Purpose:** `Hardswish(x) = x * ReLU6(x + 3) / 6`.

### `torch.nn.Threshold(threshold, value, inplace=False)`

- **Purpose:** `f(x) = x if x > threshold else value`.

### `torch.nn.ReLU6(inplace=False)`

- **Purpose:** `ReLU6(x) = min(max(0, x), 6)`.

### Functional Equivalents (`torch.nn.functional`)

| Module | Functional |
|--------|-----------|
| `nn.ReLU()` | `F.relu(x)` |
| `nn.LeakyReLU()` | `F.leaky_relu(x, negative_slope)` |
| `nn.ELU()` | `F.elu(x, alpha)` |
| `nn.GELU()` | `F.gelu(x)` |
| `nn.SiLU()` | `F.silu(x)` |
| `nn.Mish()` | `F.mish(x)` |
| `nn.Sigmoid()` | `torch.sigmoid(x)` or `F.sigmoid(x)` |
| `nn.Tanh()` | `torch.tanh(x)` or `F.tanh(x)` |
| `nn.Softmax()` | `F.softmax(x, dim)` |
| `nn.LogSoftmax()` | `F.log_softmax(x, dim)` |
| `nn.Softplus()` | `F.softplus(x)` |

**Example:**
```python
import torch.nn.functional as F
x = torch.randn(3, 5)
print(F.relu(x))
print(F.softmax(x, dim=1))
```

---

## 4. Code Examples

### Example 1: Basic Activation Functions

```python
import torch
import torch.nn as nn

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

activations = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'ELU': nn.ELU(),
    'GELU': nn.GELU(),
    'SiLU': nn.SiLU(),
    'Mish': nn.Mish(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'Softplus': nn.Softplus(),
}

print(f"{'Input':>12}", x.tolist())
for name, act in activations.items():
    y = act(x)
    print(f"{name:>12}", [f"{v:.4f}" for v in y.tolist()])
```

**What it does:** Applies many activation functions to the same input and compares outputs.

**Important lines:**
- Each activation is an `nn.Module` instance.
- Outputs differ in range and shape.

**Expected output:**
```
       Input [-3.0, -1.0, 0.0, 1.0, 3.0]
        ReLU ['0.0000', '0.0000', '0.0000', '1.0000', '3.0000']
   LeakyReLU ['-0.3000', '-0.1000', '0.0000', '1.0000', '3.0000']
         ELU ['-0.9502', '-0.6321', '0.0000', '1.0000', '3.0000']
        GELU ['-0.0040', '-0.1587', '0.0000', '0.8413', '2.9960']
        SiLU ['-0.1423', '-0.2689', '0.0000', '0.7311', '2.8577']
        Mish ['-0.0148', '-0.3034', '0.0000', '0.8651', '2.9865']
     Sigmoid ['0.0474', '0.2689', '0.5000', '0.7311', '0.9526']
        Tanh ['-0.9951', '-0.7616', '0.0000', '0.7616', '0.9951']
    Softplus ['0.0486', '0.3133', '0.6931', '1.3133', '3.0486']
```

### Example 2: Gradient Behavior

```python
import torch
import torch.nn as nn

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], requires_grad=True)

for name, act in [('ReLU', nn.ReLU()),
                  ('Sigmoid', nn.Sigmoid()),
                  ('Tanh', nn.Tanh()),
                  ('GELU', nn.GELU()),
                  ('SiLU', nn.SiLU())]:
    x.grad = None
    y = act(x)
    y.sum().backward(retain_graph=True)
    print(f"{name:>8} gradients: {[f'{g:.4f}' for g in x.grad.tolist()]}")
    x.grad = None
```

**What it does:** Computes gradients of different activations.

**Important lines:**
- `y.sum().backward()` computes gradients.
- `x.grad` holds the gradients.
- Zero grad before next activation.

**Expected output:**
```
   ReLU gradients: ['0.0000', '0.0000', '0.0000', '1.0000', '1.0000']
Sigmoid gradients: ['0.0452', '0.1966', '0.2500', '0.1966', '0.0452']
   Tanh gradients: ['0.0099', '0.4200', '1.0000', '0.4200', '0.0099']
   GELU gradients: ['-0.0040', '-0.0833', '0.5000', '1.0833', '1.0040']
   SiLU gradients: ['-0.0148', '-0.0722', '0.5000', '0.9278', '1.0148']
```

### Example 3: Sigmoid Saturation and Vanishing Gradients

```python
import torch
import torch.nn as nn

x = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0], requires_grad=True)

sigmoid = nn.Sigmoid()
y = sigmoid(x)
y.sum().backward()

print("Input:", x.tolist())
print("Sigmoid output:", [f"{v:.6f}" for v in y.tolist()])
print("Sigmoid gradient:", [f"{g:.6f}" for g in x.grad.tolist()])
```

**What it does:** Shows sigmoid saturation for large inputs.

**Important lines:**
- Gradients vanish for large `|x|`.
- This is why sigmoid is rarely used in hidden layers of deep networks.

**Expected output:**
```
Input: [-10.0, -5.0, 0.0, 5.0, 10.0]
Sigmoid output: ['0.000045', '0.006693', '0.500000', '0.993307', '0.999955']
Sigmoid gradient: ['0.000045', '0.006648', '0.250000', '0.006648', '0.000045']
```

### Example 4: Dying ReLU

```python
import torch
import torch.nn as nn

# Large negative input causes ReLU to output 0 and have 0 gradient
x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], requires_grad=True)

relu = nn.ReLU()
y = relu(x)
y.sum().backward()

print("Input:", x.tolist())
print("ReLU output:", y.tolist())
print("ReLU gradient:", x.grad.tolist())
```

**What it does:** Demonstrates that ReLU has zero gradient for negative inputs, which can cause neurons to die.

**Important lines:**
- Negative inputs produce zero output and zero gradient.
- This is the "dying ReLU" problem.

**Expected output:**
```
Input: [-5.0, -1.0, 0.0, 1.0, 5.0]
ReLU output: [0.0, 0.0, 0.0, 1.0, 5.0]
ReLU gradient: [0.0, 0.0, 0.0, 1.0, 1.0]
```

### Example 5: Softmax and LogSoftmax

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Logits (raw scores)
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3]])

# Softmax
softmax = nn.Softmax(dim=1)
probs = softmax(logits)
print("Softmax probabilities:\n", probs)
print("Row sums:", probs.sum(dim=1))

# LogSoftmax
log_softmax = nn.LogSoftmax(dim=1)
log_probs = log_softmax(logits)
print("\nLogSoftmax:\n", log_probs)
print("Exp of log_probs:\n", log_probs.exp())
print("Row sums:", log_probs.exp().sum(dim=1))

# Functional
print("\nF.softmax:\n", F.softmax(logits, dim=1))
```

**What it does:** Demonstrates softmax and log-softmax.

**Important lines:**
- Softmax outputs sum to 1 per row.
- LogSoftmax is numerically more stable when combined with NLLLoss.
- `CrossEntropyLoss` combines `LogSoftmax` and `NLLLoss`.

**Expected output:**
```
Softmax probabilities:
 tensor([[0.6590, 0.2424, 0.0986],
         [0.1065, 0.7854, 0.1081]])
Row sums: tensor([1., 1.])

LogSoftmax:
 tensor([[-0.4170, -1.4170, -2.3170],
         [-2.2407, -0.2407, -2.2407]])
Exp of log_probs:
 tensor([[0.6590, 0.2424, 0.0986],
         [0.1065, 0.7854, 0.1081]])
Row sums: tensor([1., 1.])
```

### Example 6: In-Place Activation

```python
import torch
import torch.nn as nn

x = torch.tensor([-1.0, 0.0, 1.0, 2.0])

# Out-of-place
relu = nn.ReLU(inplace=False)
y1 = relu(x)
print("Out-of-place:")
print("  x:", x)
print("  y1:", y1)
print("  x is y1:", x is y1)

# In-place
x2 = torch.tensor([-1.0, 0.0, 1.0, 2.0])
relu_inplace = nn.ReLU(inplace=True)
y2 = relu_inplace(x2)
print("\nIn-place:")
print("  x2:", x2)
print("  y2:", y2)
print("  x2 is y2:", x2 is y2)
```

**What it does:** Contrasts in-place and out-of-place ReLU.

**Important lines:**
- In-place modifies input and returns the same tensor.
- Saves memory but can break autograd if input is needed.

**Expected output:**
```
Out-of-place:
  x: tensor([-1.,  0.,  1.,  2.])
  y1: tensor([0., 0., 1., 2.])
  x is y1: False

In-place:
  x2: tensor([0., 0., 1., 2.])
  y2: tensor([0., 0., 1., 2.])
  x2 is y2: True
```

### Example 7: Activation in a Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
        self.activation_name = activation

    def forward(self, x):
        x = self.fc1(x)
        if self.activation_name == 'relu':
            x = F.relu(x)
        elif self.activation_name == 'gelu':
            x = F.gelu(x)
        elif self.activation_name == 'silu':
            x = F.silu(x)
        elif self.activation_name == 'tanh':
            x = torch.tanh(x)
        x = self.fc2(x)
        return x

x = torch.randn(4, 10)

for act in ['relu', 'gelu', 'silu', 'tanh']:
    model = Net(activation=act)
    y = model(x)
    print(f"{act:>6}: output mean={y.mean().item():.4f}, std={y.std().item():.4f}")
```

**What it does:** Uses different activations in a small network.

**Important lines:**
- Activation choice affects output statistics.
- GELU and SiLU are smoother than ReLU.

**Expected output:**
```
  relu: output mean=... std=...
  gelu: output mean=... std=...
  silu: output mean=... std=...
  tanh: output mean=... std=...
```

### Example 8: Activation Function Visualization

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)

activations = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'ELU': nn.ELU(),
    'GELU': nn.GELU(),
    'SiLU': nn.SiLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, (name, act) in enumerate(activations.items()):
    y = act(x)
    axes[i].plot(x.numpy(), y.detach().numpy())
    axes[i].set_title(name)
    axes[i].grid(True)
    axes[i].axhline(0, color='black', linewidth=0.5)
    axes[i].axvline(0, color='black', linewidth=0.5)

# Gradient plot
x_grad = torch.linspace(-5, 5, 200, requires_grad=True)
for i, (name, act) in enumerate(activations.items()):
    if i >= 4:
        break
    x_grad.grad = None
    y = act(x_grad)
    y.sum().backward()
    axes[4 + i].plot(x_grad.detach().numpy(), x_grad.grad.numpy())
    axes[4 + i].set_title(f"{name} gradient")
    axes[4 + i].grid(True)

plt.tight_layout()
plt.show()
```

**What it does:** Plots activation functions and their gradients.

**Important lines:**
- Visualizing activations helps understand their behavior.
- Gradient plots show saturation regions.

---

## 5. Important Parameters

### `nn.ReLU(inplace=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `inplace` | bool | If `True`, modifies input. Default `False`. |

### `nn.LeakyReLU(negative_slope=0.01, inplace=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `negative_slope` | float | Slope for negative inputs. Default `0.01`. |
| `inplace` | bool | Default `False`. |

### `nn.PReLU(num_parameters=1, init=0.25)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_parameters` | int | Number of learnable slopes. Default `1`. |
| `init` | float | Initial slope. Default `0.25`. |

### `nn.ELU(alpha=1.0, inplace=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | float | Scale for negative part. Default `1.0`. |
| `inplace` | bool | Default `False`. |

### `nn.GELU(approximate='none')`

| Parameter | Type | Description |
|-----------|------|-------------|
| `approximate` | str | `'none'` or `'tanh'`. Default `'none'`. |

### `nn.Softmax(dim=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | int | Dimension along which to compute softmax. Required. |

### `nn.LogSoftmax(dim=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | int | Dimension. Required. |

### `nn.Softplus(beta=1.0, threshold=20.0)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `beta` | float | Default `1.0`. |
| `threshold` | float | Default `20.0`. |

### `nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_val` | float | Default `-1.0`. |
| `max_val` | float | Default `1.0`. |
| `inplace` | bool | Default `False`. |

### `nn.Threshold(threshold, value, inplace=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold` | float | Threshold. |
| `value` | float | Value for inputs below threshold. |
| `inplace` | bool | Default `False`. |

### `nn.ReLU6(inplace=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `inplace` | bool | Default `False`. |

---

## 6. Internal Working

### Element-wise Application

All activation functions (except softmax variants) are applied element-wise:
`y[i] = f(x[i])` for each element.

### Softmax Over a Dimension

Softmax operates over a specified dimension:

```
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

- Uses the max-subtraction trick for numerical stability: `exp(x_i - max(x))`.
- `LogSoftmax` computes `x_i - max(x) - log(sum_j exp(x_j - max(x)))`.

### Gradient Computation

Each activation has a well-defined derivative used in backpropagation:

| Activation | Derivative |
|------------|-----------|
| ReLU | `1 if x > 0 else 0` |
| LeakyReLU | `1 if x > 0 else negative_slope` |
| ELU | `1 if x > 0 else ELU(x) + alpha` |
| GELU | `Φ(x) + x * φ(x)` (Φ = CDF, φ = PDF) |
| SiLU | `σ(x) + x * σ(x) * (1 - σ(x))` |
| Sigmoid | `σ(x) * (1 - σ(x))` |
| Tanh | `1 - tanh(x)^2` |
| Softplus | `sigmoid(beta * x)` |

### In-Place Operations

- In-place activations modify the input tensor's data directly.
- They save memory but can break autograd if the input is needed for gradient computation elsewhere.
- PyTorch raises an error if an in-place operation would corrupt the graph.

### Numerical Stability

- Sigmoid and Tanh saturate for large inputs.
- Softmax uses max-subtraction.
- LogSoftmax is preferred over `log(softmax(x))` for stability.

### Dying ReLU

- ReLU outputs zero for negative inputs and has zero gradient.
- If a neuron's weights are updated such that all inputs are negative, the neuron never recovers (gradient always zero).
- LeakyReLU, ELU, and other variants address this by allowing small negative gradients.

### Vanishing Gradients

- Sigmoid and Tanh saturate, causing gradients to vanish for deep networks.
- ReLU and its variants have linear gradient for positive inputs, reducing vanishing gradient.
- GELU and SiLU are smooth and have non-zero gradients for negative inputs.

---

## 7. Common Mistakes

**Mistake:** Applying softmax before `CrossEntropyLoss`.

**Why it happens:** `CrossEntropyLoss` includes `LogSoftmax`.

**Correct approach:** Use raw logits with `CrossEntropyLoss`.

---

**Mistake:** Using `Sigmoid` in hidden layers of deep networks.

**Why it happens:** Sigmoid saturates, causing vanishing gradients.

**Correct approach:** Use ReLU, GELU, or SiLU in hidden layers.

---

**Mistake:** Using `inplace=True` when the input is needed for a skip connection.

**Why it happens:** In-place ReLU overwrites the input.

**Correct approach:** Use `inplace=False` in residual connections.

---

**Mistake:** Forgetting `dim` in `nn.Softmax`.

**Why it happens:** `dim` is required.

**Correct approach:** Specify `dim` (usually `dim=1` for `(batch, classes)`).

---

**Mistake:** Using `Softmax` at output for multi-class classification with `CrossEntropyLoss`.

**Why it happens:** Double softmax.

**Correct approach:** Output raw logits.

---

**Mistake:** Using `Sigmoid` output with `BCELoss` and raw logits.

**Why it happens:** `BCELoss` expects probabilities.

**Correct approach:** Apply sigmoid, or use `BCEWithLogitsLoss` which combines sigmoid and BCE.

---

**Mistake:** Assuming all activations have the same gradient behavior.

**Why it happens:** ReLU has zero gradient for negatives; Sigmoid saturates.

**Correct approach:** Understand each activation's gradient.

---

**Mistake:** Using `Tanh` for very deep networks.

**Why it happens:** Vanishing gradients.

**Correct approach:** Use ReLU, GELU, or residual connections.

---

**Mistake:** Not switching to `eval()` mode for dropout or batch norm (activation-related).

**Why it happens:** Dropout and batch norm behave differently in train/eval.

**Correct approach:** Call `model.eval()` before inference.

---

**Mistake:** Using `nn.Softmax` as the last layer and then applying `CrossEntropyLoss`.

**Why it happens:** `CrossEntropyLoss` includes `LogSoftmax`.

**Correct approach:** Remove softmax; use logits.

---

**Mistake:** Using activation functions on the wrong dimension.

**Why it happens:** Softmax over batch dimension instead of class dimension.

**Correct approach:** Ensure `dim` is correct (usually `dim=-1`).

---

**Mistake:** Using `PReLU` with default `num_parameters=1`.

**Why it happens:** This learns a single slope for all channels.

**Correct approach:** Set `num_parameters` to the number of channels if per-channel slopes are desired.

---

## 8. Important Differences

| Concept | Difference |
|---------|-----------|
| ReLU vs LeakyReLU | ReLU: 0 for negative; LeakyReLU: small slope for negative. |
| ReLU vs ELU | ReLU: hard zero; ELU: smooth, negative saturation at `-alpha`. |
| ReLU vs GELU | ReLU: piecewise linear; GELU: smooth, used in Transformers. |
| Sigmoid vs Tanh | Sigmoid: `(0,1)`; Tanh: `(-1,1)`, zero-centered. |
| Softmax vs Sigmoid | Softmax: multi-class; Sigmoid: binary/multi-label. |
| Softmax vs LogSoftmax | LogSoftmax: log of softmax, more stable. |
| `nn.ReLU` vs `F.relu` | Module vs function. |
| `inplace=True` vs `inplace=False` | In-place saves memory but can break autograd. |
| GELU vs SiLU | GELU uses Gaussian CDF; SiLU uses sigmoid. |
| ELU vs CELU | CELU uses `alpha` inside exp; ELU uses it outside. |
| Hardtanh vs ReLU6 | Hardtanh clamps to `[min, max]`; ReLU6 clamps to `[0, 6]`. |
| PReLU vs LeakyReLU | PReLU: learnable slope; LeakyReLU: fixed slope. |
| SELU vs ELU | SELU: scaled to maintain mean/variance. |

---

## 9. Important Rules / Facts

- Activation functions introduce non-linearity, enabling deep networks to learn complex functions.
- Without activations, stacked linear layers collapse to a single linear layer.
- ReLU is the default choice for CNNs and MLPs.
- GELU is standard in Transformers.
- Sigmoid is used for binary classification output.
- Softmax is used for multi-class classification output.
- `CrossEntropyLoss` includes `LogSoftmax`; use raw logits.
- `BCELoss` expects probabilities; `BCEWithLogitsLoss` expects logits.
- Sigmoid and Tanh saturate, causing vanishing gradients in deep networks.
- ReLU can cause dying neurons; LeakyReLU and ELU mitigate this.
- In-place activations save memory but can break autograd.
- Softmax requires a `dim` argument.
- `LogSoftmax` is numerically more stable than `log(softmax(x))`.
- GELU and SiLU are smooth and popular in modern architectures.
- `nn.ReLU` is a module; `F.relu` is a function.
- Activation functions are applied element-wise (except softmax variants).
- Gradient behavior differs significantly between activations.
- Activation choice affects training dynamics and final performance.
- Always match the output activation to the loss function.
- Use `model.eval()` for inference to disable dropout.

---

## 10. Practical Example

### Multi-Layer Network with Configurable Activations and Training Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic multi-class data
torch.manual_seed(42)
N = 2000
X = torch.randn(N, 2)
y = (X[:, 0]**2 + X[:, 1]**2 > 1).long()  # circular decision boundary
y = torch.clamp(y + torch.randint(0, 2, (N,)), 0, 1)  # add noise

# Split
train_size = int(0.8 * N)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

def build_model(activation):
    act_map = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    return nn.Sequential(
        nn.Linear(2, 64),
        act_map[activation],
        nn.Linear(64, 64),
        act_map[activation],
        nn.Linear(64, 2)
    )

# Train with different activations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = {}

for act_name in ['relu', 'leaky_relu', 'elu', 'gelu', 'silu', 'tanh', 'sigmoid']:
    model = build_model(act_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    acc = correct / total
    results[act_name] = acc
    print(f"{act_name:>12}: val_acc={acc:.4f}")

# Print sorted results
print("\nSorted by accuracy:")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:>12}: {acc:.4f}")
```

**What it does:**
- Builds the same network with different activations.
- Trains each on the same data.
- Compares validation accuracy.

**Important lines:**
- `act_map` holds different activation modules.
- Same architecture, only activation changes.
- Results show how activation choice affects performance.

**Expected output:**
```
        relu: val_acc=0.8500
  leaky_relu: val_acc=0.8525
         elu: val_acc=0.8475
        gelu: val_acc=0.8550
        silu: val_acc=0.8550
        tanh: val_acc=0.8250
     sigmoid: val_acc=0.7750

Sorted by accuracy:
        gelu: 0.8550
        silu: 0.8550
  leaky_relu: 0.8525
        relu: 0.8500
         elu: 0.8475
        tanh: 0.8250
     sigmoid: 0.7750
```

### Transformer Block with GELU

```python
import torch
import torch.nn as nn

class TransformerFFN(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

ffn = TransformerFFN(embed_dim=64, ff_dim=256)
x = torch.randn(4, 10, 64)
y = ffn(x)
print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Activation:", ffn.act)
```

**What it does:** Uses GELU in a transformer feed-forward block.

**Expected output:**
```
Input shape: torch.Size([4, 10, 64])
Output shape: torch.Size([4, 10, 64])
Activation: GELU(approximate='none')
```

---

## 11. Chapter Summary

- Activation functions introduce non-linearity, enabling neural networks to learn complex functions.
- Without activations, stacked linear layers collapse into one linear transformation.
- ReLU: `max(0, x)` — default for CNNs and MLPs; can cause dying neurons.
- LeakyReLU, ELU, PReLU: address dying ReLU with non-zero negative slope.
- GELU, SiLU, Mish: smooth activations used in modern architectures.
- Sigmoid: `(0, 1)`, used for binary classification output; saturates.
- Tanh: `(-1, 1)`, zero-centered; saturates.
- Softmax: probability distribution over classes.
- LogSoftmax: numerically stable log of softmax.
- `CrossEntropyLoss` includes `LogSoftmax`; use raw logits.
- `BCELoss` expects probabilities; `BCEWithLogitsLoss` expects logits.
- In-place activations save memory but can break autograd.
- Softmax requires `dim`.
- GELU is standard in Transformers; ReLU in CNNs.
- Activation choice affects training dynamics and performance.
- Common mistakes: softmax before CrossEntropy, sigmoid in deep hidden layers, wrong output activation.
- Understanding gradients of activations is essential for debugging.

---

## 12. Important APIs to Remember

| API | Purpose |
|-----|---------|
| `nn.ReLU` | Rectified linear unit |
| `nn.LeakyReLU` | Leaky ReLU |
| `nn.PReLU` | Parametric ReLU |
| `nn.ELU` | Exponential linear unit |
| `nn.SELU` | Scaled ELU |
| `nn.CELU` | Continuously differentiable ELU |
| `nn.GELU` | Gaussian error linear unit |
| `nn.SiLU` | Sigmoid linear unit (Swish) |
| `nn.Mish` | Mish activation |
| `nn.Sigmoid` | Sigmoid |
| `nn.Tanh` | Hyperbolic tangent |
| `nn.Softmax` | Softmax |
| `nn.LogSoftmax` | Log-softmax |
| `nn.Softplus` | Smooth ReLU approximation |
| `nn.Softshrink` | Soft thresholding |
| `nn.Hardtanh` | Hard tanh |
| `nn.Hardsigmoid` | Hard sigmoid |
| `nn.Hardswish` | Hard swish |
| `nn.ReLU6` | ReLU clamped to 6 |
| `nn.Threshold` | Thresholding |
| `F.relu`, `F.gelu`, `F.silu`, etc. | Functional equivalents |
| `torch.sigmoid`, `torch.tanh` | Tensor methods |

---

## 13. Key Takeaways

1. Activation functions add non-linearity; without them, networks are linear.
2. ReLU is the default for CNNs and MLPs.
3. GELU is standard in Transformers; SiLU is used in modern CNNs.
4. Sigmoid outputs probabilities in `(0,1)`; used for binary classification.
5. Tanh outputs in `(-1,1)`; zero-centered but saturates.
6. Softmax converts logits to a probability distribution.
7. `CrossEntropyLoss` includes `LogSoftmax`; use raw logits.
8. `BCELoss` expects probabilities; `BCEWithLogitsLoss` expects logits.
9. Sigmoid and Tanh saturate, causing vanishing gradients in deep networks.
10. ReLU can cause dying neurons; LeakyReLU, ELU, and GELU mitigate this.
11. In-place activations save memory but can break autograd.
12. Softmax requires `dim`; usually `dim=-1` or `dim=1`.
13. `LogSoftmax` is more numerically stable than `log(softmax(x))`.
14. Activation choice affects training dynamics and final performance.
15. Match output activation to loss: sigmoid+BCE, softmax+CrossEntropy (or logits+CrossEntropy).
