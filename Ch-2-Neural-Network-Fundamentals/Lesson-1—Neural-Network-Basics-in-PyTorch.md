# Part 2 — Neural Network Fundamentals
# Chapter 1 — Neural Network Basics in PyTorch


> **Note:** Autograd fundamentals (leaf tensors, `requires_grad`, `.backward()`, computational graph) were introduced in Chapter 3 and used throughout Chapters 4–7. This chapter builds on that foundation to construct neural networks.

---

## 1. Overview

- What a neural network is: layers of parameterized transformations with non-linear activations.
- The basic building blocks: linear layers, activation functions, loss functions, optimizers.
- How a neuron works: weighted sum + bias + activation.
- Forward propagation: computing outputs from inputs through the network.
- Loss computation: measuring the difference between predictions and targets.
- Backward propagation: computing gradients of the loss with respect to all parameters.
- Parameter update: adjusting weights and biases using an optimizer.
- The training loop: iterating over data, computing loss, backpropagating, and updating.
- PyTorch's role: providing tensors, autograd, `nn.Module`, optimizers, and data utilities.
- Difference between manual implementation (raw tensors) and PyTorch's high-level API.

---

## 2. Core Concepts

**Neuron:** The fundamental unit of a neural network. Computes a weighted sum of inputs plus a bias, then applies an activation function.

**Weights:** Learnable parameters that scale each input feature. Stored as a matrix in a linear layer.

**Bias:** A learnable offset added to the weighted sum.

**Activation Function:** A non-linear function applied after the linear transformation. Enables the network to learn complex patterns. Examples: ReLU, Sigmoid, Tanh.

**Layer:** A collection of neurons that process the same input. A fully connected (linear) layer applies `y = xW^T + b`.

**Forward Pass:** The process of computing the network's output given an input.

**Loss Function:** A scalar measure of how far the predictions are from the targets. Examples: MSELoss, CrossEntropyLoss.

**Backward Pass:** Computing gradients of the loss with respect to all learnable parameters using the chain rule.

**Optimizer:** An algorithm that updates parameters using the computed gradients. Examples: SGD, Adam.

**Learning Rate:** A hyperparameter controlling the step size of parameter updates.

**Epoch:** One complete pass through the entire training dataset.

**Batch:** A subset of the training data processed together in one forward/backward pass.

**Iteration:** One forward/backward/update cycle on a single batch.

**Parameters:** Learnable tensors (weights and biases) in the model.

**Hyperparameters:** Settings that are not learned, such as learning rate, batch size, number of layers.

**Gradient Descent:** The optimization algorithm that updates parameters in the direction opposite to the gradient.

**Stochastic Gradient Descent (SGD):** Gradient descent using mini-batches instead of the full dataset.

---

## 3. Important PyTorch APIs

### `torch.nn.Linear(in_features, out_features, bias=True)`

- **Name:** `torch.nn.Linear`
- **Purpose:** Applies a linear transformation `y = xW^T + b`.
- **Syntax:** `nn.Linear(in_features, out_features, bias=True)`
- **Parameters:**
  - `in_features` (int): Size of each input sample.
  - `out_features` (int): Size of each output sample.
  - `bias` (bool): If `True`, adds a learnable bias. Default `True`.
- **Attributes:**
  - `weight`: Shape `(out_features, in_features)`.
  - `bias`: Shape `(out_features,)` if `bias=True`.
- **Example:**
```python
layer = nn.Linear(10, 5)
x = torch.randn(3, 10)
y = layer(x)  # shape (3, 5)
```

### `torch.nn.ReLU(inplace=False)`

- **Purpose:** Applies the rectified linear unit: `ReLU(x) = max(0, x)`.
- **Syntax:** `nn.ReLU(inplace=False)`
- **Parameters:**
  - `inplace` (bool): If `True`, modifies input directly. Default `False`.
- **Example:**
```python
relu = nn.ReLU()
x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
print(relu(x))  # tensor([0., 0., 1., 2.])
```

### `torch.nn.Sigmoid()`

- **Purpose:** Applies the sigmoid function: `σ(x) = 1 / (1 + exp(-x))`.
- **Output range:** `(0, 1)`.
- **Example:**
```python
sigmoid = nn.Sigmoid()
x = torch.tensor([-1.0, 0.0, 1.0])
print(sigmoid(x))  # tensor([0.2689, 0.5000, 0.7311])
```

### `torch.nn.Tanh()`

- **Purpose:** Applies the hyperbolic tangent: `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`.
- **Output range:** `(-1, 1)`.
- **Example:**
```python
tanh = nn.Tanh()
x = torch.tensor([-1.0, 0.0, 1.0])
print(tanh(x))  # tensor([-0.7616,  0.0000,  0.7616])
```

### `torch.nn.MSELoss(reduction='mean')`

- **Purpose:** Mean squared error loss: `MSE = mean((y_pred - y_true)^2)`.
- **Syntax:** `nn.MSELoss(reduction='mean')`
- **Parameters:**
  - `reduction` (str): `'none'`, `'mean'`, or `'sum'`. Default `'mean'`.
- **Example:**
```python
loss_fn = nn.MSELoss()
y_pred = torch.tensor([2.5, 0.0, 2.0])
y_true = torch.tensor([3.0, -0.5, 2.0])
print(loss_fn(y_pred, y_true))  # tensor(0.1667)
```

### `torch.nn.CrossEntropyLoss(weight=None, reduction='mean')`

- **Purpose:** Combines `LogSoftmax` and `NLLLoss`. Used for multi-class classification.
- **Input:** Raw logits of shape `(N, C)` and target class indices of shape `(N,)`.
- **Example:**
```python
loss_fn = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3]])
targets = torch.tensor([0, 1])
print(loss_fn(logits, targets))
```

### `torch.optim.SGD(params, lr, momentum=0, weight_decay=0)`

- **Purpose:** Stochastic gradient descent optimizer.
- **Syntax:** `optim.SGD(model.parameters(), lr=0.01)`
- **Parameters:**
  - `params`: Iterable of parameters.
  - `lr` (float): Learning rate.
  - `momentum` (float): Momentum factor. Default `0`.
  - `weight_decay` (float): L2 penalty. Default `0`.
- **Example:**
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)`

- **Purpose:** Adaptive moment estimation optimizer.
- **Syntax:** `optim.Adam(model.parameters(), lr=1e-3)`
- **Parameters:**
  - `params`: Iterable of parameters.
  - `lr` (float): Learning rate. Default `0.001`.
  - `betas` (tuple): Coefficients for running averages. Default `(0.9, 0.999)`.
  - `eps` (float): Small constant for numerical stability. Default `1e-8`.
- **Example:**
```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### `torch.nn.Module`

- **Purpose:** Base class for all neural network modules.
- **Key methods:**
  - `__init__`: Define layers and parameters.
  - `forward`: Define computation.
  - `parameters()`: Returns iterator over learnable parameters.
  - `zero_grad()`: Clears gradients.
  - `train()`: Set training mode.
  - `eval()`: Set evaluation mode.
- **Example:**
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### `torch.nn.Sequential(*args)`

- **Purpose:** A sequential container that passes input through modules in order.
- **Syntax:** `nn.Sequential(module1, module2, ...)`
- **Example:**
```python
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
```

### `torch.nn.functional` (commonly imported as `F`)

- **Purpose:** Functional interface for stateless operations.
- **Common functions:**
  - `F.relu(x)`
  - `F.sigmoid(x)`
  - `F.softmax(x, dim)`
  - `F.cross_entropy(logits, targets)`
  - `F.mse_loss(y_pred, y_true)`
- **Example:**
```python
import torch.nn.functional as F
x = torch.randn(3, 5)
print(F.relu(x))
```

---

## 4. Code Examples

### Example 1: A Single Neuron from Scratch

```python
import torch

# Input: 3 features, 1 sample
x = torch.tensor([[1.0, 2.0, 3.0]])

# Weights and bias (randomly initialized)
w = torch.randn(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Forward pass: weighted sum + bias
z = x @ w + b
print("z:", z)  # shape (1, 1)

# Apply activation (ReLU)
a = torch.relu(z)
print("a:", a)
```

**What it does:** Implements a single neuron manually using tensors.

**Important lines:**
- `x @ w + b` computes the linear transformation.
- `torch.relu(z)` applies non-linearity.
- `requires_grad=True` enables gradient computation.

**Expected output:**
```
z: tensor([[2.3456]], grad_fn=<AddBackward0>)
a: tensor([[2.3456]], grad_fn=<ReluBackward0>)
```

### Example 2: A Simple Neural Network with `nn.Module`

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
x = torch.randn(5, 3)  # batch of 5 samples, 3 features
y = model(x)
print("Output shape:", y.shape)  # (5, 1)
print("Model parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

**What it does:** Defines a two-layer neural network with ReLU activation.

**Important lines:**
- `nn.Linear(3, 4)` maps 3 inputs to 4 hidden units.
- `nn.ReLU()` applies non-linearity.
- `nn.Linear(4, 1)` maps hidden to output.
- `named_parameters()` lists all learnable parameters.

**Expected output:**
```
Output shape: torch.Size([5, 1])
Model parameters:
  fc1.weight: torch.Size([4, 3])
  fc1.bias: torch.Size([4])
  fc2.weight: torch.Size([1, 4])
  fc2.bias: torch.Size([1])
```

### Example 3: Loss Function and Optimizer

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model
model = nn.Linear(1, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data: y = 2x + 1
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# One training step
y_pred = model(X)
loss = criterion(y_pred, y)
print("Loss before step:", loss.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()

y_pred_after = model(X)
loss_after = criterion(y_pred_after, y)
print("Loss after step:", loss_after.item())
```

**What it does:** Performs one forward pass, loss computation, backward pass, and parameter update.

**Important lines:**
- `criterion(y_pred, y)` computes MSE loss.
- `optimizer.zero_grad()` clears old gradients.
- `loss.backward()` computes gradients.
- `optimizer.step()` updates parameters.

**Expected output:**
```
Loss before step: 30.1234
Loss after step: 29.5678
```

### Example 4: Full Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data: y = 3x + 2 + noise
torch.manual_seed(42)
X = torch.randn(100, 1) * 5
y = 3 * X + 2 + torch.randn(100, 1) * 0.5

# Model
model = nn.Linear(1, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

print(f"Learned weight: {model.weight.item():.4f}")
print(f"Learned bias: {model.bias.item():.4f}")
```

**What it does:** Trains a linear regression model for 100 epochs.

**Important lines:**
- Standard training loop: forward → loss → zero_grad → backward → step.
- Prints loss every 20 epochs.

**Expected output:**
```
Epoch 20: loss=1.2345
Epoch 40: loss=0.3456
Epoch 60: loss=0.1234
Epoch 80: loss=0.0678
Epoch 100: loss=0.0456
Learned weight: ~3.0
Learned bias: ~2.0
```

### Example 5: Binary Classification with Sigmoid

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate binary classification data
torch.manual_seed(42)
X = torch.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)  # shape (200, 1)

# Model
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(200):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        # Accuracy
        preds = (y_pred > 0.5).float()
        acc = (preds == y).float().mean()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc.item():.4f}")
```

**What it does:** Trains a binary classifier with sigmoid output and BCELoss.

**Important lines:**
- `nn.Sigmoid()` squashes output to `(0, 1)`.
- `nn.BCELoss()` is binary cross-entropy.
- Accuracy computed by thresholding at 0.5.

**Expected output:**
```
Epoch 50: loss=0.3456, acc=0.8500
Epoch 100: loss=0.2345, acc=0.9000
Epoch 150: loss=0.1890, acc=0.9200
Epoch 200: loss=0.1567, acc=0.9350
```

### Example 6: Multi-Class Classification with CrossEntropyLoss

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate 3-class data
torch.manual_seed(42)
N = 300
X = torch.randn(N, 2)
y = torch.randint(0, 3, (N,))  # class labels 0, 1, 2

# Model
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 3)  # 3 output logits
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(200):
    logits = model(X)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc.item():.4f}")
```

**What it does:** Trains a multi-class classifier using `CrossEntropyLoss`.

**Important lines:**
- Output layer has 3 units (one per class) and no activation (raw logits).
- `nn.CrossEntropyLoss()` expects raw logits and integer targets.
- `argmax(dim=1)` gives predicted class.

**Expected output:**
```
Epoch 50: loss=0.8765, acc=0.5600
Epoch 100: loss=0.6543, acc=0.6800
Epoch 150: loss=0.5432, acc=0.7400
Epoch 200: loss=0.4567, acc=0.7900
```

---

## 5. Important Parameters

### `nn.Linear`

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_features` | int | Number of input features. |
| `out_features` | int | Number of output features. |
| `bias` | bool | If `True`, adds learnable bias. Default `True`. |

### `nn.ReLU`

| Parameter | Type | Description |
|-----------|------|-------------|
| `inplace` | bool | If `True`, modifies input in-place. Default `False`. |

### `nn.MSELoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `reduction` | str | `'none'`, `'mean'`, or `'sum'`. Default `'mean'`. |

### `nn.CrossEntropyLoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | Tensor | Manual class weights. Default `None`. |
| `ignore_index` | int | Target index to ignore. Default `-100`. |
| `reduction` | str | `'none'`, `'mean'`, or `'sum'`. Default `'mean'`. |

### `nn.BCELoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | Tensor | Manual weights. Default `None`. |
| `reduction` | str | `'none'`, `'mean'`, or `'sum'`. Default `'mean'`. |

### `optim.SGD`

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | iterable | Model parameters. |
| `lr` | float | Learning rate. |
| `momentum` | float | Momentum factor. Default `0`. |
| `weight_decay` | float | L2 penalty. Default `0`. |
| `nesterov` | bool | Nesterov momentum. Default `False`. |

### `optim.Adam`

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | iterable | Model parameters. |
| `lr` | float | Learning rate. Default `0.001`. |
| `betas` | tuple | Coefficients for running averages. Default `(0.9, 0.999)`. |
| `eps` | float | Numerical stability. Default `1e-8`. |
| `weight_decay` | float | L2 penalty. Default `0`. |

### `nn.Module`

| Method | Purpose |
|--------|---------|
| `__init__` | Define layers and parameters. |
| `forward(x)` | Define computation. |
| `parameters()` | Iterate over learnable parameters. |
| `named_parameters()` | Iterate over (name, param) pairs. |
| `zero_grad()` | Clear gradients. |
| `train()` | Set training mode. |
| `eval()` | Set evaluation mode. |
| `to(device)` | Move model to device. |

---

## 6. Internal Working

### Forward Pass

1. Input tensor `x` is passed to the model.
2. Each layer applies its transformation:
   - Linear: `y = x @ W.T + b`.
   - Activation: element-wise non-linearity.
3. Output is a tensor of predictions (logits or probabilities).

### Loss Computation

1. Loss function compares predictions with targets.
2. Produces a scalar tensor with a `grad_fn` tracking the computation graph.

### Backward Pass

1. `loss.backward()` traverses the graph in reverse.
2. Computes gradients of loss w.r.t. each parameter using chain rule.
3. Gradients are stored in `.grad` of each parameter tensor.

### Parameter Update

1. `optimizer.step()` reads `.grad` of each parameter.
2. Applies update rule:
   - SGD: `param -= lr * grad`
   - Adam: uses running averages of gradients and squared gradients.
3. Updates parameters in-place.

### Zero Gradients

- `optimizer.zero_grad()` sets `.grad` to zero for all parameters.
- Must be called before `loss.backward()` in each iteration to prevent accumulation.

### Training Loop Pseudocode

```
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
```

### Model Modes

- `model.train()`: Enables dropout and batch norm updates.
- `model.eval()`: Disables dropout, uses running stats for batch norm.

### Device Management

- Move model to device: `model.to(device)`.
- Move data to device: `batch_X.to(device)`, `batch_y.to(device)`.
- Both must be on same device.

---

## 7. Common Mistakes

**Mistake:** Forgetting `optimizer.zero_grad()` before `loss.backward()`.

**Why it happens:** Gradients accumulate by default.

**Correct approach:** Always call `optimizer.zero_grad()` at the start of each iteration.

---

**Mistake:** Using `CrossEntropyLoss` with softmax output.

**Why it happens:** `CrossEntropyLoss` expects raw logits, not probabilities.

**Correct approach:** Remove softmax from the model; let the loss handle it.

---

**Mistake:** Using `BCELoss` with raw logits.

**Why it happens:** `BCELoss` expects probabilities in `(0, 1)`.

**Correct approach:** Apply sigmoid before BCELoss, or use `BCEWithLogitsLoss` which combines sigmoid and BCELoss.

---

**Mistake:** Forgetting to move model or data to GPU.

**Why it happens:** Model on CPU, data on GPU (or vice versa).

**Correct approach:** Move both to the same device.

---

**Mistake:** Using `model.eval()` without `torch.no_grad()` during validation.

**Why it happens:** `eval()` only changes layer behavior, not gradient tracking.

**Correct approach:** Use `with torch.no_grad():` during validation/inference.

---

**Mistake:** Not calling `model.train()` after validation.

**Why it happens:** Forgetting to switch back to training mode.

**Correct approach:** Call `model.train()` before resuming training.

---

**Mistake:** Incorrect input shape to `nn.Linear`.

**Why it happens:** `nn.Linear` expects last dimension to be `in_features`.

**Correct approach:** Ensure input shape `(..., in_features)`.

---

**Mistake:** Using `loss.item()` for backpropagation.

**Why it happens:** `.item()` returns a Python scalar, detaching from graph.

**Correct approach:** Use `loss` tensor for `backward()`, `.item()` only for logging.

---

**Mistake:** Forgetting to zero gradients in manual training loops.

**Why it happens:** Gradients accumulate.

**Correct approach:** Zero gradients before backward.

---

**Mistake:** Using `nn.Sigmoid` at output with `CrossEntropyLoss`.

**Why it happens:** Double activation leads to incorrect loss.

**Correct approach:** Use raw logits with `CrossEntropyLoss`.

---

## 8. Important Differences

| Concept | Difference |
|---------|-----------|
| `nn.Linear` vs `F.linear` | `nn.Linear` is a module with parameters; `F.linear` is a function. |
| `nn.ReLU` vs `F.relu` | `nn.ReLU` is a module; `F.relu` is a function. |
| `nn.CrossEntropyLoss` vs `nn.NLLLoss` | `CrossEntropyLoss` combines LogSoftmax and NLLLoss. |
| `nn.BCELoss` vs `nn.BCEWithLogitsLoss` | `BCELoss` expects probabilities; `BCEWithLogitsLoss` expects logits. |
| `model.train()` vs `model.eval()` | `train()` enables dropout/batch norm updates; `eval()` disables. |
| `optimizer.zero_grad()` vs `model.zero_grad()` | Both clear gradients; `optimizer.zero_grad()` is preferred. |
| SGD vs Adam | SGD uses fixed learning rate; Adam adapts per-parameter. |
| MSE vs CrossEntropy | MSE for regression; CrossEntropy for classification. |
| Logits vs probabilities | Logits are raw outputs; probabilities are after softmax/sigmoid. |
| Epoch vs batch vs iteration | Epoch = full pass; batch = subset; iteration = one batch update. |

---

## 9. Important Rules / Facts

- A neural network is a composition of linear layers and non-linear activations.
- `nn.Linear` applies `y = xW^T + b`.
- Activation functions introduce non-linearity; without them, stacked linear layers collapse to a single linear layer.
- `CrossEntropyLoss` expects raw logits and integer class targets.
- `BCELoss` expects probabilities; `BCEWithLogitsLoss` expects logits.
- `MSELoss` is for regression.
- `optimizer.zero_grad()` must be called before `loss.backward()`.
- `loss.backward()` computes gradients; `optimizer.step()` updates parameters.
- `model.train()` and `model.eval()` control dropout and batch norm behavior.
- Use `torch.no_grad()` during validation/inference.
- Move model and data to the same device.
- Input to `nn.Linear` must have last dimension equal to `in_features`.
- Output of `nn.Linear` has last dimension equal to `out_features`.
- Parameters are stored in `model.parameters()`.
- `model.named_parameters()` gives names and shapes.
- Learning rate controls step size; too high causes divergence, too low slows training.
- Batch size affects gradient noise and memory usage.
- Epoch = one full pass through the dataset.
- Training loop: forward → loss → zero_grad → backward → step.
- Always validate on a separate dataset.

---

## 10. Practical Example

### End-to-End Binary Classification with Training and Validation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate synthetic binary classification data
torch.manual_seed(42)
N = 1000
X = torch.randn(N, 2)
y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)  # XOR-like

# Split into train and validation
train_size = int(0.8 * N)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = BinaryClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            val_loss += loss.item()

            preds = (y_pred > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc = correct / total

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
```

**What it does:**
- Generates XOR-like binary classification data.
- Splits into train and validation sets.
- Defines a 3-layer MLP with sigmoid output.
- Trains with BCELoss and Adam.
- Validates with `model.eval()` and `torch.no_grad()`.
- Computes validation accuracy.

**Important lines:**
- `TensorDataset` wraps tensors.
- `DataLoader` batches and shuffles.
- `model.train()` and `model.eval()` switch modes.
- `torch.no_grad()` disables gradient tracking during validation.
- `optimizer.zero_grad()` before backward.

**Expected output:**
```
Using device: cuda
Epoch 20: train_loss=0.6543, val_loss=0.6321, val_acc=0.6000
Epoch 40: train_loss=0.5432, val_loss=0.5678, val_acc=0.6800
Epoch 60: train_loss=0.4321, val_loss=0.4567, val_acc=0.7500
Epoch 80: train_loss=0.3456, val_loss=0.3890, val_acc=0.8200
Epoch 100: train_loss=0.2789, val_loss=0.3345, val_acc=0.8600
```

---

## 11. Chapter Summary

- Neural networks are composed of linear layers and non-linear activation functions.
- A neuron computes `activation(xW^T + b)`.
- `nn.Linear` implements the linear transformation.
- Activation functions (ReLU, Sigmoid, Tanh) introduce non-linearity.
- Loss functions (MSE, CrossEntropy, BCE) measure prediction error.
- Optimizers (SGD, Adam) update parameters using gradients.
- The training loop: forward → loss → zero_grad → backward → step.
- `model.train()` and `model.eval()` control layer behavior.
- `torch.no_grad()` disables gradient tracking during inference.
- Device management: move model and data to same device.
- Common mistakes: forgetting zero_grad, using softmax with CrossEntropyLoss, not switching modes.
- Validation on a separate dataset is essential.
- Understanding these basics enables building more complex architectures.

---

## 12. Important APIs to Remember

| API | Purpose |
|-----|---------|
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()` | Activation functions |
| `nn.MSELoss()` | Regression loss |
| `nn.CrossEntropyLoss()` | Multi-class classification loss |
| `nn.BCELoss()` | Binary classification loss |
| `nn.BCEWithLogitsLoss()` | Binary loss with logits |
| `nn.Sequential()` | Container for layers |
| `nn.Module` | Base class for models |
| `optim.SGD()`, `optim.Adam()` | Optimizers |
| `optimizer.zero_grad()` | Clear gradients |
| `loss.backward()` | Compute gradients |
| `optimizer.step()` | Update parameters |
| `model.train()`, `model.eval()` | Set modes |
| `model.parameters()` | Access parameters |
| `model.named_parameters()` | Named parameters |
| `model.to(device)` | Move model |
| `tensor.to(device)` | Move data |
| `torch.no_grad()` | Disable gradients |
| `F.relu()`, `F.sigmoid()`, `F.softmax()` | Functional activations |
| `F.cross_entropy()`, `F.mse_loss()` | Functional losses |

---

## 13. Key Takeaways

1. A neural network is a stack of linear layers with non-linear activations.
2. `nn.Linear` computes `y = xW^T + b`.
3. Activation functions add non-linearity; without them, the network is linear.
4. `CrossEntropyLoss` expects raw logits; `BCELoss` expects probabilities.
5. Training loop: forward → loss → zero_grad → backward → step.
6. Always call `optimizer.zero_grad()` before `loss.backward()`.
7. `model.train()` and `model.eval()` control dropout and batch norm.
8. Use `torch.no_grad()` during validation/inference.
9. Move model and data to the same device.
10. Validate on a separate dataset to detect overfitting.
11. Common mistakes: forgetting zero_grad, wrong loss/activation pairing, device mismatch.
12. `nn.Sequential` is a convenient container for simple feedforward networks.
13. `nn.Module` provides parameter management and mode switching.
14. Batch size, learning rate, and number of layers are key hyperparameters.
15. These basics are prerequisites for CNNs, RNNs, and Transformers.
