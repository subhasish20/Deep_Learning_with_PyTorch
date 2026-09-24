# Part 2 — Neural Network Fundamentals
# Chapter 6 — Forward Pass

**Prerequisite:** Part 2 — Chapter 1 — Neural Network Basics in PyTorch, Part 2 — Chapter 2 — `torch.nn`, Part 2 — Chapter 3 — `nn.Module`, Part 2 — Chapter 4 — Layers, Part 2 — Chapter 5 — Activation Functions

---

## 1. Overview

- The forward pass is the computation that transforms input data into model predictions.
- It is defined by the `forward()` method of an `nn.Module` subclass.
- The forward pass builds the dynamic computation graph used by autograd for backpropagation.
- Each layer applies its transformation in sequence, producing intermediate activations.
- Shape tracking through the forward pass is essential for debugging.
- Hooks can intercept inputs and outputs at any point in the forward pass.
- The forward pass runs under different modes: training vs evaluation, gradient tracking vs no-grad.
- Conditional and dynamic forward passes allow data-dependent computation.
- Common patterns: residual connections, attention, branching, and multi-input/multi-output models.
- Understanding the forward pass is prerequisite to understanding backpropagation.

---

## 2. Core Concepts

**Forward Pass:** The sequence of computations that transforms input tensors into output predictions. In PyTorch, it is implemented in the `forward()` method of an `nn.Module`.

**`forward()` Method:** The user-defined method that specifies how input is transformed. It is called internally by `__call__`.

**`__call__`:** The method invoked when you do `model(x)`. It runs hooks, then calls `forward(x)`, then runs post-hooks.

**Computation Graph:** A directed acyclic graph built during the forward pass, recording every operation on tensors that require gradients. Used by autograd for backpropagation.

**Intermediate Activation:** Any tensor produced during the forward pass before the final output.

**Logits:** Raw, unnormalized outputs of the final linear layer. Input to loss functions like `CrossEntropyLoss`.

**Forward Hook:** A function called after `forward()` of a module. Signature: `hook(module, input, output)`.

**Forward Pre-Hook:** A function called before `forward()`. Signature: `hook(module, input)`.

**Training Mode:** `model.train()` sets `self.training = True`. Affects Dropout and BatchNorm.

**Evaluation Mode:** `model.eval()` sets `self.training = False`. Disables Dropout and uses running stats for BatchNorm.

**`torch.no_grad()`:** A context manager that disables gradient tracking. Used during inference.

**Dynamic Forward Pass:** A forward pass whose control flow depends on input values or shapes (e.g., early exit, conditional layers).

**Multi-Input Model:** A model whose `forward()` accepts multiple arguments (e.g., images and metadata).

**Multi-Output Model:** A model whose `forward()` returns multiple tensors (e.g., classification logits and bounding boxes).

**Skip Connection / Residual Connection:** A shortcut that adds the input of a block to its output, enabling gradient flow in deep networks.

**Weight Sharing:** Using the same module or parameter in multiple parts of the forward pass.

**Receptive Field:** The region of input that influences a particular output element.

---

## 3. Important PyTorch APIs

### `nn.Module.forward(*input)`

- **Purpose:** User-defined method that computes the output from input.
- **Syntax:** `def forward(self, x): ...`
- **Important:** Never call `forward()` directly; use `model(x)` to invoke `__call__`.
- **Example:**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
```

### `nn.Module.__call__(*input, **kwargs)`

- **Purpose:** Invoked when calling `model(x)`. Runs pre-hooks, `forward`, and post-hooks.
- **Important behavior:** Handles hooks and other module logic. Always use `model(x)` instead of `model.forward(x)`.

### `nn.Module.register_forward_pre_hook(hook)`

- **Purpose:** Register a function to run before `forward()`.
- **Hook signature:** `hook(module, input) -> None or modified input`
- **Return:** A handle with `remove()` method.
- **Example:**
```python
def pre_hook(module, input):
    print("Input shape:", input[0].shape)
handle = model.fc.register_forward_pre_hook(pre_hook)
```

### `nn.Module.register_forward_hook(hook)`

- **Purpose:** Register a function to run after `forward()`.
- **Hook signature:** `hook(module, input, output) -> None or modified output`
- **Return:** A handle with `remove()` method.
- **Example:**
```python
def hook_fn(module, input, output):
    print("Output shape:", output.shape)
handle = model.fc.register_forward_hook(hook_fn)
```

### `torch.no_grad()`

- **Purpose:** Context manager that disables gradient tracking.
- **Syntax:** `with torch.no_grad(): ...`
- **Important behavior:** Reduces memory usage and speeds up computation during inference.
- **Example:**
```python
with torch.no_grad():
    y = model(x)
```

### `torch.inference_mode()`

- **Purpose:** A stricter, faster context manager than `torch.no_grad()`. Disables gradient tracking and version counter tracking.
- **Syntax:** `with torch.inference_mode(): ...`
- **Important:** Tensors created inside cannot be used in autograd later. Use only for pure inference.
- **Example:**
```python
with torch.inference_mode():
    y = model(x)
```

### `nn.Module.training`

- **Purpose:** Boolean attribute indicating whether the module is in training mode.
- **Syntax:** `self.training`
- **Example:**
```python
def forward(self, x):
    if self.training:
        x = F.dropout(x, p=0.5)
    return x
```

### `nn.Module.train(mode=True)` / `nn.Module.eval()`

- **Purpose:** Set training/evaluation mode recursively.
- **Example:**
```python
model.train()   # training mode
model.eval()    # evaluation mode
```

### `torch.Tensor.register_hook(hook)`

- **Purpose:** Register a hook on a tensor to be called during backward.
- **Hook signature:** `hook(grad) -> Tensor or None`
- **Example:**
```python
def tensor_hook(grad):
    print("Gradient shape:", grad.shape)
x = torch.randn(3, requires_grad=True)
x.register_hook(tensor_hook)
```

### `torch.jit.trace` / `torch.jit.script`

- **Purpose:** Capture the forward pass as a graph for deployment.
- **Note:** Covered in later chapters (TorchScript).

### `torch.func.functional_call(module, parameters, input)`

- **Purpose:** Call a module's forward with a different set of parameters.
- **Syntax:** `torch.func.functional_call(module, params_dict, input)`
- **Use case:** Functional transforms, meta-learning.
- **Example:**
```python
from torch.func import functional_call
params = dict(model.named_parameters())
y = functional_call(model, params, (x,))
```

---

## 4. Code Examples

### Example 1: Basic Forward Pass

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
x = torch.randn(4, 10)  # batch of 4 samples, 10 features
y = model(x)            # invokes __call__ -> forward

print("Input shape:", x.shape)   # (4, 10)
print("Output shape:", y.shape)  # (4, 1)
print("Output:", y)
```

**What it does:** Defines a simple two-layer network and runs a forward pass.

**Important lines:**
- `model(x)` invokes `__call__`, which calls `forward(x)`.
- `torch.relu` is applied element-wise.

**Expected output:**
```
Input shape: torch.Size([4, 10])
Output shape: torch.Size([4, 1])
Output: tensor([[...], ...], grad_fn=<AddBackward0>)
```

### Example 2: Tracking Intermediate Shapes

```python
import torch
import torch.nn as nn

class ShapeTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        print("Input:", x.shape)
        x = torch.relu(self.conv1(x))
        print("After conv1:", x.shape)
        x = self.pool(x)
        print("After pool1:", x.shape)
        x = torch.relu(self.conv2(x))
        print("After conv2:", x.shape)
        x = self.pool(x)
        print("After pool2:", x.shape)
        x = x.view(x.size(0), -1)
        print("After flatten:", x.shape)
        x = self.fc(x)
        print("After fc:", x.shape)
        return x

model = ShapeTracker()
x = torch.randn(2, 3, 32, 32)
y = model(x)
```

**What it does:** Prints shapes at each stage of the forward pass.

**Important lines:**
- `x.shape` after each operation.
- `x.view(x.size(0), -1)` flattens keeping batch dimension.

**Expected output:**
```
Input: torch.Size([2, 3, 32, 32])
After conv1: torch.Size([2, 16, 32, 32])
After pool1: torch.Size([2, 16, 16, 16])
After conv2: torch.Size([2, 32, 16, 16])
After pool2: torch.Size([2, 32, 8, 8])
After flatten: torch.Size([2, 2048])
After fc: torch.Size([2, 10])
```

### Example 3: Forward Hooks

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Store outputs from hooks
activations = {}

def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model[0].register_forward_hook(get_activation('fc1'))
model[2].register_forward_hook(get_activation('fc2'))

x = torch.randn(4, 10)
y = model(x)

print("Stored activations:")
for name, act in activations.items():
    print(f"  {name}: {act.shape}")
```

**What it does:** Uses forward hooks to capture intermediate activations.

**Important lines:**
- `register_forward_hook` attaches a function.
- `output.detach()` removes from graph for storage.
- Hooks are useful for feature extraction and visualization.

**Expected output:**
```
Stored activations:
  fc1: torch.Size([4, 20])
  fc2: torch.Size([4, 5])
```

### Example 4: Forward Pre-Hooks for Input Modification

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)

# Pre-hook that adds noise to input
def add_noise_hook(module, input):
    x = input[0]
    return (x + torch.randn_like(x) * 0.1,)

handle = model.register_forward_pre_hook(add_noise_hook)

x = torch.randn(4, 10)
y1 = model(x)
print("With noise hook, output shape:", y1.shape)

# Remove hook
handle.remove()
y2 = model(x)
print("Without hook, output shape:", y2.shape)
```

**What it does:** Uses a forward pre-hook to modify the input before the linear layer.

**Important lines:**
- Pre-hook receives `(module, input)` and returns modified input tuple.
- `handle.remove()` unregisters the hook.

**Expected output:**
```
With noise hook, output shape: torch.Size([4, 5])
Without hook, output shape: torch.Size([4, 5])
```

### Example 5: Training vs Evaluation Mode

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModeAwareNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

model = ModeAwareNet()
x = torch.randn(8, 10)

# Training mode
model.train()
y_train = model(x)
print("Training mode:")
print("  model.training:", model.training)
print("  dropout.training:", model.dropout.training)
print("  Output shape:", y_train.shape)

# Evaluation mode
model.eval()
y_eval = model(x)
print("Evaluation mode:")
print("  model.training:", model.training)
print("  dropout.training:", model.dropout.training)
print("  Output shape:", y_eval.shape)

# Verify dropout differs
model.train()
y1 = model(x)
y2 = model(x)
print("\nDropout active in training (y1 != y2):", not torch.allclose(y1, y2))

model.eval()
y3 = model(x)
y4 = model(x)
print("Dropout inactive in eval (y3 == y4):", torch.allclose(y3, y4))
```

**What it does:** Shows how `train()`/`eval()` affect Dropout and BatchNorm.

**Important lines:**
- `model.train()` and `model.eval()` set modes recursively.
- Dropout is stochastic in training, identity in eval.
- BatchNorm uses batch stats in training, running stats in eval.

**Expected output:**
```
Training mode:
  model.training: True
  dropout.training: True
  Output shape: torch.Size([8, 1])
Evaluation mode:
  model.training: False
  dropout.training: False
  Output shape: torch.Size([8, 1])

Dropout active in training (y1 != y2): True
Dropout inactive in eval (y3 == y4): True
```

### Example 6: No-Grad and Inference Mode

```python
import torch
import torch.nn as nn
import time

model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10)
)

x = torch.randn(64, 1000)

# With gradient tracking
model.train()
start = time.time()
for _ in range(100):
    y = model(x)
    loss = y.sum()
    loss.backward()
    model.zero_grad()
time_grad = time.time() - start

# Without gradient tracking
model.eval()
start = time.time()
with torch.no_grad():
    for _ in range(100):
        y = model(x)
time_no_grad = time.time() - start

# Inference mode
start = time.time()
with torch.inference_mode():
    for _ in range(100):
        y = model(x)
time_inference = time.time() - start

print(f"With grad: {time_grad:.4f}s")
print(f"No grad: {time_no_grad:.4f}s")
print(f"Inference mode: {time_inference:.4f}s")
print(f"Speedup (no_grad vs grad): {time_grad / time_no_grad:.2f}x")
print(f"Speedup (inference vs grad): {time_grad / time_inference:.2f}x")
```

**What it does:** Compares forward pass speed with and without gradient tracking.

**Important lines:**
- `torch.no_grad()` disables gradient tracking.
- `torch.inference_mode()` is stricter and faster.
- Both reduce memory and increase speed.

**Expected output:**
```
With grad: 0.5678s
No grad: 0.3456s
Inference mode: 0.3123s
Speedup (no_grad vs grad): 1.64x
Speedup (inference vs grad): 1.82x
```

### Example 7: Conditional Forward Pass (Early Exit)

```python
import torch
import torch.nn as nn

class EarlyExitNet(nn.Module):
    def __init__(self, num_classes=10, confidence_threshold=0.9):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x, return_exit_layer=False):
        x = torch.relu(self.fc1(x))

        # Early exit check on first block
        logits = self.out(x)
        probs = torch.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)

        # For samples with high confidence, exit early
        confident = max_probs > self.confidence_threshold

        # If all confident, return immediately
        if confident.all():
            if return_exit_layer:
                return logits, "early"
            return logits

        # Otherwise, continue for uncertain samples
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        logits = self.out(x)

        if return_exit_layer:
            return logits, "full"
        return logits

model = EarlyExitNet(confidence_threshold=0.5)
x = torch.randn(4, 10)
logits, exit_layer = model(x, return_exit_layer=True)
print("Exit layer:", exit_layer)
print("Output shape:", logits.shape)
```

**What it does:** Demonstrates a forward pass with data-dependent control flow.

**Important lines:**
- Conditional logic in `forward()` is fully supported by PyTorch.
- The computation graph reflects the actual path taken.

**Expected output:**
```
Exit layer: early  (or "full")
Output shape: torch.Size([4, 10])
```

### Example 8: Multi-Input and Multi-Output Model

```python
import torch
import torch.nn as nn

class MultiModalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Image branch
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        # Metadata branch
        self.meta_fc = nn.Linear(5, 8)

        # Combined head
        self.fc = nn.Linear(16, 1)
        # Auxiliary output for image branch
        self.aux = nn.Linear(8, 1)

    def forward(self, image, metadata):
        img_features = self.conv(image)      # (N, 8)
        meta_features = torch.relu(self.meta_fc(metadata))  # (N, 8)
        combined = torch.cat([img_features, meta_features], dim=1)  # (N, 16)
        main_out = self.fc(combined)          # (N, 1)
        aux_out = self.aux(img_features)      # (N, 1)
        return main_out, aux_out

model = MultiModalNet()
images = torch.randn(4, 3, 32, 32)
metadata = torch.randn(4, 5)
main_out, aux_out = model(images, metadata)
print("Main output shape:", main_out.shape)  # (4, 1)
print("Aux output shape:", aux_out.shape)    # (4, 1)
```

**What it does:** Demonstrates a model with multiple inputs and multiple outputs.

**Important lines:**
- `forward(self, image, metadata)` accepts multiple arguments.
- Returns a tuple of outputs.

**Expected output:**
```
Main output shape: torch.Size([4, 1])
Aux output shape: torch.Size([4, 1])
```

### Example 9: Residual Connection in Forward

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # skip connection
        out = F.relu(out)
        return out

block = ResidualBlock(16)
x = torch.randn(2, 16, 32, 32)
y = block(x)
print("Input shape:", x.shape)
print("Output shape:", y.shape)  # same shape

# Demonstrate gradient flow through skip connection
x = torch.randn(2, 16, 32, 32, requires_grad=True)
y = block(x)
y.sum().backward()
print("Input gradient shape:", x.grad.shape)
print("Gradient is non-zero:", (x.grad != 0).any().item())
```

**What it does:** Shows a residual connection in the forward pass.

**Important lines:**
- `out = out + identity` adds the input to the output.
- Skip connections enable gradient flow in deep networks.

**Expected output:**
```
Input shape: torch.Size([2, 16, 32, 32])
Output shape: torch.Size([2, 16, 32, 32])
Input gradient shape: torch.Size([2, 16, 32, 32])
Gradient is non-zero: True
```

### Example 10: Functional Forward with `torch.func.functional_call`

```python
import torch
import torch.nn as nn
from torch.func import functional_call

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Get parameters as a dict
params = dict(model.named_parameters())

# Modify parameters on the fly
new_params = {k: v.clone() for k, v in params.items()}
new_params['0.weight'] = new_params['0.weight'] * 2

x = torch.randn(4, 10)

# Standard forward
y1 = model(x)

# Functional call with modified parameters
y2 = functional_call(model, new_params, (x,))

print("Standard output:", y1.shape)
print("Functional call output:", y2.shape)
print("Outputs differ:", not torch.allclose(y1, y2))
```

**What it does:** Uses `torch.func.functional_call` to run the forward pass with alternate parameters.

**Important lines:**
- `functional_call(model, params_dict, input)` runs the forward pass with given parameters.
- Useful for meta-learning and functional transforms.

**Expected output:**
```
Standard output: torch.Size([4, 1])
Functional call output: torch.Size([4, 1])
Outputs differ: True
```

---

## 5. Important Parameters

### `nn.Module.forward`

| Parameter | Type | Description |
|-----------|------|-------------|
| `*input` | Tensor or tuple | Input tensors. Can accept multiple arguments. |

### `nn.Module.__call__`

| Parameter | Type | Description |
|-----------|------|-------------|
| `*input` | Tensor or tuple | Same as `forward`. |
| `**kwargs` | dict | Additional keyword arguments. |

### `register_forward_hook(hook)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `hook` | callable | Function `hook(module, input, output)`. Can modify output. |

### `register_forward_pre_hook(hook)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `hook` | callable | Function `hook(module, input)`. Can modify input. |

### `torch.no_grad()`

| Parameter | Type | Description |
|-----------|------|-------------|
| None | — | Context manager. Disables gradient tracking. |

### `torch.inference_mode()`

| Parameter | Type | Description |
|-----------|------|-------------|
| None | — | Context manager. Disables gradient tracking and version counters. |

### `torch.func.functional_call(module, parameters, input)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | nn.Module | Module whose forward is called. |
| `parameters` | dict | Mapping of parameter names to tensors. |
| `input` | tuple | Input tensors. |

---

## 6. Internal Working

### Execution of `model(x)`

1. Python calls `model.__call__(x)`.
2. `__call__` runs all registered forward pre-hooks.
3. `forward(x)` is called with the (possibly modified) input.
4. `__call__` runs all registered forward hooks.
5. The output (possibly modified by hooks) is returned.

### Building the Computation Graph

During the forward pass:

1. Each operation on a tensor with `requires_grad=True` creates a new tensor.
2. The new tensor has a `grad_fn` attribute referencing the operation.
3. This builds a directed acyclic graph (DAG) from outputs back to leaf tensors.
4. The graph is used during `backward()` for gradient computation.

### Intermediate Activations

- Intermediate activations are tensors produced by layers.
- They are stored in memory during the forward pass if needed for backward.
- PyTorch frees intermediate activations after backward unless retained.

### Forward Hooks

- Hooks are called in registration order.
- Forward pre-hooks run before `forward`.
- Forward hooks run after `forward`.
- Hooks can modify input/output by returning new values.

### Mode Effects

- `model.train()`: `self.training = True` for all submodules.
- `model.eval()`: `self.training = False` for all submodules.
- Dropout: zeroes elements with probability `p` in training; identity in eval.
- BatchNorm: uses batch statistics in training; running statistics in eval.

### `torch.no_grad()` Internals

- Disables `requires_grad` tracking for operations inside the context.
- Tensors created inside have `requires_grad=False`.
- Saves memory by not storing intermediate activations.
- Speeds up computation by skipping graph construction.

### `torch.inference_mode()` Internals

- Stricter than `no_grad`: also disables version counter tracking.
- Faster but tensors cannot be used in autograd later.
- Should be used only for pure inference.

### Dynamic Control Flow

- Python `if`, `for`, `while` statements are executed normally.
- The computation graph reflects the actual path taken.
- This is the "define-by-run" paradigm of PyTorch.

### Residual Connection

- `out = out + identity` adds the input to the output.
- The gradient flows through both the main path and the skip connection.
- This helps mitigate vanishing gradients in deep networks.

---

## 7. Common Mistakes

**Mistake:** Calling `model.forward(x)` directly instead of `model(x)`.

**Why it happens:** Directly calling `forward` bypasses hooks and `__call__` logic.

**Correct approach:** Always use `model(x)`.

---

**Mistake:** Not switching between `train()` and `eval()` modes.

**Why it happens:** Dropout and BatchNorm behave differently.

**Correct approach:** Call `model.train()` before training and `model.eval()` before inference.

---

**Mistake:** Forgetting `torch.no_grad()` during validation/inference.

**Why it happens:** Gradients are tracked unnecessarily, wasting memory and time.

**Correct approach:** Wrap inference in `with torch.no_grad():` or `with torch.inference_mode():`.

---

**Mistake:** Assuming `forward()` is called directly.

**Why it happens:** The `__call__` method invokes `forward`, but not directly.

**Correct approach:** Define `forward`; call `model(x)`.

---

**Mistake:** Modifying input in-place in the forward pass when it's needed for backward.

**Why it happens:** In-place operations can corrupt the autograd graph.

**Correct approach:** Use out-of-place operations or clone inputs.

---

**Mistake:** Forgetting to track shapes during the forward pass.

**Why it happens:** Shape mismatches are a common source of errors.

**Correct approach:** Print or assert shapes at key points.

---

**Mistake:** Using `torch.inference_mode()` and then trying to use the output in autograd.

**Why it happens:** Tensors from `inference_mode` cannot be used in autograd.

**Correct approach:** Use `torch.no_grad()` if the output will be used later in a graph.

---

**Mistake:** Not moving input to the same device as the model.

**Why it happens:** Device mismatch.

**Correct approach:** `x = x.to(device)` before `model(x)`.

---

**Mistake:** Registering hooks and forgetting to remove them.

**Why it happens:** Hooks persist and can cause unexpected behavior.

**Correct approach:** Keep the handle and call `handle.remove()` when done.

---

**Mistake:** Incorrect dtype in input to layers like `nn.Embedding`.

**Why it happens:** Embedding requires `torch.long` indices.

**Correct approach:** Ensure `x = x.long()` before embedding.

---

**Mistake:** Using conditional logic that depends on tensor values in a way that breaks tracing.

**Why it happens:** TorchScript tracing cannot handle data-dependent control flow.

**Correct approach:** Use `torch.jit.script` for dynamic control flow, or restructure.

---

**Mistake:** Not handling multiple inputs/outputs correctly.

**Why it happens:** `forward` can accept and return multiple tensors.

**Correct approach:** Use tuples or dicts for multi-input/output.

---

## 8. Important Differences

| Concept | Difference |
|---------|-----------|
| `model(x)` vs `model.forward(x)` | `model(x)` runs hooks and `__call__`; `forward` is the raw computation. |
| `train()` vs `eval()` | `train()` enables dropout/batch norm updates; `eval()` disables. |
| `torch.no_grad()` vs `torch.inference_mode()` | `inference_mode` is stricter and faster but tensors cannot be used in autograd. |
| Forward hook vs pre-hook | Pre-hook runs before `forward`; hook runs after. |
| Training mode vs eval mode | Dropout and BatchNorm differ. |
| Static vs dynamic forward | PyTorch is dynamic; control flow is Python-native. |
| Single-output vs multi-output | `forward` can return one or multiple tensors. |
| Single-input vs multi-input | `forward` can accept one or multiple arguments. |
| Residual vs sequential | Residual adds input to output; sequential just chains. |

---

## 9. Important Rules / Facts

- The forward pass transforms input into predictions.
- It is implemented in the `forward()` method of `nn.Module`.
- `model(x)` invokes `__call__`, which runs hooks and then `forward`.
- The forward pass builds the computation graph for autograd.
- Intermediate activations are stored for backward.
- `torch.no_grad()` disables gradient tracking during inference.
- `torch.inference_mode()` is stricter and faster than `no_grad`.
- `model.train()` and `model.eval()` set modes recursively.
- Dropout is active only in training mode.
- BatchNorm uses batch stats in training and running stats in eval.
- Forward pre-hooks run before `forward`; forward hooks run after.
- Hooks can modify input or output.
- Always use `model(x)`, not `model.forward(x)`.
- The forward pass can contain arbitrary Python control flow.
- Multi-input and multi-output models are supported.
- Residual connections add input to output.
- Shape tracking is essential for debugging.
- Device consistency between model and input is required.
- In-place operations can break autograd.
- Use `torch.func.functional_call` for functional forward passes.

---

## 10. Practical Example

### Building and Running a Complete Forward Pass with Hooks, Modes, and Shape Tracking

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompleteNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        # Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CompleteNet(num_classes=10).to(device)

# Register forward hooks to track activations
activation_shapes = {}
def make_hook(name):
    def hook(module, input, output):
        activation_shapes[name] = output.shape
    return hook

handles = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
        handles.append(module.register_forward_hook(make_hook(name)))

# Create input
x = torch.randn(4, 3, 32, 32).to(device)

# Training mode forward
print("=== Training Mode ===")
model.train()
y_train = model(x)
print("Output shape:", y_train.shape)
print("Training mode:", model.training)

# Print activation shapes
print("\nActivation shapes:")
for name, shape in activation_shapes.items():
    print(f"  {name}: {shape}")

# Evaluation mode forward
print("\n=== Evaluation Mode ===")
activation_shapes.clear()
model.eval()
with torch.no_grad():
    y_eval = model(x)
print("Output shape:", y_eval.shape)
print("Eval mode:", model.training)

# Clean up hooks
for h in handles:
    h.remove()

# Compute loss and backward (training)
print("\n=== Loss and Backward ===")
model.train()
criterion = nn.CrossEntropyLoss()
targets = torch.randint(0, 10, (4,)).to(device)
logits = model(x)
loss = criterion(logits, targets)
print("Loss:", loss.item())
loss.backward()
print("Gradient computed for fc2.weight:", model.fc2.weight.grad is not None)
```

**What it does:**
- Builds a CNN and runs forward passes in training and evaluation modes.
- Uses forward hooks to track activation shapes.
- Computes loss and runs backward.

**Important lines:**
- `model.train()` and `model.eval()` switch modes.
- `with torch.no_grad():` for inference.
- Hooks capture intermediate shapes.
- `loss.backward()` computes gradients.

**Expected output:**
```
=== Training Mode ===
Output shape: torch.Size([4, 10])
Training mode: True

Activation shapes:
  conv1: torch.Size([4, 32, 32, 32])
  bn1: torch.Size([4, 32, 32, 32])
  conv2: torch.Size([4, 64, 16, 16])
  bn2: torch.Size([4, 64, 16, 16])
  fc1: torch.Size([4, 128])
  fc2: torch.Size([4, 10])

=== Evaluation Mode ===
Output shape: torch.Size([4, 10])
Eval mode: False

=== Loss and Backward ===
Loss: 2.3456
Gradient computed for fc2.weight: True
```

---

## 11. Chapter Summary

- The forward pass transforms input into predictions via `forward()`.
- `model(x)` invokes `__call__`, which runs hooks and `forward`.
- The forward pass builds the computation graph for autograd.
- Intermediate activations are stored for backward unless in `no_grad`.
- `torch.no_grad()` and `torch.inference_mode()` disable gradient tracking.
- `model.train()` and `model.eval()` control layer behavior.
- Dropout is active only in training; BatchNorm uses running stats in eval.
- Forward pre-hooks run before `forward`; forward hooks run after.
- Hooks can modify input or output.
- PyTorch supports dynamic control flow in `forward`.
- Multi-input and multi-output models are supported.
- Residual connections add input to output.
- Shape tracking is essential for debugging.
- Device consistency is required.
- In-place operations can break autograd.
- `torch.func.functional_call` runs forward with alternate parameters.
- Common mistakes: calling `forward` directly, not switching modes, forgetting `no_grad`.
- Understanding the forward pass is prerequisite to backpropagation.

---

## 12. Important APIs to Remember

| API | Purpose |
|-----|---------|
| `nn.Module.forward` | User-defined computation |
| `nn.Module.__call__` | Invokes forward with hooks |
| `register_forward_hook` | Post-forward hook |
| `register_forward_pre_hook` | Pre-forward hook |
| `torch.no_grad()` | Disable gradient tracking |
| `torch.inference_mode()` | Stricter no-grad mode |
| `model.train()` | Set training mode |
| `model.eval()` | Set evaluation mode |
| `model.training` | Boolean mode flag |
| `torch.func.functional_call` | Forward with alternate params |
| `tensor.register_hook` | Tensor backward hook |

---

## 13. Key Takeaways

1. The forward pass is defined by `forward()` and invoked via `model(x)`.
2. `model(x)` runs hooks and then `forward`; never call `forward` directly.
3. The forward pass builds the computation graph for backpropagation.
4. `torch.no_grad()` disables gradient tracking; `torch.inference_mode()` is faster.
5. `model.train()` and `model.eval()` control Dropout and BatchNorm behavior.
6. Forward pre-hooks run before `forward`; forward hooks run after.
7. Hooks can modify input or output and are useful for debugging.
8. PyTorch supports dynamic Python control flow in `forward`.
9. Multi-input and multi-output models are natural in PyTorch.
10. Residual connections add input to output, aiding gradient flow.
11. Shape tracking is essential for debugging the forward pass.
12. Device consistency between model and input is required.
13. In-place operations can corrupt autograd; use out-of-place.
14. `torch.func.functional_call` enables functional forward passes.
15. Mastering the forward pass is prerequisite to understanding backpropagation.


# Loss Functions

**Prerequisite:** Part 2 — Chapter 1 — Neural Network Basics in PyTorch, Part 2 — Chapter 2 — `torch.nn`, Part 2 — Chapter 3 — `nn.Module`, Part 2 — Chapter 4 — Layers, Part 2 — Chapter 5 — Activation Functions, Part 2 — Chapter 6 — Forward Pass

---

## 1. Overview

- Loss functions quantify the difference between model predictions and true targets.
- They produce a scalar tensor whose gradient drives parameter updates via backpropagation.
- Loss functions are task-specific: regression, binary classification, multi-class classification, ranking, segmentation, detection, and more.
- PyTorch provides losses in `torch.nn` (module form) and `torch.nn.functional` (functional form).
- Reduction modes (`'none'`, `'mean'`, `'sum'`) control how per-sample losses are aggregated.
- Loss functions must match the output activation and target format.
- Custom loss functions can be implemented as `nn.Module` subclasses or plain functions.
- Numerical stability is crucial: log-sum-exp tricks, epsilon clipping, and stable implementations.
- Weighting, masking, and ignoring indices are common requirements in real datasets.
- Understanding loss functions is essential for correct training and debugging.

---

## 2. Core Concepts

**Loss Function:** A scalar function `L(y_pred, y_true)` that measures how far predictions are from targets. Minimizing this loss during training improves the model.

**Objective / Cost Function:** The quantity minimized during training. Usually the average loss over the dataset.

**Reduction:** How per-element or per-sample losses are combined. Options: `'none'` (no reduction), `'mean'` (average), `'sum'` (sum).

**Logits:** Raw, unnormalized outputs of the final linear layer. Many classification losses expect logits, not probabilities.

**Probabilities:** Outputs after sigmoid or softmax. Required by some losses (e.g., `BCELoss`).

**Target:** The ground-truth label or value. Format depends on the loss (integer class index, one-hot, float, etc.).

**Regression Loss:** Measures error between continuous predictions and targets. Examples: MSE, MAE, Huber.

**Classification Loss:** Measures error between predicted class scores and true classes. Examples: CrossEntropy, BCE, NLL.

**Cross-Entropy:** Measures the difference between two probability distributions. For classification: `L = -sum(y_true * log(y_pred))`.

**Binary Cross-Entropy (BCE):** Cross-entropy for binary classification. `L = -(y * log(p) + (1-y) * log(1-p))`.

**Negative Log-Likelihood (NLL):** `L = -log(p_true)`. Used after log-softmax.

**Kullback-Leibler Divergence (KLDiv):** Measures how one probability distribution diverges from another.

**Hinge Loss:** Used in SVMs. `L = max(0, 1 - y * y_pred)`.

**Huber Loss:** Smooth combination of MSE and MAE. Robust to outliers.

**Focal Loss:** Down-weights easy examples, focuses on hard ones. Used in detection.

**Triplet Loss:** Encourages anchor-positive pairs to be closer than anchor-negative pairs.

**Contrastive Loss:** Pulls similar pairs together and pushes dissimilar pairs apart.

**Dice Loss:** Used in segmentation. Measures overlap between predicted and true masks.

**IoU Loss:** Intersection over Union. Used in detection and segmentation.

**CTC Loss:** Connectionist Temporal Classification. Used for sequence alignment without explicit alignment.

**Class Weights:** Per-class scaling factors in the loss to handle class imbalance.

**Ignore Index:** Target value that is ignored in the loss (e.g., padding tokens).

**Masking:** Applying a mask to exclude certain positions from the loss.

---

## 3. Important PyTorch APIs

### `torch.nn.MSELoss(reduction='mean')`

- **Purpose:** Mean squared error: `L = (y_pred - y_true)^2`.
- **Parameters:**
  - `reduction` (str): `'none'`, `'mean'`, or `'sum'`. Default `'mean'`.
- **Formula:**
  ```
  L = (1/N) * sum_i (y_pred_i - y_true_i)^2
  ```
- **Use case:** Regression.
- **Example:**
```python
loss_fn = nn.MSELoss()
y_pred = torch.tensor([2.5, 0.0, 2.0])
y_true = torch.tensor([3.0, -0.5, 2.0])
print(loss_fn(y_pred, y_true))  # tensor(0.1667)
```

### `torch.nn.L1Loss(reduction='mean')`

- **Purpose:** Mean absolute error: `L = |y_pred - y_true|`.
- **Formula:**
  ```
  L = (1/N) * sum_i |y_pred_i - y_true_i|
  ```
- **Use case:** Regression, robust to outliers.
- **Example:**
```python
loss_fn = nn.L1Loss()
print(loss_fn(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5])))  # tensor(0.5)
```

### `torch.nn.SmoothL1Loss(beta=1.0, reduction='mean')`

- **Purpose:** Huber loss. Smooth L1. Quadratic for small errors, linear for large.
- **Formula:**
  ```
  L = 0.5 * (y_pred - y_true)^2 / beta          if |y_pred - y_true| < beta
  L = |y_pred - y_true| - 0.5 * beta            otherwise
  ```
- **Parameters:**
  - `beta` (float): Threshold. Default `1.0`.
  - `reduction` (str): Default `'mean'`.
- **Use case:** Regression with outliers (e.g., bounding box regression).
- **Example:**
```python
loss_fn = nn.SmoothL1Loss()
print(loss_fn(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5])))
```

### `torch.nn.HuberLoss(delta=1.0, reduction='mean')`

- **Purpose:** Huber loss, similar to SmoothL1.
- **Parameters:**
  - `delta` (float): Threshold. Default `1.0`.
  - `reduction` (str): Default `'mean'`.

### `torch.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)`

- **Purpose:** Combines `LogSoftmax` and `NLLLoss`. For multi-class classification.
- **Input:** Raw logits `(N, C)` and target class indices `(N,)`.
- **Formula:**
  ```
  L = -log( exp(logits[target]) / sum_j exp(logits[j]) )
  ```
- **Parameters:**
  - `weight` (Tensor): Manual class weights `(C,)`. Default `None`.
  - `ignore_index` (int): Target value to ignore. Default `-100`.
  - `reduction` (str): `'none'`, `'mean'`, or `'sum'`. Default `'mean'`.
  - `label_smoothing` (float): Smoothing factor. Default `0.0`.
- **Example:**
```python
loss_fn = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3]])
targets = torch.tensor([0, 1])
print(loss_fn(logits, targets))
```

### `torch.nn.NLLLoss(weight=None, ignore_index=-100, reduction='mean')`

- **Purpose:** Negative log-likelihood. Expects log-probabilities.
- **Input:** Log-probabilities `(N, C)` and target indices `(N,)`.
- **Use case:** After `LogSoftmax`.
- **Example:**
```python
log_probs = F.log_softmax(torch.randn(4, 3), dim=1)
targets = torch.tensor([0, 1, 2, 0])
loss_fn = nn.NLLLoss()
print(loss_fn(log_probs, targets))
```

### `torch.nn.BCELoss(weight=None, reduction='mean')`

- **Purpose:** Binary cross-entropy. Expects probabilities in `(0, 1)`.
- **Input:** Predictions `(N, *)` and targets `(N, *)` of same shape.
- **Formula:**
  ```
  L = -(y * log(p) + (1-y) * log(1-p))
  ```
- **Parameters:**
  - `weight` (Tensor): Manual weights. Default `None`.
  - `reduction` (str): Default `'mean'`.
- **Use case:** Binary classification or multi-label classification.
- **Example:**
```python
loss_fn = nn.BCELoss()
preds = torch.tensor([0.8, 0.2, 0.6])
targets = torch.tensor([1.0, 0.0, 1.0])
print(loss_fn(preds, targets))
```

### `torch.nn.BCEWithLogitsLoss(weight=None, pos_weight=None, reduction='mean')`

- **Purpose:** Combines sigmoid and BCE. Expects raw logits.
- **Input:** Logits `(N, *)` and targets `(N, *)`.
- **Parameters:**
  - `weight` (Tensor): Element-wise weights.
  - `pos_weight` (Tensor): Weight for positive examples.
  - `reduction` (str): Default `'mean'`.
- **Use case:** Binary classification, more numerically stable than `BCELoss`.
- **Example:**
```python
loss_fn = nn.BCEWithLogitsLoss()
logits = torch.tensor([1.5, -0.5, 0.8])
targets = torch.tensor([1.0, 0.0, 1.0])
print(loss_fn(logits, targets))
```

### `torch.nn.KLDivLoss(reduction='mean', log_target=False)`

- **Purpose:** Kullback-Leibler divergence: `L = y_true * (log(y_true) - log(y_pred))`.
- **Input:** Log-probabilities `(N, C)` and probabilities `(N, C)`.
- **Parameters:**
  - `reduction` (str): Default `'mean'`. Note: `'mean'` is not equal to `'batchmean'`.
  - `log_target` (bool): If `True`, target is log-probabilities.
- **Use case:** Distribution matching, knowledge distillation.
- **Example:**
```python
loss_fn = nn.KLDivLoss(reduction='batchmean')
log_preds = F.log_softmax(torch.randn(4, 3), dim=1)
targets = F.softmax(torch.randn(4, 3), dim=1)
print(loss_fn(log_preds, targets))
```

### `torch.nn.HingeEmbeddingLoss(margin=1.0, reduction='mean')`

- **Purpose:** Hinge loss for embedding learning.
- **Formula:**
  ```
  L = x                        if y == 1
  L = max(0, margin - x)       if y == -1
  ```

### `torch.nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean')`

- **Purpose:** Multi-label classification with one-vs-all.
- **Input:** Logits `(N, C)` and binary targets `(N, C)`.

### `torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, reduction='mean')`

- **Purpose:** Multi-class hinge loss.

### `torch.nn.SoftMarginLoss(reduction='mean')`

- **Purpose:** Two-class logistic loss for arbitrary targets.

### `torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean')`

- **Purpose:** Triplet loss for metric learning.
- **Input:** Anchor, positive, negative `(N, D)`.
- **Formula:**
  ```
  L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
  ```
- **Example:**
```python
loss_fn = nn.TripletMarginLoss(margin=1.0)
anchor = torch.randn(4, 128)
positive = torch.randn(4, 128)
negative = torch.randn(4, 128)
print(loss_fn(anchor, positive, negative))
```

### `torch.nn.TripletMarginWithDistanceLoss(distance_function, margin=1.0, reduction='mean')`

- **Purpose:** Triplet loss with custom distance function.

### `torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')`

- **Purpose:** Cosine embedding loss.
- **Input:** Two tensors and a target `y` (`1` for similar, `-1` for dissimilar).

### `torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)`

- **Purpose:** Connectionist Temporal Classification for sequence alignment.
- **Input:** Log-probabilities `(T, N, C)`, targets `(N, S)`, input lengths `(N,)`, target lengths `(N,)`.
- **Use case:** Speech recognition, handwriting recognition.

### `torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, reduction='mean')`

- **Purpose:** Negative log-likelihood for Poisson distribution.
- **Use case:** Count data regression.

### `torch.nn.GaussianNLLLoss(full=False, eps=1e-6, reduction='mean')`

- **Purpose:** Negative log-likelihood for Gaussian distribution.
- **Input:** Predictions, targets, variance.

### `torch.nn.MarginRankingLoss(margin=0.0, reduction='mean')`

- **Purpose:** Ranking loss. `L = max(0, -y * (x1 - x2) + margin)`.

### `torch.nn.FocalLoss` (Not built-in, custom)

- **Purpose:** Focal loss for dense object detection.
- **Formula:** `L = -alpha * (1 - p_t)^gamma * log(p_t)`.

### Functional Equivalents

| Module | Functional |
|--------|-----------|
| `nn.MSELoss()` | `F.mse_loss(input, target)` |
| `nn.L1Loss()` | `F.l1_loss(input, target)` |
| `nn.CrossEntropyLoss()` | `F.cross_entropy(logits, target)` |
| `nn.NLLLoss()` | `F.nll_loss(log_probs, target)` |
| `nn.BCELoss()` | `F.binary_cross_entropy(pred, target)` |
| `nn.BCEWithLogitsLoss()` | `F.binary_cross_entropy_with_logits(logits, target)` |
| `nn.SmoothL1Loss()` | `F.smooth_l1_loss(input, target)` |
| `nn.KLDivLoss()` | `F.kl_div(log_pred, target)` |
| `nn.TripletMarginLoss()` | `F.triplet_margin_loss(anchor, positive, negative)` |

---

## 4. Code Examples

### Example 1: Regression Losses

```python
import torch
import torch.nn as nn

y_pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
y_true = torch.tensor([3.0, -0.5, 2.0, 1.0])

# MSE
mse = nn.MSELoss()
print("MSE:", mse(y_pred, y_true).item())

# MAE
mae = nn.L1Loss()
print("MAE:", mae(y_pred, y_true).item())

# Smooth L1 (Huber)
huber = nn.SmoothL1Loss()
print("SmoothL1:", huber(y_pred, y_true).item())

# Reduction modes
mse_none = nn.MSELoss(reduction='none')
print("MSE none:", mse_none(y_pred, y_true))
print("MSE sum:", nn.MSELoss(reduction='sum')(y_pred, y_true).item())
```

**What it does:** Compares MSE, MAE, and Smooth L1 on the same data.

**Important lines:**
- MSE penalizes large errors heavily.
- MAE is linear.
- Smooth L1 is quadratic for small errors, linear for large.
- Reduction mode controls aggregation.

**Expected output:**
```
MSE: 12.1875
MAE: 2.375
SmoothL1: 2.0
MSE none: tensor([0.2500, 0.2500, 0.0000, 49.0000])
MSE sum: 48.75
```

### Example 2: CrossEntropyLoss for Multi-Class

```python
import torch
import torch.nn as nn

# Logits from a model with 3 classes
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3],
                       [0.1, 0.2, 3.0]])

# Targets: class indices
targets = torch.tensor([0, 1, 2])

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)
print("CrossEntropyLoss:", loss.item())

# Per-sample losses
loss_none = nn.CrossEntropyLoss(reduction='none')
print("Per-sample:", loss_none(logits, targets))

# With label smoothing
loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
print("With label smoothing:", loss_smooth(logits, targets).item())

# With class weights (class 0 is rare)
weights = torch.tensor([2.0, 1.0, 1.0])
loss_weighted = nn.CrossEntropyLoss(weight=weights)
print("With weights:", loss_weighted(logits, targets).item())
```

**What it does:** Demonstrates `CrossEntropyLoss` with various options.

**Important lines:**
- Input is raw logits, not softmax.
- Targets are integer class indices.
- `weight` handles class imbalance.
- `label_smoothing` softens targets.

**Expected output:**
```
CrossEntropyLoss: 0.3771
Per-sample: tensor([0.4170, 0.2407, 0.4736])
With label smoothing: 0.5491
With weights: 0.4775
```

### Example 3: BCEWithLogitsLoss for Binary Classification

```python
import torch
import torch.nn as nn

# Raw logits (no sigmoid applied)
logits = torch.tensor([2.0, -1.0, 0.5, -2.0])
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

# BCEWithLogitsLoss (recommended)
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, targets)
print("BCEWithLogitsLoss:", loss.item())

# BCELoss (requires probabilities)
probs = torch.sigmoid(logits)
loss_bce = nn.BCELoss()
print("BCELoss:", loss_bce(probs, targets).item())

# pos_weight for imbalanced data
pos_weight = torch.tensor([5.0])
loss_pos = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print("With pos_weight:", loss_pos(logits, targets).item())
```

**What it does:** Compares `BCEWithLogitsLoss` and `BCELoss`.

**Important lines:**
- `BCEWithLogitsLoss` is more numerically stable.
- `BCELoss` requires probabilities.
- `pos_weight` upweights positive examples.

**Expected output:**
```
BCEWithLogitsLoss: 0.4741
BCELoss: 0.4741
With pos_weight: 0.8138
```

### Example 4: NLLLoss with LogSoftmax

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.randn(4, 3)
targets = torch.tensor([0, 2, 1, 0])

# LogSoftmax + NLLLoss = CrossEntropyLoss
log_probs = F.log_softmax(logits, dim=1)
nll_loss = nn.NLLLoss()
print("NLLLoss:", nll_loss(log_probs, targets).item())

# Equivalent CrossEntropyLoss
ce_loss = nn.CrossEntropyLoss()
print("CrossEntropyLoss:", ce_loss(logits, targets).item())

# Verify equivalence
print("Equal:", abs(nll_loss(log_probs, targets).item() - ce_loss(logits, targets).item()) < 1e-6)
```

**What it does:** Shows that `LogSoftmax` + `NLLLoss` equals `CrossEntropyLoss`.

**Important lines:**
- `NLLLoss` expects log-probabilities.
- `CrossEntropyLoss` combines both.

**Expected output:**
```
NLLLoss: 1.2345
CrossEntropyLoss: 1.2345
Equal: True
```

### Example 5: Reduction Modes

```python
import torch
import torch.nn as nn

y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y_true = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

loss_fn = nn.MSELoss(reduction='none')
loss_none = loss_fn(y_pred, y_true)
print("none:\n", loss_none)

loss_mean = nn.MSELoss(reduction='mean')(y_pred, y_true)
print("mean:", loss_mean.item())

loss_sum = nn.MSELoss(reduction='sum')(y_pred, y_true)
print("sum:", loss_sum.item())

# Verify
print("mean == mean of none:", torch.allclose(loss_mean, loss_none.mean()))
print("sum == sum of none:", torch.allclose(loss_sum, loss_none.sum()))
```

**What it does:** Demonstrates the effect of reduction modes.

**Important lines:**
- `'none'` returns per-element losses.
- `'mean'` averages them.
- `'sum'` sums them.

**Expected output:**
```
none:
 tensor([[0.2500, 0.2500],
         [0.2500, 0.2500]])
mean: 0.25
sum: 1.0
mean == mean of none: True
sum == sum of none: True
```

### Example 6: Custom Loss Function

```python
import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        loss = (y_pred - y_true) ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Test
loss_fn = CustomMSELoss()
y_pred = torch.tensor([2.5, 0.0, 2.0])
y_true = torch.tensor([3.0, -0.5, 2.0])
print("Custom MSE:", loss_fn(y_pred, y_true).item())

# Compare with built-in
builtin = nn.MSELoss()
print("Built-in MSE:", builtin(y_pred, y_true).item())
```

**What it does:** Implements a custom MSE loss as an `nn.Module`.

**Important lines:**
- Custom losses can be `nn.Module` subclasses.
- `forward(y_pred, y_true)` computes the loss.

**Expected output:**
```
Custom MSE: 0.1667
Built-in MSE: 0.1667
```

### Example 7: Focal Loss (Custom Implementation)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Compute BCE with logits
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Compute p_t
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
logits = torch.tensor([2.0, -1.0, 0.5, -2.0])
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
print("Focal Loss:", loss_fn(logits, targets).item())

# Compare with BCE
bce_loss = nn.BCEWithLogitsLoss()
print("BCE Loss:", bce_loss(logits, targets).item())
```

**What it does:** Implements focal loss, which down-weights easy examples.

**Important lines:**
- `p_t` is the probability of the true class.
- `(1 - p_t)^gamma` down-weights easy examples.
- `alpha` balances positive/negative.

**Expected output:**
```
Focal Loss: 0.0891
BCE Loss: 0.4741
```

### Example 8: Triplet Loss for Metric Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Anchor, positive, negative embeddings
anchor = torch.randn(4, 128)
positive = anchor + torch.randn(4, 128) * 0.1  # close to anchor
negative = torch.randn(4, 128) * 2             # far from anchor

loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
loss = loss_fn(anchor, positive, negative)
print("Triplet Loss:", loss.item())

# With cosine distance
loss_cos = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
    margin=0.5
)
print("Triplet Loss (cosine):", loss_cos(anchor, positive, negative).item())
```

**What it does:** Demonstrates triplet loss for metric learning.

**Important lines:**
- Triplet loss pulls anchor and positive together, pushes anchor and negative apart.
- Margin controls the separation.

**Expected output:**
```
Triplet Loss: 0.8765
Triplet Loss (cosine): 0.4321
```

### Example 9: Masked Loss for Variable-Length Sequences

```python
import torch
import torch.nn as nn

# Batch of sequences with padding
# Predictions and targets have shape (batch, seq_len)
y_pred = torch.randn(3, 5)
y_true = torch.randn(3, 5)

# Mask: 1 for valid, 0 for padding
mask = torch.tensor([[1, 1, 1, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 1, 1]], dtype=torch.float32)

# Compute per-element loss
loss_fn = nn.MSELoss(reduction='none')
loss = loss_fn(y_pred, y_true)

# Apply mask
masked_loss = loss * mask
# Average over valid elements only
avg_loss = masked_loss.sum() / mask.sum()
print("Masked MSE:", avg_loss.item())

# Compare with unmasked
print("Unmasked MSE:", loss.mean().item())
```

**What it does:** Applies a mask to ignore padding in the loss.

**Important lines:**
- `reduction='none'` gives per-element loss.
- Multiply by mask and normalize by mask sum.
- Common in sequence models.

**Expected output:**
```
Masked MSE: ~0.85
Unmasked MSE: ~1.20
```

### Example 10: Loss with Ignore Index

```python
import torch
import torch.nn as nn

logits = torch.randn(4, 5)
targets = torch.tensor([0, 2, -100, 1])  # -100 is ignored

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fn(logits, targets)
print("Loss with ignore_index:", loss.item())

# Without ignoring
targets_no_ignore = torch.tensor([0, 2, 0, 1])
loss_no_ignore = nn.CrossEntropyLoss()(logits, targets_no_ignore)
print("Loss without ignore:", loss_no_ignore.item())
```

**What it does:** Demonstrates ignoring certain target values in the loss.

**Important lines:**
- `ignore_index=-100` is the default.
- Ignored samples do not contribute to the loss.

**Expected output:**
```
Loss with ignore_index: 1.6094
Loss without ignore: 1.7345
```

---

## 5. Important Parameters

### `nn.CrossEntropyLoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | Tensor | Per-class weights `(C,)`. Default `None`. |
| `ignore_index` | int | Target value to ignore. Default `-100`. |
| `reduction` | str | `'none'`, `'mean'`, `'sum'`. Default `'mean'`. |
| `label_smoothing` | float | Smoothing factor. Default `0.0`. |

### `nn.BCEWithLogitsLoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | Tensor | Element-wise weights. Default `None`. |
| `pos_weight` | Tensor | Weight for positive examples. Default `None`. |
| `reduction` | str | `'none'`, `'mean'`, `'sum'`. Default `'mean'`. |

### `nn.BCELoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | Tensor | Element-wise weights. Default `None`. |
| `reduction` | str | `'none'`, `'mean'`, `'sum'`. Default `'mean'`. |

### `nn.MSELoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `reduction` | str | `'none'`, `'mean'`, `'sum'`. Default `'mean'`. |

### `nn.L1Loss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `reduction` | str | `'none'`, `'mean'`, `'sum'`. Default `'mean'`. |

### `nn.SmoothL1Loss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `beta` | float | Threshold. Default `1.0`. |
| `reduction` | str | Default `'mean'`. |

### `nn.NLLLoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | Tensor | Per-class weights. Default `None`. |
| `ignore_index` | int | Default `-100`. |
| `reduction` | str | Default `'mean'`. |

### `nn.KLDivLoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `reduction` | str | Default `'mean'`. Use `'batchmean'` for correct KL. |
| `log_target` | bool | If `True`, target is log-probabilities. Default `False`. |

### `nn.TripletMarginLoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `margin` | float | Default `1.0`. |
| `p` | float | Norm degree. Default `2.0`. |
| `eps` | float | Default `1e-6`. |
| `swap` | bool | Swap positive/negative. Default `False`. |
| `reduction` | str | Default `'mean'`. |

### `nn.CTCLoss`

| Parameter | Type | Description |
|-----------|------|-------------|
| `blank` | int | Blank label index. Default `0`. |
| `reduction` | str | Default `'mean'`. |
| `zero_infinity` | bool | Zero infinite losses. Default `False`. |

---

## 6. Internal Working

### Loss as a Scalar

- Loss functions produce a scalar tensor (when `reduction='mean'` or `'sum'`).
- The scalar has a `grad_fn` that tracks the computation graph.
- Calling `.backward()` on the scalar computes gradients for all parameters.

### CrossEntropyLoss Internals

1. Applies `LogSoftmax` to the logits:
   ```
   log_probs = logits - log(sum(exp(logits)))
   ```
2. Selects the log-probability of the target class:
   ```
   nll = -log_probs[target]
   ```
3. Averages over the batch.

Implementation uses the log-sum-exp trick for numerical stability:
```
log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
```

### BCEWithLogitsLoss Internals

1. Applies sigmoid to logits:
   ```
   p = sigmoid(logits)
   ```
2. Computes binary cross-entropy:
   ```
   L = -(y * log(p) + (1-y) * log(1-p))
   ```
3. Uses a numerically stable formulation:
   ```
   L = max(x, 0) - x * y + log(1 + exp(-|x|))
   ```

### MSELoss Internals

```
L = (y_pred - y_true)^2
```

Element-wise, then reduced according to `reduction`.

### Reduction Modes

- `'none'`: Returns per-element or per-sample losses.
- `'mean'`: Averages over all elements (or weighted average if `weight` provided).
- `'sum'`: Sums all elements.

### Weight Handling

- For `CrossEntropyLoss`, `weight` is applied per class.
- Weighted mean: `sum(weight[target] * loss) / sum(weight[target])`.
- For `BCELoss`, `weight` is applied element-wise.

### Ignore Index

- Targets equal to `ignore_index` are excluded from the loss.
- The denominator for mean reduction excludes ignored samples.

### Label Smoothing

- Instead of hard targets (0 or 1), uses soft targets:
  ```
  y_smooth = (1 - epsilon) * y + epsilon / C
  ```
- Prevents overconfidence.

### Gradient Flow

- Loss gradients flow back through the computation graph.
- Each loss has a well-defined derivative w.r.t. its inputs.
- `CrossEntropyLoss` derivative w.r.t. logits: `softmax(logits) - one_hot(target)`.
- `MSELoss` derivative w.r.t. prediction: `2 * (y_pred - y_true) / N`.

### Numerical Stability

- Log-sum-exp trick for cross-entropy.
- Stable sigmoid for BCE.
- Epsilon clipping for log operations.
- Avoid `log(0)` and `exp(large)`.

---

## 7. Common Mistakes

**Mistake:** Applying softmax before `CrossEntropyLoss`.

**Why it happens:** `CrossEntropyLoss` includes `LogSoftmax`.

**Correct approach:** Pass raw logits.

---

**Mistake:** Using `BCELoss` with raw logits.

**Why it happens:** `BCELoss` expects probabilities.

**Correct approach:** Use `BCEWithLogitsLoss` or apply sigmoid first.

---

**Mistake:** Wrong target format for `CrossEntropyLoss`.

**Why it happens:** Targets must be integer class indices, not one-hot.

**Correct approach:** Use `targets.long()` with shape `(N,)`.

---

**Mistake:** Using `MSELoss` for classification.

**Why it happens:** MSE is for regression.

**Correct approach:** Use `CrossEntropyLoss` or `BCEWithLogitsLoss`.

---

**Mistake:** Forgetting `reduction='none'` when applying masks.

**Why it happens:** Default `'mean'` already reduces.

**Correct approach:** Use `reduction='none'` then apply mask and normalize.

---

**Mistake:** Not handling class imbalance.

**Why it happens:** Rare classes contribute little to loss.

**Correct approach:** Use `weight` in `CrossEntropyLoss` or `pos_weight` in `BCEWithLogitsLoss`.

---

**Mistake:** Using `ignore_index` with the wrong value.

**Why it happens:** Default is `-100`; padding may be `0`.

**Correct approach:** Set `ignore_index` to the padding value.

---

**Mistake:** Forgetting that `KLDivLoss` with `reduction='mean'` is not true KL.

**Why it happens:** `'mean'` divides by total elements, not batch size.

**Correct approach:** Use `reduction='batchmean'`.

---

**Mistake:** Not detaching loss for logging.

**Why it happens:** Loss tensor has graph; `.item()` detaches.

**Correct approach:** Use `loss.item()` for logging.

---

**Mistake:** Accumulating loss without `.item()`.

**Why it happens:** Accumulating tensors keeps graph alive.

**Correct approach:** `total_loss += loss.item()`.

---

**Mistake:** Using `TripletMarginLoss` with incorrect input shapes.

**Why it happens:** Anchor, positive, negative must have same shape.

**Correct approach:** Ensure all three have shape `(N, D)`.

---

**Mistake:** Ignoring numerical stability in custom losses.

**Why it happens:** `log(0)` or `exp(large)` causes NaN.

**Correct approach:** Add epsilon or use stable implementations.

---

**Mistake:** Using `reduction='sum'` without normalizing.

**Why it happens:** Loss magnitude depends on batch size.

**Correct approach:** Use `'mean'` or divide by batch size.

---

**Mistake:** Not matching loss to output activation.

**Why it happens:** Some losses include activation, some do not.

**Correct approach:** Know which losses expect logits vs probabilities.

---

## 8. Important Differences

| Concept | Difference |
|---------|-----------|
| MSE vs MAE | MSE squares errors; MAE takes absolute value. |
| MSE vs Smooth L1 | Smooth L1 is quadratic for small errors, linear for large. |
| CrossEntropyLoss vs NLLLoss | CrossEntropy includes LogSoftmax; NLL expects log-probs. |
| BCEWithLogitsLoss vs BCELoss | BCEWithLogits includes sigmoid; BCELoss expects probabilities. |
| CrossEntropyLoss vs BCEWithLogitsLoss | CrossEntropy for multi-class; BCE for binary/multi-label. |
| `'mean'` vs `'sum'` vs `'none'` | Aggregation modes. |
| `weight` vs `pos_weight` | `weight` is per-class/element; `pos_weight` upweights positives in BCE. |
| `ignore_index` vs masking | `ignore_index` ignores specific target values; masking applies custom mask. |
| Label smoothing vs hard targets | Smoothing prevents overconfidence. |
| Focal loss vs BCE | Focal down-weights easy examples. |
| Triplet loss vs Contrastive loss | Triplet uses anchor/positive/negative; contrastive uses pairs. |
| KLDiv `'mean'` vs `'batchmean'` | `'batchmean'` gives true KL divergence. |

---

## 9. Important Rules / Facts

- Loss functions produce a scalar for backpropagation.
- `CrossEntropyLoss` expects raw logits and integer targets.
- `BCEWithLogitsLoss` expects raw logits and float targets.
- `BCELoss` expects probabilities in `(0, 1)`.
- `NLLLoss` expects log-probabilities.
- `MSELoss`, `L1Loss`, `SmoothL1Loss` are for regression.
- `CrossEntropyLoss` combines `LogSoftmax` and `NLLLoss`.
- `BCEWithLogitsLoss` combines sigmoid and BCE.
- `reduction='mean'` is the default for most losses.
- `reduction='none'` returns per-element losses.
- Use `weight` or `pos_weight` for class imbalance.
- `ignore_index` excludes specific targets from the loss.
- `label_smoothing` softens hard targets.
- `KLDivLoss` with `reduction='batchmean'` gives true KL divergence.
- `TripletMarginLoss` requires anchor, positive, negative.
- `CTCLoss` is for sequence alignment.
- Custom losses can be `nn.Module` subclasses.
- Numerical stability is crucial: use stable loss implementations.
- Loss must match output activation and target format.
- Logging with `loss.item()` detaches from graph.

---

## 10. Practical Example

### Multi-Task Learning with Multiple Losses

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Classification head
        self.class_head = nn.Linear(64, num_classes)
        # Regression head
        self.reg_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.class_head(features)
        reg_output = self.reg_head(features)
        return class_logits, reg_output

# Create model and data
model = MultiTaskNet(num_classes=5)
x = torch.randn(16, 10)
class_targets = torch.randint(0, 5, (16,))
reg_targets = torch.randn(16, 1)

# Forward pass
class_logits, reg_output = model(x)

# Define losses
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

# Compute individual losses
loss_class = ce_loss(class_logits, class_targets)
loss_reg = mse_loss(reg_output, reg_targets)

# Weighted sum of losses
total_loss = loss_class + 0.5 * loss_reg

print("Classification loss:", loss_class.item())
print("Regression loss:", loss_reg.item())
print("Total loss:", total_loss.item())

# Backward pass
total_loss.backward()

# Check gradients
print("Backbone grad exists:", model.backbone[0].weight.grad is not None)
print("Class head grad exists:", model.class_head.weight.grad is not None)
print("Reg head grad exists:", model.reg_head.weight.grad is not None)
```

**What it does:**
- Builds a multi-task model with shared backbone and two heads.
- Computes classification and regression losses.
- Combines them with a weight.
- Backpropagates the total loss.

**Important lines:**
- Multiple losses are combined into a single scalar.
- `total_loss.backward()` propagates through both heads and the shared backbone.
- Loss weights control the balance between tasks.

**Expected output:**
```
Classification loss: 1.6094
Regression loss: 0.9876
Total loss: 2.1032
Backbone grad exists: True
Class head grad exists: True
Reg head grad exists: True
```

### Complete Training with Loss Selection

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(task='classification', epochs=50):
    # Generate data
    torch.manual_seed(42)
    N = 1000
    X = torch.randn(N, 2)

    if task == 'regression':
        y = 3 * X[:, 0] + 2 * X[:, 1] + torch.randn(N) * 0.1
        y = y.unsqueeze(1)
        criterion = nn.MSELoss()
        output_dim = 1
    elif task == 'binary':
        y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)
        criterion = nn.BCEWithLogitsLoss()
        output_dim = 1
    else:  # multiclass
        y = torch.randint(0, 3, (N,))
        criterion = nn.CrossEntropyLoss()
        output_dim = 3

    # Split
    train_size = int(0.8 * N)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    # Model
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim)
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_X, batch_y in val_loader:
                    output = model(batch_X)
                    val_loss += criterion(output, batch_y).item()
                val_loss /= len(val_loader)
            print(f"[{task}] Epoch {epoch+1}: val_loss={val_loss:.4f}")

    return model

# Train for different tasks
print("=== Regression ===")
train_model('regression', epochs=30)

print("\n=== Binary Classification ===")
train_model('binary', epochs=30)

print("\n=== Multi-Class Classification ===")
train_model('multiclass', epochs=30)
```

**What it does:**
- Trains models for regression, binary classification, and multi-class classification.
- Selects the appropriate loss for each task.
- Validates on a held-out set.

**Important lines:**
- Loss selection depends on task.
- `MSELoss` for regression.
- `BCEWithLogitsLoss` for binary.
- `CrossEntropyLoss` for multi-class.

**Expected output:**
```
=== Regression ===
[regression] Epoch 10: val_loss=0.0123
[regression] Epoch 20: val_loss=0.0098
[regression] Epoch 30: val_loss=0.0087

=== Binary Classification ===
[binary] Epoch 10: val_loss=0.4321
[binary] Epoch 20: val_loss=0.3456
[binary] Epoch 30: val_loss=0.2987

=== Multi-Class Classification ===
[multiclass] Epoch 10: val_loss=1.0987
[multiclass] Epoch 20: val_loss=1.0456
[multiclass] Epoch 30: val_loss=0.9876
```

---

## 11. Chapter Summary

- Loss functions measure the difference between predictions and targets.
- They produce a scalar for backpropagation.
- Regression: `MSELoss`, `L1Loss`, `SmoothL1Loss`, `HuberLoss`.
- Multi-class classification: `CrossEntropyLoss` (expects logits).
- Binary classification: `BCEWithLogitsLoss` (expects logits) or `BCELoss` (expects probabilities).
- `NLLLoss` expects log-probabilities; used with `LogSoftmax`.
- `KLDivLoss` with `reduction='batchmean'` for distribution matching.
- `TripletMarginLoss` for metric learning.
- `CTCLoss` for sequence alignment.
- `reduction` controls aggregation: `'none'`, `'mean'`, `'sum'`.
- `weight` and `pos_weight` handle class imbalance.
- `ignore_index` excludes specific targets.
- `label_smoothing` prevents overconfidence.
- Custom losses can be `nn.Module` subclasses.
- Numerical stability is crucial; use stable implementations.
- Match loss to output activation and target format.
- Common mistakes: softmax before CrossEntropy, BCE with logits, wrong target format.
- Loss selection is task-dependent.
- Multi-task learning combines multiple losses.
- Logging with `loss.item()` detaches from graph.

---

## 12. Important APIs to Remember

| API | Purpose |
|-----|---------|
| `nn.MSELoss` | Mean squared error |
| `nn.L1Loss` | Mean absolute error |
| `nn.SmoothL1Loss` | Huber loss (smooth L1) |
| `nn.HuberLoss` | Huber loss |
| `nn.CrossEntropyLoss` | Multi-class classification |
| `nn.NLLLoss` | Negative log-likelihood |
| `nn.BCELoss` | Binary cross-entropy (probabilities) |
| `nn.BCEWithLogitsLoss` | Binary cross-entropy (logits) |
| `nn.KLDivLoss` | KL divergence |
| `nn.TripletMarginLoss` | Triplet loss |
| `nn.TripletMarginWithDistanceLoss` | Triplet loss with custom distance |
| `nn.CosineEmbeddingLoss` | Cosine embedding loss |
| `nn.CTCLoss` | CTC loss |
| `nn.PoissonNLLLoss` | Poisson NLL |
| `nn.GaussianNLLLoss` | Gaussian NLL |
| `nn.MarginRankingLoss` | Ranking loss |
| `nn.MultiLabelSoftMarginLoss` | Multi-label classification |
| `nn.MultiMarginLoss` | Multi-class hinge loss |
| `nn.HingeEmbeddingLoss` | Hinge embedding loss |
| `nn.SoftMarginLoss` | Two-class logistic loss |
| `F.mse_loss`, `F.cross_entropy`, etc. | Functional equivalents |

---

## 13. Key Takeaways

1. Loss functions quantify prediction error and produce a scalar for backpropagation.
2. `MSELoss`, `L1Loss`, `SmoothL1Loss` are for regression.
3. `CrossEntropyLoss` is for multi-class classification; expects raw logits.
4. `BCEWithLogitsLoss` is for binary classification; expects raw logits.
5. `BCELoss` expects probabilities; `NLLLoss` expects log-probabilities.
6. `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`.
7. `BCEWithLogitsLoss` = sigmoid + `BCELoss`.
8. `reduction` controls aggregation: `'none'`, `'mean'`, `'sum'`.
9. `weight` and `pos_weight` handle class imbalance.
10. `ignore_index` excludes specific target values.
11. `label_smoothing` prevents overconfidence.
12. `KLDivLoss` with `reduction='batchmean'` gives true KL divergence.
13. `TripletMarginLoss` pulls positives together, pushes negatives apart.
14. Custom losses can be `nn.Module` subclasses.
15. Match loss to output activation and target format.
16. Numerical stability is crucial; use stable loss implementations.
17. Logging with `loss.item()` detaches from graph.
18. Multi-task learning combines multiple losses with weights.
19. Common mistakes: softmax before CrossEntropy, BCE with logits, wrong target format.
20. Loss selection depends on task: regression, binary, multi-class, ranking, etc.
