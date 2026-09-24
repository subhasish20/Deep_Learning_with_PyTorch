# Part 2 — Neural Network Fundamentals
# Chapter 2 — `torch.nn`

**Prerequisite:** Part 2 — Chapter 1 — Neural Network Basics in PyTorch

---

## 1. Overview

- `torch.nn` is the core package for building neural networks in PyTorch.
- It provides pre-built layers, activation functions, loss functions, and containers.
- Central class: `nn.Module` — base class for all neural network components.
- Parameter management: `nn.Parameter` and automatic registration of submodules.
- Containers: `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`.
- Functional interface: `torch.nn.functional` (commonly imported as `F`) for stateless operations.
- Relationship between `nn.Module` and `nn.functional`: modules hold state (parameters), functions do not.
- How `torch.nn` integrates with `torch.optim`, `torch.utils.data`, and autograd.
- Best practices for organizing complex models.

---

## 2. Core Concepts

**`torch.nn`:** A package containing building blocks for neural networks: layers, activations, losses, containers, and utilities.

**`nn.Module`:** The base class for all neural network modules. It provides:
- Parameter registration.
- Submodule registration.
- `forward` method for computation.
- Hooks for forward/backward passes.
- `train()`/`eval()` mode switching.
- Device movement (`to`, `cpu`, `cuda`).

**`nn.Parameter`:** A subclass of `torch.Tensor` that is automatically registered as a learnable parameter when assigned as an attribute of an `nn.Module`.

**Submodule:** An `nn.Module` assigned as an attribute of another `nn.Module`. It is automatically registered and its parameters are included in `parameters()`.

**Container:** A module that holds other modules. Examples: `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`.

**Functional (`torch.nn.functional`):** A module containing stateless functions (e.g., `F.relu`, `F.linear`, `F.cross_entropy`). These do not hold parameters.

**Stateless vs Stateful:** Modules are stateful (they hold parameters); functions are stateless.

**Parameter Registration:** When you assign an `nn.Parameter` to an attribute of an `nn.Module`, it is automatically added to the module's parameter list.

**Module Registration:** When you assign an `nn.Module` to an attribute of another `nn.Module`, it is automatically registered as a submodule.

**Forward Hook:** A function that can be registered to be called during the forward pass.

**`__call__` vs `forward`:** Calling `module(x)` invokes `__call__`, which runs hooks and then calls `forward(x)`.

---

## 3. Important PyTorch APIs

### `torch.nn.Module`

- **Name:** `torch.nn.Module`
- **Purpose:** Base class for all neural network modules.
- **Syntax:** Subclass and implement `__init__` and `forward`.
- **Key methods:**
  - `__init__(self)`: Define layers and parameters.
  - `forward(self, *input)`: Define computation.
  - `parameters(recurse=True)`: Iterator over learnable parameters.
  - `named_parameters(prefix='', recurse=True)`: Iterator over (name, parameter) pairs.
  - `children()`: Iterator over immediate submodules.
  - `named_children()`: Iterator over (name, submodule) pairs.
  - `modules()`: Iterator over all modules in the network.
  - `named_modules()`: Iterator over (name, module) pairs.
  - `train(mode=True)`: Set training mode.
  - `eval()`: Set evaluation mode.
  - `to(device)`: Move module to device.
  - `zero_grad(set_to_none=True)`: Clear gradients.
  - `state_dict()`: Return dictionary of parameters and buffers.
  - `load_state_dict(state_dict)`: Load parameters and buffers.
  - `apply(fn)`: Apply function to all submodules.
- **Example:**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
```

### `torch.nn.Parameter(data, requires_grad=True)`

- **Purpose:** A tensor that is automatically registered as a parameter when assigned to an `nn.Module` attribute.
- **Syntax:** `nn.Parameter(torch.randn(3, 4))`
- **Parameters:**
  - `data`: Tensor.
  - `requires_grad`: If `True`, gradients are computed. Default `True`.
- **Example:**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 5))
        self.bias = nn.Parameter(torch.zeros(5))
```

### `torch.nn.Sequential(*args)`

- **Purpose:** A container that passes input through a sequence of modules in order.
- **Syntax:** `nn.Sequential(module1, module2, ...)`
- **Parameters:** `*args`: Modules or ordered dict of named modules.
- **Example:**
```python
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
```

### `torch.nn.ModuleList(modules=None)`

- **Purpose:** Holds submodules in a list. Registers them properly (unlike a plain Python list).
- **Syntax:** `nn.ModuleList([module1, module2, ...])`
- **Parameters:** `modules`: Iterable of modules.
- **Example:**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### `torch.nn.ModuleDict(modules=None)`

- **Purpose:** Holds submodules in a dictionary. Registers them properly.
- **Syntax:** `nn.ModuleDict({'name1': module1, 'name2': module2})`
- **Example:**
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({
            'linear1': nn.Linear(10, 5),
            'linear2': nn.Linear(5, 1)
        })

    def forward(self, x):
        x = self.layers['linear1'](x)
        x = self.layers['linear2'](x)
        return x
```

### `torch.nn.functional` (commonly `F`)

- **Purpose:** Stateless functions for building neural networks.
- **Common functions:**
  - `F.relu(x)`, `F.sigmoid(x)`, `F.tanh(x)`
  - `F.softmax(x, dim)`, `F.log_softmax(x, dim)`
  - `F.linear(x, weight, bias)`
  - `F.conv2d(x, weight, bias, stride, padding)`
  - `F.max_pool2d(x, kernel_size, stride)`
  - `F.cross_entropy(logits, targets)`
  - `F.mse_loss(y_pred, y_true)`
  - `F.binary_cross_entropy(y_pred, y_true)`
  - `F.dropout(x, p, training)`
- **Example:**
```python
import torch.nn.functional as F

x = torch.randn(3, 5)
print(F.relu(x))
```

### `torch.nn.Linear`, `torch.nn.Conv2d`, `torch.nn.ReLU`, etc.

- **Purpose:** Pre-built layers and activations. These are `nn.Module` subclasses.
- **Example:**
```python
layer = nn.Linear(10, 5)
conv = nn.Conv2d(3, 16, kernel_size=3)
relu = nn.ReLU()
```

### `torch.nn.Module.apply(fn)`

- **Purpose:** Apply a function recursively to all submodules.
- **Syntax:** `model.apply(fn)`
- **Example:**
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

### `torch.nn.Module.register_forward_hook(hook)`

- **Purpose:** Register a hook to be called after the forward pass of a module.
- **Syntax:** `module.register_forward_hook(hook)`
- **Hook signature:** `hook(module, input, output)`
- **Example:**
```python
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__} output shape: {output.shape}")

model.fc.register_forward_hook(hook_fn)
```

### `torch.nn.Module.state_dict()` and `load_state_dict()`

- **Purpose:** Save and load model parameters and buffers.
- **Syntax:**
  - `state_dict = model.state_dict()`
  - `model.load_state_dict(state_dict)`
- **Example:**
```python
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

---

## 4. Code Examples

### Example 1: Custom Module with Parameters

```python
import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias

model = CustomLinear(10, 5)
x = torch.randn(3, 10)
y = model(x)
print("Output shape:", y.shape)  # (3, 5)

# List parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

**What it does:** Implements a custom linear layer using `nn.Parameter`.

**Important lines:**
- `nn.Parameter` automatically registers weights and bias.
- `x @ self.weight.T + self.bias` computes the linear transformation.

**Expected output:**
```
Output shape: torch.Size([3, 5])
weight: torch.Size([5, 10])
bias: torch.Size([5])
```

### Example 2: `nn.Sequential` vs `nn.ModuleList`

```python
import torch
import torch.nn as nn

# Sequential: fixed order
seq_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
x = torch.randn(4, 10)
print("Sequential output:", seq_model(x).shape)  # (4, 1)

# ModuleList: dynamic, used in loops
class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        self.final = nn.Linear(10, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.final(x)

dynamic_model = DynamicModel()
print("Dynamic output:", dynamic_model(x).shape)  # (4, 1)

# Parameters are registered
print("Dynamic parameters:", len(list(dynamic_model.parameters())))
```

**What it does:** Contrasts `nn.Sequential` (fixed order) with `nn.ModuleList` (dynamic loops).

**Important lines:**
- `nn.Sequential` applies modules in order.
- `nn.ModuleList` stores modules in a list and registers them.
- Both properly register parameters.

**Expected output:**
```
Sequential output: torch.Size([4, 1])
Dynamic output: torch.Size([4, 1])
Dynamic parameters: 8
```

### Example 3: Using `nn.functional` for Custom Forward

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CustomNet()
x = torch.randn(3, 10)
y = model(x)
print("Output shape:", y.shape)  # (3, 5)
print("Output sums to 1 per row:", y.exp().sum(dim=1))
```

**What it does:** Uses `F.relu`, `F.dropout`, and `F.log_softmax` in a custom forward.

**Important lines:**
- `self.training` is `True` in training mode, `False` in eval mode.
- `F.dropout` respects the `training` flag.
- `F.log_softmax` applies log-softmax.

**Expected output:**
```
Output shape: torch.Size([3, 5])
Output sums to 1 per row: tensor([1.0000, 1.0000, 1.0000], grad_fn=<SumBackward1>)
```

### Example 4: Parameter Initialization with `apply`

```python
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Apply initialization
model.apply(init_weights)

# Check one weight
print("First layer weight mean:", model[0].weight.mean().item())
print("First layer bias mean:", model[0].bias.mean().item())
```

**What it does:** Applies custom weight initialization to all layers.

**Important lines:**
- `model.apply(fn)` recursively applies `fn` to all submodules.
- `isinstance(m, nn.Linear)` checks module type.
- `nn.init.xavier_uniform_` initializes weights.

**Expected output:**
```
First layer weight mean: ~0.0
First layer bias mean: 0.0
```

### Example 5: Forward Hooks for Debugging

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Register a forward hook on the second layer
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__} output shape: {output.shape}")

model[2].register_forward_hook(hook_fn)

x = torch.randn(4, 10)
y = model(x)
```

**What it does:** Registers a forward hook to print the output shape of a specific layer.

**Important lines:**
- `register_forward_hook` attaches a function to a module.
- Hook is called after the forward pass of that module.

**Expected output:**
```
Linear output shape: torch.Size([4, 5])
```

### Example 6: Saving and Loading State Dict

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)

# Save
torch.save(model.state_dict(), 'linear.pth')

# Load into a new instance
model2 = nn.Linear(10, 5)
model2.load_state_dict(torch.load('linear.pth'))

# Verify weights are the same
print("Weights equal:", torch.equal(model.weight, model2.weight))
```

**What it does:** Saves and loads model parameters using `state_dict`.

**Important lines:**
- `state_dict()` returns a dictionary of parameters.
- `load_state_dict()` loads them into a model with the same architecture.

**Expected output:**
```
Weights equal: True
```

---

## 5. Important Parameters

### `nn.Module.__init__`

| Parameter | Type | Description |
|-----------|------|-------------|
| None | — | Must call `super().__init__()` first. |

### `nn.Parameter(data, requires_grad=True)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | Tensor | Parameter data. |
| `requires_grad` | bool | If `True`, gradients are computed. Default `True`. |

### `nn.Sequential(*args)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `*args` | Module or OrderedDict | Modules to be added in order. |

### `nn.ModuleList(modules=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `modules` | iterable of Modules | Modules to store. |

### `nn.ModuleDict(modules=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `modules` | dict of str -> Module | Modules to store. |

### `nn.Module.apply(fn)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | callable | Function to apply to each submodule. |

### `nn.Module.register_forward_hook(hook)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `hook` | callable | Function with signature `hook(module, input, output)`. |

### `nn.Module.state_dict(destination=None, prefix='', keep_vars=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `destination` | dict | Optional dict to store state. |
| `prefix` | str | Prefix for parameter names. |
| `keep_vars` | bool | If `True`, returns tensors instead of detached. |

### `nn.Module.load_state_dict(state_dict, strict=True)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_dict` | dict | State dict to load. |
| `strict` | bool | If `True`, keys must match exactly. Default `True`. |

---

## 6. Internal Working

### Parameter Registration

When you assign an `nn.Parameter` to an attribute of an `nn.Module`:

1. `nn.Module.__setattr__` is called.
2. If the value is an `nn.Parameter`, it is added to `self._parameters` dictionary.
3. If the value is an `nn.Module`, it is added to `self._modules` dictionary.
4. Otherwise, it is set as a regular attribute.

This automatic registration is why you must assign parameters as attributes (e.g., `self.weight = nn.Parameter(...)`) rather than storing them in a plain list.

### Module Registration

When you assign an `nn.Module` to an attribute:

1. It is added to `self._modules`.
2. Its parameters are included when calling `model.parameters()`.
3. Its modules are included when calling `model.modules()`.

### `__call__` vs `forward`

- `module(x)` invokes `__call__`.
- `__call__` runs forward hooks and then calls `forward(x)`.
- Always define `forward`, not `__call__`.

### `parameters()` and `named_parameters()`

- `parameters()` recursively yields all `nn.Parameter` objects.
- `named_parameters()` yields `(name, parameter)` tuples.
- Names are dotted paths (e.g., `fc1.weight`).

### `train()` and `eval()`

- `train(mode=True)` sets `self.training = True` for all submodules.
- `eval()` sets `self.training = False` for all submodules.
- Affects layers like `Dropout` and `BatchNorm`.

### `state_dict()`

- Returns a dictionary mapping parameter names to tensors.
- Includes parameters and buffers (e.g., running mean in BatchNorm).
- Does not include the model architecture.

### `apply(fn)`

- Recursively applies `fn` to all submodules (including self).
- Commonly used for weight initialization.

### Forward Hooks

- `register_forward_hook(hook)` adds a function to be called after `forward`.
- Hook signature: `hook(module, input, output)`.
- Useful for debugging, visualization, and feature extraction.

---

## 7. Common Mistakes

**Mistake:** Forgetting to call `super().__init__()` in `__init__`.

**Why it happens:** Without it, `nn.Module` is not properly initialized.

**Correct approach:** Always call `super().__init__()` first.

---

**Mistake:** Using a plain Python list to store modules.

**Why it happens:** Plain lists do not register modules, so their parameters are not included in `model.parameters()`.

**Correct approach:** Use `nn.ModuleList` or `nn.ModuleDict`.

---

**Mistake:** Assigning a tensor directly as an attribute instead of using `nn.Parameter`.

**Why it happens:** Plain tensors are not registered as parameters.

**Correct approach:** Wrap in `nn.Parameter`.

---

**Mistake:** Defining `__call__` instead of `forward`.

**Why it happens:** `__call__` bypasses hooks and other module logic.

**Correct approach:** Always define `forward`.

---

**Mistake:** Forgetting `model.train()` after validation.

**Why it happens:** `eval()` mode persists.

**Correct approach:** Call `model.train()` before resuming training.

---

**Mistake:** Using `F.dropout` without passing `training=self.training`.

**Why it happens:** `F.dropout` defaults to `training=True`, which is wrong during eval.

**Correct approach:** Pass `training=self.training` or use `nn.Dropout`.

---

**Mistake:** Modifying `state_dict` keys and expecting `load_state_dict` to work with `strict=True`.

**Why it happens:** Strict mode requires exact key match.

**Correct approach:** Use `strict=False` or adjust keys.

---

**Mistake:** Not moving model to device before training.

**Why it happens:** Model on CPU, data on GPU.

**Correct approach:** `model.to(device)`.

---

**Mistake:** Using `nn.Sequential` when dynamic branching is needed.

**Why it happens:** `nn.Sequential` only supports sequential flow.

**Correct approach:** Use `nn.Module` with custom `forward` for complex logic.

---

**Mistake:** Assuming `model.parameters()` returns a list.

**Why it happens:** It returns an iterator.

**Correct approach:** Convert to list if needed: `list(model.parameters())`.

---

## 8. Important Differences

| Concept | Difference |
|---------|-----------|
| `nn.Module` vs `nn.functional` | Modules hold state (parameters); functions are stateless. |
| `nn.Sequential` vs `nn.ModuleList` | `Sequential` applies modules in order; `ModuleList` stores them for dynamic use. |
| `nn.ModuleList` vs plain list | `ModuleList` registers modules; plain list does not. |
| `nn.Parameter` vs tensor | `Parameter` is automatically registered; tensor is not. |
| `model.train()` vs `model.eval()` | `train()` enables dropout/batch norm updates; `eval()` disables. |
| `model.parameters()` vs `model.named_parameters()` | `parameters()` yields tensors; `named_parameters()` yields (name, tensor) pairs. |
| `state_dict()` vs `model.parameters()` | `state_dict()` includes buffers and names; `parameters()` only learnable tensors. |
| `__call__` vs `forward` | `__call__` runs hooks; `forward` is the user-defined computation. |
| `apply(fn)` vs manual loop | `apply` recursively applies to all submodules. |
| `register_forward_hook` vs manual print | Hooks are called automatically during forward. |

---

## 9. Important Rules / Facts

- `torch.nn` is the package for neural network building blocks.
- `nn.Module` is the base class for all models and layers.
- Always call `super().__init__()` in `__init__`.
- Parameters must be `nn.Parameter` to be registered.
- Modules must be assigned as attributes to be registered.
- Use `nn.ModuleList` or `nn.ModuleDict` for lists/dicts of modules.
- `model.parameters()` yields all learnable parameters.
- `model.named_parameters()` yields names and parameters.
- `model.state_dict()` returns parameters and buffers.
- `model.train()` and `model.eval()` set modes recursively.
- `F.dropout` requires `training` flag; `nn.Dropout` handles it automatically.
- `model.apply(fn)` applies `fn` to all submodules.
- Forward hooks can be registered on any module.
- `nn.Sequential` is for simple feedforward stacks.
- Custom `forward` allows arbitrary computation.
- `nn.functional` provides stateless operations.
- Modules are stateful; functions are stateless.
- `state_dict` is the recommended way to save/load models.
- `load_state_dict` with `strict=True` requires exact key match.
- `model.to(device)` moves all parameters and buffers.

---

## 10. Practical Example

### Building a Configurable MLP with `nn.ModuleList` and `nn.Sequential`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Create model
model = ConfigurableMLP(
    input_dim=20,
    hidden_dims=[64, 32, 16],
    output_dim=5,
    dropout=0.3
)

# Print architecture
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Forward pass
x = torch.randn(8, 20)
y = model(x)
print("Output shape:", y.shape)  # (8, 5)

# Save and load
torch.save(model.state_dict(), 'mlp.pth')
model2 = ConfigurableMLP(20, [64, 32, 16], 5, 0.3)
model2.load_state_dict(torch.load('mlp.pth'))
print("Load successful:", torch.equal(model.net[0].weight, model2.net[0].weight))
```

**What it does:**
- Builds an MLP dynamically based on hidden dimensions.
- Uses `nn.Sequential` to stack layers.
- Counts parameters.
- Saves and loads state dict.

**Important lines:**
- `layers` list is populated with `nn.Linear`, `nn.ReLU`, `nn.Dropout`.
- `nn.Sequential(*layers)` creates the model.
- `sum(p.numel() for p in model.parameters())` counts total parameters.
- `state_dict()` and `load_state_dict()` for persistence.

**Expected output:**
```
ConfigurableMLP(
  (net): Sequential(
    (0): Linear(in_features=20, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=64, out_features=32, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.3, inplace=False)
    (6): Linear(in_features=32, out_features=16, bias=True)
    (7): ReLU()
    (8): Dropout(p=0.3, inplace=False)
    (9): Linear(in_features=16, out_features=5, bias=True)
  )
)
Total parameters: 3,381
Output shape: torch.Size([8, 5])
Load successful: True
```

---

## 11. Chapter Summary

- `torch.nn` provides building blocks for neural networks.
- `nn.Module` is the base class; subclass it for custom models.
- `nn.Parameter` registers tensors as learnable parameters.
- Submodules assigned as attributes are automatically registered.
- `nn.Sequential` is for simple feedforward stacks.
- `nn.ModuleList` and `nn.ModuleDict` are for dynamic collections.
- `nn.functional` provides stateless operations.
- `model.parameters()` and `model.named_parameters()` list learnable parameters.
- `model.state_dict()` and `load_state_dict()` save/load parameters.
- `model.train()` and `model.eval()` control layer behavior.
- Forward hooks enable debugging and feature extraction.
- `model.apply(fn)` applies initialization to all submodules.
- Common mistakes: forgetting `super().__init__()`, using plain lists, not using `nn.Parameter`, wrong dropout mode.
- Understanding `torch.nn` is essential for building complex architectures.

---

## 12. Important APIs to Remember

| API | Purpose |
|-----|---------|
| `nn.Module` | Base class for models |
| `nn.Parameter` | Register learnable parameter |
| `nn.Sequential` | Sequential container |
| `nn.ModuleList` | List of modules |
| `nn.ModuleDict` | Dictionary of modules |
| `nn.Linear`, `nn.Conv2d`, `nn.ReLU`, etc. | Pre-built layers |
| `nn.functional` (`F`) | Stateless functions |
| `model.parameters()` | Iterator of parameters |
| `model.named_parameters()` | Iterator of (name, param) |
| `model.state_dict()` | Dictionary of parameters/buffers |
| `model.load_state_dict()` | Load parameters |
| `model.train()`, `model.eval()` | Set modes |
| `model.to(device)` | Move model |
| `model.apply(fn)` | Apply function recursively |
| `module.register_forward_hook()` | Register forward hook |
| `model.zero_grad()` | Clear gradients |

---

## 13. Key Takeaways

1. `torch.nn` is the core package for neural network components.
2. `nn.Module` is the base class; always call `super().__init__()`.
3. Parameters must be `nn.Parameter` to be registered.
4. Submodules must be assigned as attributes to be registered.
5. Use `nn.ModuleList`/`nn.ModuleDict` for collections of modules.
6. `nn.Sequential` is for simple sequential models.
7. `nn.functional` provides stateless operations.
8. `model.parameters()` and `model.named_parameters()` list parameters.
9. `state_dict()` and `load_state_dict()` are used for saving/loading.
10. `model.train()` and `model.eval()` control dropout and batch norm.
11. `model.apply(fn)` is useful for weight initialization.
12. Forward hooks enable debugging and feature extraction.
13. Common mistakes: forgetting `super().__init__()`, using plain lists, not using `nn.Parameter`.
14. Dropout in functional form requires `training=self.training`.
15. Mastery of `torch.nn` enables building and organizing complex models.
