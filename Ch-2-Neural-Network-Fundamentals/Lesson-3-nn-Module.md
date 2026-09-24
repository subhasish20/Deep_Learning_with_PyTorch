# Part 2 — Neural Network Fundamentals
# Chapter 3 — `nn.Module`

**Prerequisite:** Part 2 — Chapter 1 — Neural Network Basics in PyTorch, Part 2 — Chapter 2 — `torch.nn`

---

## 1. Overview

- `nn.Module` is the fundamental building block for all neural network components in PyTorch.
- It provides automatic parameter registration, submodule management, device movement, mode switching, and state serialization.
- Understanding `nn.Module` internals: `_parameters`, `_modules`, `_buffers`, `_non_persistent_buffers_set`.
- Parameter vs buffer: what gets saved in `state_dict()` and what gets moved to device.
- Module composition: nesting modules, shared modules, weight tying.
- Forward and backward hooks: registering, executing, and use cases.
- `train()`/`eval()` propagation through the module tree.
- `state_dict()` and `load_state_dict()` mechanics.
- Advanced patterns: custom `__init__`, dynamic module creation, conditional forward.
- Common pitfalls when subclassing `nn.Module`.

---

## 2. Core Concepts

**`nn.Module`:** The base class for all neural network modules. It is a Python class that:
- Tracks parameters via `nn.Parameter`.
- Tracks submodules via `nn.Module` attributes.
- Provides `forward()` for computation.
- Supports hooks.
- Handles device movement and mode switching.
- Serializes parameters and buffers via `state_dict`.

**Parameter (`_parameters`):** A dictionary mapping attribute names to `nn.Parameter` objects. Automatically populated when `nn.Parameter` is assigned as an attribute.

**Buffer (`_buffers`):** A dictionary mapping attribute names to tensors that are part of the module's state but are not learnable (e.g., running mean/var in BatchNorm). Registered via `register_buffer()`.

**Non-persistent buffer:** A buffer that is not saved in `state_dict()` but is moved to device. Registered with `persistent=False`.

**Submodule (`_modules`):** A dictionary mapping attribute names to child `nn.Module` objects. Automatically populated when an `nn.Module` is assigned as an attribute.

**Module tree:** The hierarchical structure of modules. The root module contains submodules, which may contain submodules, etc.

**Leaf module:** A module with no submodules (e.g., `nn.Linear`, `nn.ReLU`).

**Container module:** A module that holds other modules (e.g., `nn.Sequential`, `nn.ModuleList`).

**Hook:** A function registered to execute at a specific point during forward or backward pass.

**Forward hook:** Called after `forward()` of a module.

**Forward pre-hook:** Called before `forward()` of a module.

**Backward hook:** Called during backward pass for a module's inputs/outputs.

**`train()` / `eval()`:** Set the `training` attribute recursively for all submodules. Affects layers like `Dropout` and `BatchNorm`.

**`state_dict()`:** A dictionary mapping parameter and buffer names to tensors. Used for saving/loading model state.

**`load_state_dict()`:** Loads a state dict into a module. `strict=True` requires exact key match.

**Weight tying:** Sharing the same `nn.Parameter` object between multiple modules (e.g., encoder and decoder in autoencoders).

**Shared module:** Using the same `nn.Module` instance in multiple places in the network.

---

## 3. Important PyTorch APIs

### `nn.Module.__init__()`

- **Purpose:** Initialize the module. Must call `super().__init__()` first.
- **Syntax:** `def __init__(self): super().__init__(); ...`
- **Example:**
```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
```

### `nn.Module.forward(*input)`

- **Purpose:** Define the computation performed at every call.
- **Syntax:** `def forward(self, x): ...`
- **Important:** Do not call `forward()` directly; call `module(x)` instead.
- **Example:**
```python
def forward(self, x):
    return self.linear(x)
```

### `nn.Module.parameters(recurse=True)`

- **Purpose:** Returns an iterator over module parameters.
- **Syntax:** `module.parameters(recurse=True)`
- **Parameters:**
  - `recurse` (bool): If `True`, includes parameters of submodules. Default `True`.
- **Example:**
```python
for p in model.parameters():
    print(p.shape)
```

### `nn.Module.named_parameters(prefix='', recurse=True)`

- **Purpose:** Returns an iterator over (name, parameter) pairs.
- **Syntax:** `module.named_parameters(prefix='', recurse=True)`
- **Example:**
```python
for name, p in model.named_parameters():
    print(name, p.shape)
```

### `nn.Module.buffers(recurse=True)`

- **Purpose:** Returns an iterator over module buffers.
- **Syntax:** `module.buffers(recurse=True)`
- **Example:**
```python
for b in model.buffers():
    print(b.shape)
```

### `nn.Module.named_buffers(prefix='', recurse=True)`

- **Purpose:** Returns an iterator over (name, buffer) pairs.
- **Example:**
```python
for name, b in model.named_buffers():
    print(name, b.shape)
```

### `nn.Module.children()`

- **Purpose:** Returns an iterator over immediate child modules.
- **Syntax:** `module.children()`
- **Example:**
```python
for child in model.children():
    print(child)
```

### `nn.Module.named_children()`

- **Purpose:** Returns an iterator over (name, child) pairs.
- **Example:**
```python
for name, child in model.named_children():
    print(name, child)
```

### `nn.Module.modules()`

- **Purpose:** Returns an iterator over all modules in the network (including self).
- **Syntax:** `module.modules()`
- **Example:**
```python
for m in model.modules():
    print(m.__class__.__name__)
```

### `nn.Module.named_modules(prefix='', remove_duplicate=True)`

- **Purpose:** Returns an iterator over (name, module) pairs.
- **Example:**
```python
for name, m in model.named_modules():
    print(name, m.__class__.__name__)
```

### `nn.Module.train(mode=True)`

- **Purpose:** Set the module and all submodules to training mode.
- **Syntax:** `module.train(mode=True)`
- **Parameters:**
  - `mode` (bool): If `True`, training mode; if `False`, evaluation mode.
- **Example:**
```python
model.train()
model.train(False)  # same as eval()
```

### `nn.Module.eval()`

- **Purpose:** Set the module and all submodules to evaluation mode.
- **Syntax:** `module.eval()`
- **Example:**
```python
model.eval()
```

### `nn.Module.to(*args, **kwargs)`

- **Purpose:** Move and/or cast the module's parameters and buffers.
- **Syntax:** `module.to(device)`, `module.to(dtype)`, `module.to(device, dtype)`
- **Example:**
```python
model.to('cuda')
model.to(torch.float16)
```

### `nn.Module.cpu()`

- **Purpose:** Move all parameters and buffers to CPU.
- **Syntax:** `module.cpu()`

### `nn.Module.cuda(device=None)`

- **Purpose:** Move all parameters and buffers to GPU.
- **Syntax:** `module.cuda(device=0)`

### `nn.Module.register_buffer(name, tensor, persistent=True)`

- **Purpose:** Register a tensor as a buffer.
- **Syntax:** `module.register_buffer('running_mean', torch.zeros(10))`
- **Parameters:**
  - `name` (str): Buffer name.
  - `tensor` (Tensor): Buffer tensor.
  - `persistent` (bool): If `True`, buffer is saved in `state_dict()`. Default `True`.
- **Example:**
```python
self.register_buffer('running_mean', torch.zeros(num_features))
```

### `nn.Module.register_parameter(name, param)`

- **Purpose:** Register a parameter.
- **Syntax:** `module.register_parameter('weight', nn.Parameter(torch.randn(3, 4)))`
- **Parameters:**
  - `name` (str): Parameter name.
  - `param` (nn.Parameter or None): Parameter object.
- **Example:**
```python
self.register_parameter('weight', nn.Parameter(torch.randn(3, 4)))
```

### `nn.Module.register_forward_hook(hook)`

- **Purpose:** Register a hook to be called after `forward()`.
- **Syntax:** `module.register_forward_hook(hook)`
- **Hook signature:** `hook(module, input, output)`
- **Return:** A handle with `remove()` method.
- **Example:**
```python
def hook_fn(module, input, output):
    print(output.shape)
handle = model.fc.register_forward_hook(hook_fn)
# Later: handle.remove()
```

### `nn.Module.register_forward_pre_hook(hook)`

- **Purpose:** Register a hook to be called before `forward()`.
- **Hook signature:** `hook(module, input)`
- **Example:**
```python
def pre_hook(module, input):
    print("Input shape:", input[0].shape)
model.fc.register_forward_pre_hook(pre_hook)
```

### `nn.Module.register_full_backward_hook(hook)`

- **Purpose:** Register a hook to be called during backward pass.
- **Hook signature:** `hook(module, grad_input, grad_output)`
- **Example:**
```python
def backward_hook(module, grad_input, grad_output):
    print("Grad output:", grad_output[0].shape)
model.fc.register_full_backward_hook(backward_hook)
```

### `nn.Module.state_dict(destination=None, prefix='', keep_vars=False)`

- **Purpose:** Return a dictionary of parameters and buffers.
- **Syntax:** `module.state_dict()`
- **Parameters:**
  - `destination` (dict): Optional dict to store state.
  - `prefix` (str): Prefix for names.
  - `keep_vars` (bool): If `True`, returns tensors; if `False`, returns detached tensors. Default `False`.
- **Example:**
```python
sd = model.state_dict()
print(sd.keys())
```

### `nn.Module.load_state_dict(state_dict, strict=True)`

- **Purpose:** Load parameters and buffers from a state dict.
- **Syntax:** `module.load_state_dict(state_dict, strict=True)`
- **Parameters:**
  - `state_dict` (dict): State dict to load.
  - `strict` (bool): If `True`, keys must match exactly. Default `True`.
- **Return:** `NamedTuple` with `missing_keys` and `unexpected_keys`.
- **Example:**
```python
model.load_state_dict(torch.load('model.pth'))
```

### `nn.Module.zero_grad(set_to_none=True)`

- **Purpose:** Clear gradients of all parameters.
- **Syntax:** `module.zero_grad(set_to_none=True)`
- **Parameters:**
  - `set_to_none` (bool): If `True`, sets `.grad` to `None` instead of zero. Default `True`.
- **Example:**
```python
model.zero_grad()
```

### `nn.Module.apply(fn)`

- **Purpose:** Apply `fn` recursively to all submodules (including self).
- **Syntax:** `module.apply(fn)`
- **Example:**
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
model.apply(init_weights)
```

### `nn.Module.requires_grad_(requires_grad=True)`

- **Purpose:** Set `requires_grad` for all parameters.
- **Syntax:** `module.requires_grad_(requires_grad=True)`
- **Example:**
```python
model.requires_grad_(False)  # freeze all parameters
```

---

## 4. Code Examples

### Example 1: Inspecting Module Internals

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.register_buffer('running_mean', torch.zeros(5))
        self.register_buffer('non_persistent', torch.zeros(5), persistent=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MyModel()

# Inspect _parameters
print("Parameters:")
for name, param in model._parameters.items():
    print(f"  {name}: {param.shape}")

# Inspect _modules
print("\nModules:")
for name, module in model._modules.items():
    print(f"  {name}: {module.__class__.__name__}")

# Inspect _buffers
print("\nBuffers:")
for name, buf in model._buffers.items():
    print(f"  {name}: {buf.shape}")

# Non-persistent buffers
print("\nNon-persistent buffers:", model._non_persistent_buffers_set)

# State dict
print("\nState dict keys:", list(model.state_dict().keys()))
```

**What it does:** Inspects internal dictionaries of `nn.Module`.

**Important lines:**
- `_parameters`, `_modules`, `_buffers` are internal dicts.
- `register_buffer` adds to `_buffers`.
- `persistent=False` adds to `_non_persistent_buffers_set`.

**Expected output:**
```
Parameters:
  fc1.weight: torch.Size([5, 10])
  fc1.bias: torch.Size([5])
  fc2.weight: torch.Size([1, 5])
  fc2.bias: torch.Size([1])

Modules:
  fc1: Linear
  fc2: Linear

Buffers:
  running_mean: torch.Size([5])
  non_persistent: torch.Size([5])

Non-persistent buffers: {'non_persistent'}

State dict keys: ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'running_mean']
```

### Example 2: Parameter vs Buffer

```python
import torch
import torch.nn as nn

class ModelWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4))
        self.register_buffer('running_mean', torch.zeros(4))
        self.register_buffer('non_persistent', torch.ones(4), persistent=False)

    def forward(self, x):
        return x @ self.weight.T + self.running_mean

model = ModelWithBuffer()

# Parameters
print("Parameters:", [name for name, _ in model.named_parameters()])

# Buffers
print("Buffers:", [name for name, _ in model.named_buffers()])

# State dict
print("State dict keys:", list(model.state_dict().keys()))

# Device movement
model.to('cpu')
print("Buffer device:", model.running_mean.device)
print("Non-persistent buffer device:", model.non_persistent.device)
```

**What it does:** Shows difference between parameters and buffers.

**Important lines:**
- Parameters are learnable; buffers are not.
- Non-persistent buffers are not saved in `state_dict` but are moved to device.

**Expected output:**
```
Parameters: ['weight']
Buffers: ['running_mean', 'non_persistent']
State dict keys: ['weight', 'running_mean']
Buffer device: cpu
Non-persistent buffer device: cpu
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

# Forward hook
def forward_hook(module, input, output):
    print(f"Forward hook: {module.__class__.__name__}, output shape: {output.shape}")

# Forward pre-hook
def forward_pre_hook(module, input):
    print(f"Forward pre-hook: {module.__class__.__name__}, input shape: {input[0].shape}")

# Register hooks
handle1 = model[0].register_forward_pre_hook(forward_pre_hook)
handle2 = model[0].register_forward_hook(forward_hook)
handle3 = model[2].register_forward_hook(forward_hook)

x = torch.randn(4, 10)
y = model(x)

# Remove hooks
handle1.remove()
handle2.remove()
handle3.remove()
```

**What it does:** Demonstrates forward pre-hooks and forward hooks.

**Important lines:**
- `register_forward_pre_hook` runs before `forward`.
- `register_forward_hook` runs after `forward`.
- Handles have `remove()` to unregister.

**Expected output:**
```
Forward pre-hook: Linear, input shape: torch.Size([4, 10])
Forward hook: Linear, output shape: torch.Size([4, 20])
Forward hook: Linear, output shape: torch.Size([4, 5])
```

### Example 4: Backward Hooks

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

def backward_hook(module, grad_input, grad_output):
    print(f"Backward hook: {module.__class__.__name__}")
    print(f"  grad_input shapes: {[g.shape if g is not None else None for g in grad_input]}")
    print(f"  grad_output shapes: {[g.shape if g is not None else None for g in grad_output]}")

model[0].register_full_backward_hook(backward_hook)
model[2].register_full_backward_hook(backward_hook)

x = torch.randn(4, 10)
y = model(x)
loss = y.sum()
loss.backward()
```

**What it does:** Registers backward hooks to inspect gradients.

**Important lines:**
- `register_full_backward_hook` is the modern API.
- Hook receives `grad_input` and `grad_output`.

**Expected output:**
```
Backward hook: Linear
  grad_input shapes: [torch.Size([4, 10]), torch.Size([5]), torch.Size([4, 5])]
  grad_output shapes: [torch.Size([4, 5])]
Backward hook: Linear
  grad_input shapes: [torch.Size([4, 5]), torch.Size([1]), torch.Size([4, 1])]
  grad_output shapes: [torch.Size([4, 1])]
```

### Example 5: Shared Modules and Weight Tying

```python
import torch
import torch.nn as nn

# Shared module
shared_linear = nn.Linear(10, 10)

class SharedModel(nn.Module):
    def __init__(self, shared_layer):
        super().__init__()
        self.layer1 = shared_layer
        self.layer2 = shared_layer  # same instance
        self.final = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.final(x)

model = SharedModel(shared_linear)

# Check parameter count
params = list(model.parameters())
print("Number of parameter tensors:", len(params))
for name, p in model.named_parameters():
    print(f"  {name}: {p.shape}")

# Weight tying
class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 10)
        self.decoder = nn.Linear(10, 100, bias=False)
        # Tie weights
        self.decoder.weight = self.embedding.weight

    def forward(self, x):
        emb = self.embedding(x)
        return self.decoder(emb)

tied = TiedModel()
print("\nTied model parameters:")
for name, p in tied.named_parameters():
    print(f"  {name}: {p.shape}")
```

**What it does:** Shows shared modules (same instance used twice) and weight tying (same parameter assigned to two layers).

**Important lines:**
- Shared module: `self.layer1 = shared_layer; self.layer2 = shared_layer`.
- Weight tying: `self.decoder.weight = self.embedding.weight`.
- Parameters are shared, reducing total count.

**Expected output:**
```
Number of parameter tensors: 4
  layer1.weight: torch.Size([10, 10])
  layer1.bias: torch.Size([10])
  final.weight: torch.Size([1, 10])
  final.bias: torch.Size([1])

Tied model parameters:
  embedding.weight: torch.Size([100, 10])
```

### Example 6: Custom `__init__` and Dynamic Module Creation

```python
import torch
import torch.nn as nn

class DynamicModel(nn.Module):
    def __init__(self, layer_dims, activation=nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                self.layers.append(activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = DynamicModel([10, 20, 15, 1])
print(model)

x = torch.randn(4, 10)
y = model(x)
print("Output shape:", y.shape)
```

**What it does:** Creates a dynamic MLP based on a list of dimensions.

**Important lines:**
- `nn.ModuleList` registers modules.
- Activation added between layers except after the last.

**Expected output:**
```
DynamicModel(
  (layers): ModuleList(
    (0): Linear(in_features=10, out_features=20, bias=True)
    (1): ReLU()
    (2): Linear(in_features=20, out_features=15, bias=True)
    (3): ReLU()
    (4): Linear(in_features=15, out_features=1, bias=True)
  )
)
Output shape: torch.Size([4, 1])
```

### Example 7: Train/Eval Mode Propagation

```python
import torch
import torch.nn as nn

class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(10)

    def forward(self, x):
        return self.bn(self.dropout(x))

class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(self.inner(x))

model = Outer()

print("Initial training mode:", model.training)
print("Inner training mode:", model.inner.training)

model.eval()
print("\nAfter eval():")
print("Outer training mode:", model.training)
print("Inner training mode:", model.inner.training)
print("Dropout training mode:", model.inner.dropout.training)
print("BatchNorm training mode:", model.inner.bn.training)

model.train()
print("\nAfter train():")
print("Outer training mode:", model.training)
print("Inner training mode:", model.inner.training)
```

**What it does:** Shows that `train()`/`eval()` propagate recursively.

**Important lines:**
- `model.eval()` sets `training=False` for all submodules.
- `model.train()` sets `training=True` for all submodules.

**Expected output:**
```
Initial training mode: True
Inner training mode: True

After eval():
Outer training mode: False
Inner training mode: False
Dropout training mode: False
BatchNorm training mode: False

After train():
Outer training mode: True
Inner training mode: True
```

---

## 5. Important Parameters

### `nn.Module.register_buffer(name, tensor, persistent=True)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Buffer name. |
| `tensor` | Tensor | Buffer tensor. |
| `persistent` | bool | If `True`, saved in `state_dict()`. Default `True`. |

### `nn.Module.register_parameter(name, param)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Parameter name. |
| `param` | nn.Parameter or None | Parameter object. |

### `nn.Module.register_forward_hook(hook)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `hook` | callable | Function with signature `hook(module, input, output)`. |

### `nn.Module.register_forward_pre_hook(hook)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `hook` | callable | Function with signature `hook(module, input)`. |

### `nn.Module.register_full_backward_hook(hook)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `hook` | callable | Function with signature `hook(module, grad_input, grad_output)`. |

### `nn.Module.state_dict(destination=None, prefix='', keep_vars=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `destination` | dict | Optional dict to store state. |
| `prefix` | str | Prefix for parameter names. |
| `keep_vars` | bool | If `True`, returns tensors instead of detached. Default `False`. |

### `nn.Module.load_state_dict(state_dict, strict=True)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_dict` | dict | State dict to load. |
| `strict` | bool | If `True`, keys must match exactly. Default `True`. |

### `nn.Module.zero_grad(set_to_none=True)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `set_to_none` | bool | If `True`, sets `.grad` to `None`. Default `True`. |

### `nn.Module.to(*args, **kwargs)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | torch.device | Target device. |
| `dtype` | torch.dtype | Target dtype. |
| `non_blocking` | bool | Asynchronous transfer. Default `False`. |

---

## 6. Internal Working

### Attribute Assignment and Registration

When you do `self.attr = value` inside an `nn.Module`:

1. `nn.Module.__setattr__` is called.
2. If `value` is an `nn.Parameter`, it is added to `self._parameters`.
3. If `value` is an `nn.Module`, it is added to `self._modules`.
4. Otherwise, it is stored in `self.__dict__`.

This is why assigning a plain tensor does not register it as a parameter.

### Module Tree Traversal

- `children()` iterates over immediate submodules (`_modules.values()`).
- `named_children()` yields `(name, module)` pairs.
- `modules()` recursively traverses all submodules.
- `named_modules()` yields `(name, module)` pairs with dotted names.
- `parameters()` recursively collects all `_parameters` from all modules.
- `named_parameters()` yields `(name, parameter)` pairs.

### `train()` / `eval()` Propagation

- `train(mode)` sets `self.training = mode` and recursively calls `train(mode)` on all submodules.
- `eval()` is equivalent to `train(False)`.
- Layers like `Dropout` and `BatchNorm` check `self.training` in their `forward`.

### `state_dict()` Mechanics

- `state_dict()` traverses the module tree.
- For each module, it adds its `_parameters` and `_buffers` to the dictionary.
- Names are prefixed with the module's name in the tree.
- Non-persistent buffers are excluded.
- The returned tensors are detached from the graph by default (`keep_vars=False`).

### `load_state_dict()` Mechanics

- `load_state_dict()` matches keys in the provided state dict to the module's parameters and buffers.
- If `strict=True`, it raises an error if keys are missing or unexpected.
- If `strict=False`, it loads matching keys and ignores the rest.
- It uses `copy_()` to load values in-place.

### Hooks Execution Order

For a forward pass:

1. Forward pre-hooks of the module.
2. `forward()` computation.
3. Forward hooks of the module.

For a backward pass:

1. Backward hooks of the module.

### Module `__call__`

- `module(x)` invokes `nn.Module.__call__`.
- `__call__` runs forward pre-hooks, calls `forward(x)`, and then runs forward hooks.
- Always define `forward`, not `__call__`.

### `apply(fn)`

- Recursively applies `fn` to self and all submodules.
- Commonly used for weight initialization.

### Device Movement

- `model.to(device)` moves all parameters and buffers to the device.
- It recursively calls `to(device)` on all submodules.
- Non-persistent buffers are also moved.

---

## 7. Common Mistakes

**Mistake:** Forgetting to call `super().__init__()`.

**Why it happens:** Without it, `nn.Module` is not initialized, and attribute assignment fails.

**Correct approach:** Always call `super().__init__()` first.

---

**Mistake:** Using a plain Python list to store modules.

**Why it happens:** Plain lists do not register modules, so their parameters are not included in `model.parameters()`.

**Correct approach:** Use `nn.ModuleList` or `nn.ModuleDict`.

---

**Mistake:** Assigning a tensor directly as an attribute instead of using `nn.Parameter`.

**Why it happens:** Plain tensors are not registered as parameters.

**Correct approach:** Wrap in `nn.Parameter` or use `register_parameter`.

---

**Mistake:** Forgetting to register a buffer.

**Why it happens:** Plain tensors assigned as attributes are not moved to device or saved in `state_dict`.

**Correct approach:** Use `register_buffer`.

---

**Mistake:** Modifying `state_dict` keys and expecting `load_state_dict` to work with `strict=True`.

**Why it happens:** Strict mode requires exact key match.

**Correct approach:** Use `strict=False` or adjust keys.

---

**Mistake:** Using `F.dropout` without passing `training=self.training`.

**Why it happens:** `F.dropout` defaults to `training=True`, which is wrong during eval.

**Correct approach:** Pass `training=self.training` or use `nn.Dropout`.

---

**Mistake:** Calling `forward()` directly instead of `module(x)`.

**Why it happens:** Direct `forward()` bypasses hooks and `__call__` logic.

**Correct approach:** Always call `module(x)`.

---

**Mistake:** Assuming `model.parameters()` returns a list.

**Why it happens:** It returns an iterator.

**Correct approach:** Convert to list if needed: `list(model.parameters())`.

---

**Mistake:** Not moving model to device before training.

**Why it happens:** Model on CPU, data on GPU.

**Correct approach:** `model.to(device)`.

---

**Mistake:** Using shared modules without understanding parameter sharing.

**Why it happens:** The same module instance used in multiple places shares parameters.

**Correct approach:** Be aware of parameter sharing; use separate instances if independent parameters are needed.

---

**Mistake:** Forgetting that `train()`/`eval()` propagate recursively.

**Why it happens:** Submodules also change mode.

**Correct approach:** Call `model.train()` or `model.eval()` on the root module.

---

## 8. Important Differences

| Concept | Difference |
|---------|-----------|
| `nn.Parameter` vs buffer | Parameter is learnable; buffer is not. Both are moved to device. |
| Persistent vs non-persistent buffer | Persistent is saved in `state_dict`; non-persistent is not. |
| `parameters()` vs `buffers()` | `parameters()` yields learnable tensors; `buffers()` yields non-learnable state. |
| `children()` vs `modules()` | `children()` yields immediate submodules; `modules()` yields all recursively. |
| `named_parameters()` vs `parameters()` | `named_parameters()` yields names; `parameters()` does not. |
| `train()` vs `eval()` | `train()` sets training mode; `eval()` sets evaluation mode. |
| `state_dict()` vs `parameters()` | `state_dict()` includes buffers and names; `parameters()` only learnable tensors. |
| `register_buffer` vs assigning tensor | `register_buffer` moves to device and saves in state dict; assigning tensor does not. |
| `register_parameter` vs `nn.Parameter` | Both register; `register_parameter` is explicit. |
| Shared module vs separate module | Shared module shares parameters; separate modules have independent parameters. |
| Weight tying vs shared module | Weight tying shares a single parameter; shared module shares the whole module. |
| `__call__` vs `forward` | `__call__` runs hooks; `forward` is user-defined computation. |
| `apply(fn)` vs manual loop | `apply` recursively applies to all submodules. |

---

## 9. Important Rules / Facts

- `nn.Module` is the base class for all neural network components.
- Always call `super().__init__()` in `__init__`.
- Parameters must be `nn.Parameter` to be registered.
- Buffers must be registered with `register_buffer`.
- Submodules must be assigned as attributes to be registered.
- Use `nn.ModuleList`/`nn.ModuleDict` for collections of modules.
- `model.parameters()` yields all learnable parameters.
- `model.buffers()` yields all non-learnable state tensors.
- `model.state_dict()` includes parameters and persistent buffers.
- `model.load_state_dict()` loads parameters and buffers.
- `model.train()` and `model.eval()` set modes recursively.
- `model.to(device)` moves parameters and buffers.
- `model.apply(fn)` applies `fn` to all submodules.
- Hooks can be registered for forward pre, forward, and backward.
- `register_full_backward_hook` is the modern API for backward hooks.
- Shared modules share parameters.
- Weight tying shares a single parameter between layers.
- Non-persistent buffers are moved to device but not saved.
- `zero_grad(set_to_none=True)` is the default and preferred.
- `module(x)` invokes `__call__`, which runs hooks and `forward`.

---

## 10. Practical Example

### Building a Residual Block with Custom `nn.Module`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class ResNetMini(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create model
model = ResNetMini(num_classes=10)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Forward pass
x = torch.randn(4, 3, 32, 32)
y = model(x)
print("Output shape:", y.shape)  # (4, 10)

# Inspect modules
print("\nModule names:")
for name, module in model.named_modules():
    if name:
        print(f"  {name}: {module.__class__.__name__}")

# Save and load
torch.save(model.state_dict(), 'resnet_mini.pth')
model2 = ResNetMini(num_classes=10)
model2.load_state_dict(torch.load('resnet_mini.pth'))
print("\nLoad successful:", torch.equal(model.fc.weight, model2.fc.weight))
```

**What it does:**
- Implements a residual block with shortcut connections.
- Builds a mini ResNet using `nn.Module` composition.
- Uses `nn.Sequential`, `nn.Conv2d`, `nn.BatchNorm2d`, and `F.relu`.
- Demonstrates parameter counting, state dict save/load, and module inspection.

**Important lines:**
- `ResidualBlock` uses a shortcut connection with `nn.Sequential`.
- `_make_layer` creates multiple residual blocks.
- `AdaptiveAvgPool2d(1)` reduces spatial dimensions to 1×1.
- `state_dict()` and `load_state_dict()` for persistence.

**Expected output:**
```
ResNetMini(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer1): Sequential(
    (0): ResidualBlock(...)
    (1): ResidualBlock(...)
  )
  ...
  (fc): Linear(in_features=64, out_features=10, bias=True)
)

Total parameters: 174,474
Output shape: torch.Size([4, 10])
...
Load successful: True
```

---

## 11. Chapter Summary

- `nn.Module` is the base class for all neural network components.
- It manages parameters, buffers, submodules, device placement, and mode switching.
- `nn.Parameter` registers learnable tensors; `register_buffer` registers non-learnable state.
- Submodules are registered when assigned as attributes.
- `nn.ModuleList` and `nn.ModuleDict` are containers for dynamic collections.
- `parameters()`, `buffers()`, `children()`, `modules()` traverse the module tree.
- `train()`/`eval()` set modes recursively.
- `state_dict()` saves parameters and persistent buffers.
- `load_state_dict()` loads them back.
- Hooks enable debugging and feature extraction.
- `apply(fn)` applies initialization recursively.
- Shared modules and weight tying reduce parameter count.
- Common mistakes: forgetting `super().__init__()`, using plain lists, not using `nn.Parameter`, wrong dropout mode.
- Understanding `nn.Module` internals is essential for building complex architectures.

---

## 12. Important APIs to Remember

| API | Purpose |
|-----|---------|
| `nn.Module` | Base class for models |
| `nn.Parameter` | Register learnable parameter |
| `register_buffer` | Register non-learnable buffer |
| `register_parameter` | Explicitly register parameter |
| `parameters()` | Iterator of parameters |
| `named_parameters()` | Iterator of (name, param) |
| `buffers()` | Iterator of buffers |
| `named_buffers()` | Iterator of (name, buffer) |
| `children()` | Iterator of immediate submodules |
| `named_children()` | Iterator of (name, child) |
| `modules()` | Iterator of all modules |
| `named_modules()` | Iterator of (name, module) |
| `train()`, `eval()` | Set modes |
| `to()`, `cpu()`, `cuda()` | Move device |
| `state_dict()` | Dictionary of parameters/buffers |
| `load_state_dict()` | Load parameters/buffers |
| `zero_grad()` | Clear gradients |
| `apply(fn)` | Apply function recursively |
| `register_forward_hook` | Forward hook |
| `register_forward_pre_hook` | Forward pre-hook |
| `register_full_backward_hook` | Backward hook |
| `requires_grad_()` | Set requires_grad |

---

## 13. Key Takeaways

1. `nn.Module` is the foundation for all neural network components.
2. Always call `super().__init__()` in `__init__`.
3. Parameters must be `nn.Parameter`; buffers must be registered with `register_buffer`.
4. Submodules must be assigned as attributes to be registered.
5. Use `nn.ModuleList`/`nn.ModuleDict` for collections of modules.
6. `parameters()` and `buffers()` list learnable and non-learnable state.
7. `state_dict()` saves parameters and persistent buffers.
8. `load_state_dict()` loads them back with strict key matching.
9. `train()`/`eval()` set modes recursively for all submodules.
10. `model.to(device)` moves parameters and buffers.
11. Hooks enable debugging and feature extraction.
12. `apply(fn)` is useful for weight initialization.
13. Shared modules and weight tying reduce parameter count.
14. Common mistakes: forgetting `super().__init__()`, using plain lists, not using `nn.Parameter`.
15. Understanding `nn.Module` internals is essential for building and organizing complex models.
