# Part 2 — Neural Network Fundamentals
# Chapter 4 — Layers

**Prerequisite:** Part 2 — Chapter 1 — Neural Network Basics in PyTorch, Part 2 — Chapter 2 — `torch.nn`, Part 2 — Chapter 3 — `nn.Module`

---

## 1. Overview

- Layers are the parameterized building blocks of neural networks.
- PyTorch provides layers for fully connected, convolutional, recurrent, normalization, pooling, dropout, embedding, and attention operations.
- Every layer is an `nn.Module` subclass with registered parameters and buffers.
- Layer configuration: input/output dimensions, kernel sizes, strides, padding, dilation, groups.
- Weight initialization schemes and their defaults.
- Difference between layers (stateful modules) and functional equivalents (`torch.nn.functional`).
- Layer composition: stacking, nesting, and sharing.
- Common layer categories: linear, convolutional, pooling, normalization, dropout, recurrent, embedding, attention.
- Understanding parameter shapes for each layer type.
- Practical layer selection for different tasks.

---

## 2. Core Concepts

**Layer:** A parameterized transformation applied to an input tensor. In PyTorch, layers are `nn.Module` subclasses that hold parameters (weights and biases) and optionally buffers.

**Fully Connected Layer (`nn.Linear`):** Applies `y = xW^T + b`. Each input feature connects to every output feature.

**Convolutional Layer (`nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`):** Applies a convolution operation using learnable filters. Preserves spatial structure for grid-like data.

**Transposed Convolution (`nn.ConvTranspose1d/2d/3d`):** Learns upsampling. Used in decoders and generative models.

**Pooling Layer (`nn.MaxPool2d`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d`):** Reduces spatial dimensions. No learnable parameters.

**Normalization Layer (`nn.BatchNorm1d/2d/3d`, `nn.LayerNorm`, `nn.GroupNorm`, `nn.InstanceNorm2d`):** Stabilizes training by normalizing activations.

**Dropout (`nn.Dropout`, `nn.Dropout2d`):** Randomly zeroes elements during training to prevent overfitting.

**Embedding (`nn.Embedding`):** Maps discrete indices to dense vectors. Used for words, tokens, categories.

**Recurrent Layer (`nn.RNN`, `nn.LSTM`, `nn.GRU`):** Processes sequences with hidden state.

**Attention Layer (`nn.MultiheadAttention`):** Applies scaled dot-product attention with multiple heads.

**Padding Layer (`nn.ZeroPad2d`, `nn.ReflectionPad2d`):** Adds padding to inputs.

**Upsampling Layer (`nn.Upsample`):** Resizes inputs via interpolation.

**Weight Initialization:** The scheme used to set initial parameter values (e.g., Kaiming, Xavier).

**In-place Operation:** Some layers support `inplace=True` to modify input directly, saving memory.

**Groups:** In convolutions, divides input/output channels into groups, reducing parameters (used in depthwise/grouped convolutions).

**Dilation:** Expands the convolution kernel by inserting spaces, increasing receptive field without increasing parameters.

**Stride:** Step size of the convolution or pooling kernel.

**Padding:** Border added to input before convolution/pooling to control output size.

**Receptive Field:** The region of the input that affects a particular output element.

---

## 3. Important PyTorch APIs

### `torch.nn.Linear(in_features, out_features, bias=True)`

- **Purpose:** Fully connected layer: `y = xW^T + b`.
- **Parameters:**
  - `in_features` (int): Input feature size.
  - `out_features` (int): Output feature size.
  - `bias` (bool): If `True`, adds bias. Default `True`.
- **Parameter shapes:**
  - `weight`: `(out_features, in_features)`
  - `bias`: `(out_features,)` if `bias=True`
- **Example:**
```python
layer = nn.Linear(128, 64)
x = torch.randn(32, 128)
y = layer(x)  # (32, 64)
```

### `torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')`

- **Purpose:** 2D convolution over `(N, C, H, W)` input.
- **Parameters:**
  - `in_channels` (int): Number of input channels.
  - `out_channels` (int): Number of output channels (filters).
  - `kernel_size` (int or tuple): Size of convolving kernel.
  - `stride` (int or tuple): Stride. Default `1`.
  - `padding` (int or tuple): Zero-padding. Default `0`.
  - `dilation` (int or tuple): Spacing between kernel elements. Default `1`.
  - `groups` (int): Grouped convolution. Default `1`.
  - `bias` (bool): If `True`, adds bias. Default `True`.
  - `padding_mode` (str): `'zeros'`, `'reflect'`, `'replicate'`, `'circular'`.
- **Parameter shapes:**
  - `weight`: `(out_channels, in_channels // groups, kH, kW)`
  - `bias`: `(out_channels,)` if `bias=True`
- **Output shape (per spatial dim):**
  - `H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)`
- **Example:**
```python
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
x = torch.randn(8, 3, 32, 32)
y = conv(x)  # (8, 16, 32, 32)
```

### `torch.nn.Conv1d` and `torch.nn.Conv3d`

- Same API as `Conv2d` but for 1D and 3D inputs.
- `Conv1d` input: `(N, C, L)`. `Conv3d` input: `(N, C, D, H, W)`.

### `torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)`

- **Purpose:** Transposed convolution (learned upsampling).
- **Extra Parameter:** `output_padding` (int or tuple): Additional size to output.
- **Example:**
```python
deconv = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
x = torch.randn(8, 16, 16, 16)
y = deconv(x)  # (8, 3, 32, 32)
```

### `torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`

- **Purpose:** Max pooling over spatial dimensions.
- **Parameters:**
  - `kernel_size` (int or tuple): Pooling window size.
  - `stride` (int or tuple): Stride. Defaults to `kernel_size`.
  - `padding` (int or tuple): Implicit zero-padding. Default `0`.
  - `return_indices` (bool): If `True`, returns argmax indices. Default `False`.
- **Example:**
```python
pool = nn.MaxPool2d(2)
x = torch.randn(8, 16, 32, 32)
y = pool(x)  # (8, 16, 16, 16)
```

### `torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)`

- **Purpose:** Average pooling.

### `torch.nn.AdaptiveAvgPool2d(output_size)`

- **Purpose:** Pools to a fixed output size regardless of input size.
- **Parameters:**
  - `output_size` (int or tuple): Target `(H, W)`. Use `1` for global pooling.
- **Example:**
```python
gap = nn.AdaptiveAvgPool2d(1)
x = torch.randn(8, 64, 7, 7)
y = gap(x)  # (8, 64, 1, 1)
```

### `torch.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)`

- **Purpose:** Batch normalization over channels for 2D inputs.
- **Parameters:**
  - `num_features` (int): Number of channels.
  - `eps` (float): Small constant for numerical stability. Default `1e-5`.
  - `momentum` (float): Running average momentum. Default `0.1`.
  - `affine` (bool): If `True`, learnable `weight` and `bias`. Default `True`.
  - `track_running_stats` (bool): If `True`, tracks running mean/var. Default `True`.
- **Parameter shapes:**
  - `weight`: `(num_features,)` if `affine=True`
  - `bias`: `(num_features,)` if `affine=True`
- **Buffer shapes:**
  - `running_mean`: `(num_features,)`
  - `running_var`: `(num_features,)`
  - `num_batches_tracked`: scalar
- **Example:**
```python
bn = nn.BatchNorm2d(16)
x = torch.randn(8, 16, 32, 32)
y = bn(x)  # (8, 16, 32, 32)
```

### `torch.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`

- **Purpose:** Layer normalization over the last dimensions.
- **Parameters:**
  - `normalized_shape` (int or list): Shape of last dimensions to normalize.
  - `eps` (float): Default `1e-5`.
  - `elementwise_affine` (bool): If `True`, learnable `weight` and `bias`. Default `True`.
- **Example:**
```python
ln = nn.LayerNorm(128)
x = torch.randn(32, 128)
y = ln(x)  # (32, 128)
```

### `torch.nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)`

- **Purpose:** Group normalization. Divides channels into groups and normalizes within each group.
- **Example:**
```python
gn = nn.GroupNorm(4, 16)
x = torch.randn(8, 16, 32, 32)
y = gn(x)  # (8, 16, 32, 32)
```

### `torch.nn.InstanceNorm2d(num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False)`

- **Purpose:** Instance normalization. Normalizes each sample and channel independently.

### `torch.nn.Dropout(p=0.5, inplace=False)`

- **Purpose:** Randomly zeroes elements with probability `p` during training.
- **Parameters:**
  - `p` (float): Probability of zeroing. Default `0.5`.
  - `inplace` (bool): If `True`, modifies input. Default `False`.
- **Example:**
```python
dropout = nn.Dropout(0.5)
x = torch.randn(4, 10)
y = dropout(x)
```

### `torch.nn.Dropout2d(p=0.5, inplace=False)`

- **Purpose:** Randomly zeroes entire channels (for 2D inputs).

### `torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)`

- **Purpose:** Lookup table mapping integer indices to dense vectors.
- **Parameters:**
  - `num_embeddings` (int): Size of vocabulary.
  - `embedding_dim` (int): Dimension of embedding vectors.
  - `padding_idx` (int, optional): Index whose embedding is zero and not updated.
  - `max_norm` (float, optional): Renormalizes embeddings to max norm.
  - `scale_grad_by_freq` (bool): Scales gradients by inverse frequency.
  - `sparse` (bool): If `True`, gradients are sparse.
- **Parameter shape:**
  - `weight`: `(num_embeddings, embedding_dim)`
- **Example:**
```python
emb = nn.Embedding(1000, 128)
indices = torch.tensor([[1, 5, 10], [2, 8, 12]])
vectors = emb(indices)  # (2, 3, 128)
```

### `torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False)`

- **Purpose:** Long Short-Term Memory recurrent layer.
- **Parameters:**
  - `input_size` (int): Feature size of input.
  - `hidden_size` (int): Feature size of hidden state.
  - `num_layers` (int): Number of stacked LSTM layers.
  - `batch_first` (bool): If `True`, input shape is `(batch, seq, feature)`. Default `False`.
  - `dropout` (float): Dropout between layers. Default `0`.
  - `bidirectional` (bool): If `True`, bidirectional LSTM. Default `False`.
- **Example:**
```python
lstm = nn.LSTM(10, 20, batch_first=True)
x = torch.randn(4, 5, 10)  # (batch, seq, input)
out, (h, c) = lstm(x)
```

### `torch.nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False)`

- **Purpose:** Gated Recurrent Unit.

### `torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False)`

- **Purpose:** Vanilla RNN.

### `torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False)`

- **Purpose:** Multi-head scaled dot-product attention.
- **Parameters:**
  - `embed_dim` (int): Model dimension.
  - `num_heads` (int): Number of attention heads.
  - `dropout` (float): Attention dropout. Default `0`.
  - `bias` (bool): If `True`, adds bias. Default `True`.
  - `batch_first` (bool): If `True`, input shape is `(batch, seq, feature)`. Default `False`.
- **Example:**
```python
mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
q = torch.randn(4, 10, 64)
k = torch.randn(4, 10, 64)
v = torch.randn(4, 10, 64)
attn_out, attn_weights = mha(q, k, v)
```

### `torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)`

- **Purpose:** Upsample input via interpolation.

### `torch.nn.ZeroPad2d(padding)`

- **Purpose:** Pad with zeros.

### `torch.nn.Flatten(start_dim=1, end_dim=-1)`

- **Purpose:** Flatten a contiguous range of dimensions.

### Weight Initialization Functions

- `nn.init.xavier_uniform_(tensor)`
- `nn.init.xavier_normal_(tensor)`
- `nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')`
- `nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')`
- `nn.init.zeros_(tensor)`
- `nn.init.ones_(tensor)`
- `nn.init.constant_(tensor, val)`
- `nn.init.normal_(tensor, mean=0, std=1)`
- `nn.init.uniform_(tensor, a=0, b=1)`

---

## 4. Code Examples

### Example 1: Linear Layer

```python
import torch
import torch.nn as nn

layer = nn.Linear(10, 5)
x = torch.randn(3, 10)
y = layer(x)

print("Input shape:", x.shape)   # (3, 10)
print("Output shape:", y.shape)  # (3, 5)
print("Weight shape:", layer.weight.shape)  # (5, 10)
print("Bias shape:", layer.bias.shape)      # (5,)

# Access parameters
print("Parameters:", list(layer.parameters()))
```

**What it does:** Creates a linear layer and inspects parameter shapes.

**Important lines:**
- `nn.Linear(10, 5)` maps 10 inputs to 5 outputs.
- `weight` shape is `(out_features, in_features)`.

**Expected output:**
```
Input shape: torch.Size([3, 10])
Output shape: torch.Size([3, 5])
Weight shape: torch.Size([5, 10])
Bias shape: torch.Size([5])
```

### Example 2: Convolutional Layer

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
x = torch.randn(8, 3, 32, 32)
y = conv(x)

print("Input shape:", x.shape)   # (8, 3, 32, 32)
print("Output shape:", y.shape)  # (8, 16, 32, 32)
print("Weight shape:", conv.weight.shape)  # (16, 3, 3, 3)
print("Bias shape:", conv.bias.shape)      # (16,)

# With stride 2
conv2 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
y2 = conv2(x)
print("Stride 2 output shape:", y2.shape)  # (8, 16, 16, 16)
```

**What it does:** Demonstrates 2D convolution with different strides.

**Important lines:**
- `weight` shape is `(out_channels, in_channels, kH, kW)`.
- Stride 2 halves the spatial dimensions (with padding 1).

**Expected output:**
```
Input shape: torch.Size([8, 3, 32, 32])
Output shape: torch.Size([8, 16, 32, 32])
Weight shape: torch.Size([16, 3, 3, 3])
Bias shape: torch.Size([16])
Stride 2 output shape: torch.Size([8, 16, 16, 16])
```

### Example 3: Pooling Layers

```python
import torch
import torch.nn as nn

x = torch.randn(8, 16, 32, 32)

# Max pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
y1 = maxpool(x)
print("MaxPool2d(2):", y1.shape)  # (8, 16, 16, 16)

# Average pooling
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
y2 = avgpool(x)
print("AvgPool2d(2):", y2.shape)  # (8, 16, 16, 16)

# Adaptive average pooling
gap = nn.AdaptiveAvgPool2d(1)
y3 = gap(x)
print("AdaptiveAvgPool2d(1):", y3.shape)  # (8, 16, 1, 1)

# Adaptive pooling to 7x7
adapt = nn.AdaptiveAvgPool2d((7, 7))
y4 = adapt(x)
print("AdaptiveAvgPool2d((7,7)):", y4.shape)  # (8, 16, 7, 7)
```

**What it does:** Demonstrates max, average, and adaptive pooling.

**Important lines:**
- `MaxPool2d(2)` reduces spatial dims by half.
- `AdaptiveAvgPool2d(1)` produces global average pooling.
- Adaptive pooling works with any input size.

**Expected output:** As commented.

### Example 4: Normalization Layers

```python
import torch
import torch.nn as nn

x = torch.randn(8, 16, 32, 32)

# BatchNorm2d
bn = nn.BatchNorm2d(16)
y1 = bn(x)
print("BatchNorm2d output:", y1.shape)  # (8, 16, 32, 32)
print("BN weight shape:", bn.weight.shape)  # (16,)
print("BN running_mean shape:", bn.running_mean.shape)  # (16,)

# LayerNorm on (batch, features)
x2 = torch.randn(8, 128)
ln = nn.LayerNorm(128)
y2 = ln(x2)
print("LayerNorm output:", y2.shape)  # (8, 128)

# GroupNorm
gn = nn.GroupNorm(num_groups=4, num_channels=16)
y3 = gn(x)
print("GroupNorm output:", y3.shape)  # (8, 16, 32, 32)

# InstanceNorm2d
inorm = nn.InstanceNorm2d(16)
y4 = inorm(x)
print("InstanceNorm2d output:", y4.shape)  # (8, 16, 32, 32)
```

**What it does:** Demonstrates different normalization layers.

**Important lines:**
- `BatchNorm2d` has learnable `weight`/`bias` and running buffers.
- `LayerNorm` normalizes over the last dimension.
- `GroupNorm` divides channels into groups.
- `InstanceNorm2d` normalizes per sample per channel.

**Expected output:** As commented.

### Example 5: Dropout

```python
import torch
import torch.nn as nn

x = torch.ones(4, 10)

# Dropout in training mode
dropout = nn.Dropout(p=0.5)
dropout.train()
y_train = dropout(x)
print("Training mode:\n", y_train)
print("Non-zero count:", (y_train != 0).sum().item())

# Dropout in eval mode
dropout.eval()
y_eval = dropout(x)
print("Eval mode:\n", y_eval)
print("Non-zero count:", (y_eval != 0).sum().item())

# Dropout2d zeroes entire channels
x2 = torch.ones(2, 4, 3, 3)
dropout2d = nn.Dropout2d(p=0.5)
dropout2d.train()
y2 = dropout2d(x2)
print("Dropout2d output shape:", y2.shape)
```

**What it does:** Shows dropout behavior in training vs evaluation.

**Important lines:**
- In training mode, dropout randomly zeroes elements and scales survivors by `1/(1-p)`.
- In eval mode, dropout is identity.

**Expected output:**
```
Training mode:
 tensor([[2., 0., 0., 2., 2., 0., 2., 0., 0., 2.], ...])
Non-zero count: ~20
Eval mode:
 tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], ...])
Non-zero count: 40
Dropout2d output shape: torch.Size([2, 4, 3, 3])
```

### Example 6: Embedding Layer

```python
import torch
import torch.nn as nn

emb = nn.Embedding(num_embeddings=100, embedding_dim=8)

# Integer indices
indices = torch.tensor([[1, 5, 10], [2, 8, 12]])
vectors = emb(indices)
print("Indices shape:", indices.shape)    # (2, 3)
print("Vectors shape:", vectors.shape)    # (2, 3, 8)
print("Weight shape:", emb.weight.shape)  # (100, 8)

# With padding_idx
emb_pad = nn.Embedding(100, 8, padding_idx=0)
vectors_pad = emb_pad(torch.tensor([0, 1, 2]))
print("Padding index vector:", vectors_pad[0])  # zeros
```

**What it does:** Demonstrates embedding lookup.

**Important lines:**
- `nn.Embedding(100, 8)` maps 100 indices to 8-dim vectors.
- `padding_idx=0` ensures index 0 maps to zero vector and receives no gradient.

**Expected output:**
```
Indices shape: torch.Size([2, 3])
Vectors shape: torch.Size([2, 3, 8])
Weight shape: torch.Size([100, 8])
Padding index vector: tensor([0., 0., 0., 0., 0., 0., 0., 0.], grad_fn=<SelectBackward0>)
```

### Example 7: Recurrent Layers

```python
import torch
import torch.nn as nn

# LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True, dropout=0.2)
x = torch.randn(4, 5, 10)  # (batch, seq, input)
out, (h, c) = lstm(x)
print("LSTM output shape:", out.shape)  # (4, 5, 20)
print("Hidden state shape:", h.shape)   # (2, 4, 20)
print("Cell state shape:", c.shape)     # (2, 4, 20)

# GRU
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
out_g, h_g = gru(x)
print("GRU output shape:", out_g.shape)  # (4, 5, 20)

# Vanilla RNN
rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)
out_r, h_r = rnn(x)
print("RNN output shape:", out_r.shape)  # (4, 5, 20)
```

**What it does:** Demonstrates LSTM, GRU, and RNN layers.

**Important lines:**
- `batch_first=True` expects input `(batch, seq, feature)`.
- LSTM returns output, hidden state, and cell state.
- GRU and RNN return output and hidden state.

**Expected output:** As commented.

### Example 8: Multihead Attention

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

# Query, key, value
q = torch.randn(4, 10, 64)
k = torch.randn(4, 10, 64)
v = torch.randn(4, 10, 64)

attn_out, attn_weights = mha(q, k, v)
print("Attention output shape:", attn_out.shape)      # (4, 10, 64)
print("Attention weights shape:", attn_weights.shape) # (4, 10, 10)

# Inspect parameters
for name, param in mha.named_parameters():
    print(f"  {name}: {param.shape}")
```

**What it does:** Applies multi-head attention.

**Important lines:**
- `embed_dim` must be divisible by `num_heads`.
- `batch_first=True` expects `(batch, seq, feature)`.
- Returns attention output and attention weights.

**Expected output:**
```
Attention output shape: torch.Size([4, 10, 64])
Attention weights shape: torch.Size([4, 10, 10])
  in_proj_weight: torch.Size([192, 64])
  in_proj_bias: torch.Size([192])
  out_proj.weight: torch.Size([64, 64])
  out_proj.bias: torch.Size([64])
```

### Example 9: Weight Initialization

```python
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16*32*32, 10)
)

model.apply(init_weights)

# Check initialized weights
print("Conv weight std:", model[0].weight.std().item())
print("Linear weight mean:", model[3].weight.mean().item())
print("Linear bias mean:", model[3].bias.mean().item())
```

**What it does:** Applies Xavier and Kaiming initialization.

**Important lines:**
- `nn.init.xavier_uniform_` for linear layers.
- `nn.init.kaiming_normal_` for conv layers with ReLU.
- `model.apply(fn)` recursively applies initialization.

**Expected output:**
```
Conv weight std: ~0.5
Linear weight mean: ~0.0
Linear bias mean: 0.0
```

### Example 10: Custom Layer with Parameter

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomScaleLayer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.weight + self.bias

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.scale = CustomScaleLayer(5)

    def forward(self, x):
        x = self.fc(x)
        x = self.scale(x)
        return x

model = Net()
x = torch.randn(3, 10)
y = model(x)
print("Output shape:", y.shape)  # (3, 5)

for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

**What it does:** Defines a custom layer with learnable scale and bias.

**Important lines:**
- `nn.Parameter` registers learnable tensors.
- Custom layer can be composed with other layers.

**Expected output:**
```
Output shape: torch.Size([3, 5])
  fc.weight: torch.Size([5, 10])
  fc.bias: torch.Size([5])
  scale.weight: torch.Size([5])
  scale.bias: torch.Size([5])
```

---

## 5. Important Parameters

### `nn.Conv2d`

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_channels` | int | Input channels. |
| `out_channels` | int | Output channels. |
| `kernel_size` | int or tuple | Kernel size. |
| `stride` | int or tuple | Stride. Default `1`. |
| `padding` | int or tuple | Padding. Default `0`. |
| `dilation` | int or tuple | Dilation. Default `1`. |
| `groups` | int | Grouped convolution. Default `1`. |
| `bias` | bool | Add bias. Default `True`. |
| `padding_mode` | str | `'zeros'`, `'reflect'`, `'replicate'`, `'circular'`. |

### `nn.BatchNorm2d`

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_features` | int | Number of channels. |
| `eps` | float | Numerical stability. Default `1e-5`. |
| `momentum` | float | Running average momentum. Default `0.1`. |
| `affine` | bool | Learnable affine params. Default `True`. |
| `track_running_stats` | bool | Track running mean/var. Default `True`. |

### `nn.LayerNorm`

| Parameter | Type | Description |
|-----------|------|-------------|
| `normalized_shape` | int or list | Shape of last dimensions. |
| `eps` | float | Default `1e-5`. |
| `elementwise_affine` | bool | Learnable affine params. Default `True`. |

### `nn.Dropout`

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | float | Probability of zeroing. Default `0.5`. |
| `inplace` | bool | Modify input in-place. Default `False`. |

### `nn.Embedding`

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_embeddings` | int | Vocabulary size. |
| `embedding_dim` | int | Embedding dimension. |
| `padding_idx` | int | Index for padding (zero vector, no grad). |
| `max_norm` | float | Max norm for renormalization. |
| `scale_grad_by_freq` | bool | Scale gradients by frequency. |
| `sparse` | bool | Sparse gradients. |

### `nn.LSTM` / `nn.GRU` / `nn.RNN`

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_size` | int | Input feature size. |
| `hidden_size` | int | Hidden state size. |
| `num_layers` | int | Number of stacked layers. Default `1`. |
| `bias` | bool | Add bias. Default `True`. |
| `batch_first` | bool | If `True`, batch is first dim. Default `False`. |
| `dropout` | float | Dropout between layers. Default `0`. |
| `bidirectional` | bool | Bidirectional. Default `False`. |

### `nn.MultiheadAttention`

| Parameter | Type | Description |
|-----------|------|-------------|
| `embed_dim` | int | Model dimension. |
| `num_heads` | int | Number of heads. |
| `dropout` | float | Attention dropout. Default `0`. |
| `bias` | bool | Add bias. Default `True`. |
| `batch_first` | bool | If `True`, batch is first dim. Default `False`. |

---

## 6. Internal Working

### Linear Layer

- Computes `y = xW^T + b`.
- If input is `(N, *, in_features)`, output is `(N, *, out_features)`.
- Implemented as a matrix multiplication with optional bias addition.

### Convolutional Layer

- For each output channel, applies a learnable filter across the input.
- Output at position `(i, j)` is `sum over input channels and kernel positions of (input * weight) + bias`.
- Implemented with im2col + GEMM or specialized convolution algorithms (e.g., cuDNN).
- `groups` splits input and output channels into groups; each group operates independently.
- Depthwise convolution: `groups = in_channels = out_channels`.

### Pooling Layer

- Slides a window over the input and applies a reduction (max or average).
- No learnable parameters.
- `AdaptiveAvgPool2d(1)` computes global average pooling.

### Normalization Layers

- **BatchNorm:** Normalizes using mean and variance computed over the batch and spatial dimensions for each channel. Maintains running statistics for eval mode.
- **LayerNorm:** Normalizes over the last `normalized_shape` dimensions per sample. No running stats.
- **GroupNorm:** Divides channels into groups and normalizes within each group per sample.
- **InstanceNorm:** Normalizes per sample per channel.

### Dropout

- During training, each element is zeroed with probability `p`. Surviving elements are scaled by `1/(1-p)` to maintain expected value.
- During eval, dropout is identity.

### Embedding

- A lookup table: `output = weight[index]`.
- Gradients are sparse; only the rows corresponding to used indices receive gradients.
- `padding_idx` ensures that row is zero and receives no gradient.

### Recurrent Layers

- Process sequences step by step.
- LSTM maintains cell state and hidden state; GRU merges them.
- `batch_first=True` swaps batch and sequence dimensions for convenience.
- `bidirectional=True` runs two RNNs in opposite directions and concatenates outputs.

### Multihead Attention

- Projects query, key, value to `embed_dim` using learned linear projections.
- Splits into `num_heads` heads of dimension `embed_dim / num_heads`.
- Computes scaled dot-product attention per head.
- Concatenates heads and projects output.

### Weight Initialization

- Default initialization in PyTorch:
  - `nn.Linear`: Kaiming uniform for weights, uniform for bias.
  - `nn.Conv2d`: Kaiming uniform for weights, uniform for bias.
  - `nn.Embedding`: Normal with mean 0, std 1.
  - `nn.LSTM`/`GRU`/`RNN`: Uniform over `(-1/sqrt(hidden_size), 1/sqrt(hidden_size))`.
- Custom initialization via `nn.init` functions applied with `model.apply`.

---

## 7. Common Mistakes

**Mistake:** Forgetting that `nn.Conv2d` expects `(N, C, H, W)` input.

**Why it happens:** Input may be `(N, H, W, C)` from other libraries.

**Correct approach:** Permute input or use `channels_last` memory format.

---

**Mistake:** Using `nn.CrossEntropyLoss` with `nn.Softmax` output.

**Why it happens:** `CrossEntropyLoss` includes softmax.

**Correct approach:** Output raw logits.

---

**Mistake:** Forgetting that `nn.BatchNorm2d` behaves differently in train vs eval.

**Why it happens:** Running stats used in eval mode.

**Correct approach:** Always call `model.train()` or `model.eval()`.

---

**Mistake:** Using `nn.Dropout` without switching modes.

**Why it happens:** Dropout is active in training mode only.

**Correct approach:** Switch modes with `model.train()`/`model.eval()`.

---

**Mistake:** Setting `inplace=True` for activation layers that are used in residual connections.

**Why it happens:** In-place ReLU may overwrite values needed for skip connection.

**Correct approach:** Use `inplace=False` or ensure the input is not needed.

---

**Mistake:** Using `nn.Embedding` with float indices.

**Why it happens:** Embedding expects integer (long) indices.

**Correct approach:** Ensure indices are `torch.long`.

---

**Mistake:** Forgetting `batch_first=True` for RNNs.

**Why it happens:** RNNs default to `(seq, batch, feature)`.

**Correct approach:** Set `batch_first=True` or permute input.

---

**Mistake:** Incorrect `num_heads` in `MultiheadAttention` that doesn't divide `embed_dim`.

**Why it happens:** `embed_dim` must be divisible by `num_heads`.

**Correct approach:** Ensure `embed_dim % num_heads == 0`.

---

**Mistake:** Using too high `p` in dropout.

**Why it happens:** Too much dropout underfits.

**Correct approach:** Use `p=0.1` to `0.5` typically.

---

**Mistake:** Forgetting that BatchNorm running stats are buffers, not parameters.

**Why it happens:** They are updated during forward pass in training mode.

**Correct approach:** Understand that they are state, not learnable.

---

**Mistake:** Using `nn.AdaptiveAvgPool2d(1)` but then not flattening.

**Why it happens:** Output shape is `(N, C, 1, 1)`.

**Correct approach:** Use `x.view(x.size(0), -1)` or `nn.Flatten()`.

---

**Mistake:** Not initializing embeddings or RNN weights correctly.

**Why it happens:** Default init may be suboptimal.

**Correct approach:** Use appropriate `nn.init` functions.

---

## 8. Important Differences

| Concept | Difference |
|---------|-----------|
| `nn.Linear` vs `nn.Conv2d` | Linear treats input as flattened features; Conv2d preserves spatial structure. |
| `nn.Conv2d` vs `nn.ConvTranspose2d` | Conv2d downsamples; ConvTranspose2d upsamples. |
| `nn.MaxPool2d` vs `nn.AvgPool2d` | Max takes maximum; Avg takes average. |
| `nn.BatchNorm2d` vs `nn.LayerNorm` | BatchNorm normalizes over batch; LayerNorm over features. |
| `nn.BatchNorm2d` vs `nn.InstanceNorm2d` | BatchNorm uses batch stats; InstanceNorm uses per-sample stats. |
| `nn.Dropout` vs `nn.Dropout2d` | Dropout zeroes elements; Dropout2d zeroes channels. |
| `nn.Embedding` vs `nn.Linear` | Embedding is a lookup table; Linear is a dense transformation. |
| `nn.LSTM` vs `nn.GRU` | LSTM has cell state; GRU has update/reset gates. |
| `nn.RNN` vs `nn.LSTM` | RNN has simple recurrence; LSTM has gating. |
| `nn.MultiheadAttention` vs `nn.Linear` | MultiheadAttention computes attention; Linear is a transformation. |
| `nn.ReLU` vs `F.relu` | Module vs function. |
| `inplace=True` vs `inplace=False` | In-place saves memory but can break autograd. |
| `groups=1` vs `groups=in_channels` | Standard vs depthwise convolution. |
| `padding=0` vs `padding=1` | No padding vs same-size output (with kernel 3). |

---

## 9. Important Rules / Facts

- All layers are `nn.Module` subclasses.
- `nn.Linear` computes `y = xW^T + b`.
- `nn.Conv2d` weight shape: `(out_channels, in_channels // groups, kH, kW)`.
- Output size formula: `H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)`.
- Pooling layers have no learnable parameters.
- BatchNorm has learnable `weight`/`bias` and running buffers.
- LayerNorm normalizes over the last dimension(s).
- GroupNorm divides channels into groups.
- Dropout is active only in training mode.
- `nn.Dropout` zeroes elements; `nn.Dropout2d` zeroes channels.
- `nn.Embedding` weight shape: `(num_embeddings, embedding_dim)`.
- `padding_idx` in Embedding produces zero vector and no gradient.
- RNNs default to `(seq, batch, feature)`; use `batch_first=True` for `(batch, seq, feature)`.
- LSTM returns `(output, (h, c))`; GRU/RNN return `(output, h)`.
- MultiheadAttention requires `embed_dim % num_heads == 0`.
- Weight initialization can be customized with `nn.init` functions.
- Default PyTorch initialization is Kaiming uniform for weights.
- `inplace=True` saves memory but can break autograd.
- `AdaptiveAvgPool2d(1)` is global average pooling.
- Use `nn.Flatten()` or `.view(x.size(0), -1)` after conv layers.
- Layer selection depends on data type: linear for tabular, conv for images, RNN/Transformer for sequences.

---

## 10. Practical Example

### Building a Convolutional Neural Network for Image Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1: (3, 32, 32) -> (32, 16, 16)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2: (32, 16, 16) -> (64, 8, 8)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3: (64, 8, 8) -> (128, 4, 4)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128*4*4)
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = ConvNet(num_classes=10)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Forward pass
x = torch.randn(4, 3, 32, 32)
y = model(x)
print("Output shape:", y.shape)  # (4, 10)

# Inspect layer output shapes
print("\nLayer output shapes:")
for name, module in model.named_children():
    print(f"  {name}: {module.__class__.__name__}")
```

**What it does:**
- Builds a 3-block CNN for image classification.
- Uses `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `Linear`, and `Dropout`.
- Demonstrates parameter counting and layer inspection.

**Important lines:**
- Each block: Conv → BatchNorm → ReLU → MaxPool.
- `x.view(x.size(0), -1)` flattens for FC layers.
- `Dropout(0.5)` before final layer for regularization.

**Expected output:**
```
ConvNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32, ...)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64, ...)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(128, ...)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=2048, out_features=256, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=256, out_features=10, bias=True)
)

Total parameters: 612,810
Trainable parameters: 612,810
Output shape: torch.Size([4, 10])
...
```

### Building a Small Transformer Encoder Block

```python
import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# Create block
block = TransformerEncoderBlock(embed_dim=64, num_heads=8, ff_dim=256, dropout=0.1)
x = torch.randn(4, 10, 64)  # (batch, seq, embed_dim)
y = block(x)
print("Output shape:", y.shape)  # (4, 10, 64)

# Parameters
for name, param in block.named_parameters():
    print(f"  {name}: {param.shape}")
```

**What it does:** Builds a transformer encoder block with self-attention, layer norm, and feed-forward network.

**Important lines:**
- `nn.MultiheadAttention` with `batch_first=True`.
- LayerNorm before and after feed-forward.
- Residual connections.

**Expected output:**
```
Output shape: torch.Size([4, 10, 64])
  attn.in_proj_weight: torch.Size([192, 64])
  attn.in_proj_bias: torch.Size([192])
  attn.out_proj.weight: torch.Size([64, 64])
  attn.out_proj.bias: torch.Size([64])
  norm1.weight: torch.Size([64])
  norm1.bias: torch.Size([64])
  norm2.weight: torch.Size([64])
  norm2.bias: torch.Size([64])
  ff.0.weight: torch.Size([256, 64])
  ff.0.bias: torch.Size([256])
  ff.3.weight: torch.Size([64, 256])
  ff.3.bias: torch.Size([64])
```

---

## 11. Chapter Summary

- Layers are the parameterized building blocks of neural networks.
- `nn.Linear`: fully connected layer.
- `nn.Conv2d`: 2D convolution with configurable kernel, stride, padding, dilation, groups.
- `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d`: pooling layers.
- `nn.BatchNorm2d`, `nn.LayerNorm`, `nn.GroupNorm`, `nn.InstanceNorm2d`: normalization layers.
- `nn.Dropout`, `nn.Dropout2d`: regularization.
- `nn.Embedding`: lookup table for discrete inputs.
- `nn.LSTM`, `nn.GRU`, `nn.RNN`: recurrent layers.
- `nn.MultiheadAttention`: attention mechanism.
- Weight initialization via `nn.init` and `model.apply`.
- Layer parameters have specific shapes; understanding them is essential.
- Common mistakes: wrong input shape, forgetting mode switching, wrong loss pairing, dtype mismatch.
- Layer selection depends on data type and task.
- Understanding layers enables building complex architectures like CNNs, RNNs, and Transformers.

---

## 12. Important APIs to Remember

| API | Purpose |
|-----|---------|
| `nn.Linear` | Fully connected layer |
| `nn.Conv1d/2d/3d` | Convolutional layers |
| `nn.ConvTranspose1d/2d/3d` | Transposed convolution |
| `nn.MaxPool1d/2d/3d` | Max pooling |
| `nn.AvgPool1d/2d/3d` | Average pooling |
| `nn.AdaptiveAvgPool2d` | Adaptive average pooling |
| `nn.BatchNorm1d/2d/3d` | Batch normalization |
| `nn.LayerNorm` | Layer normalization |
| `nn.GroupNorm` | Group normalization |
| `nn.InstanceNorm2d` | Instance normalization |
| `nn.Dropout` | Element dropout |
| `nn.Dropout2d` | Channel dropout |
| `nn.Embedding` | Embedding lookup |
| `nn.LSTM`, `nn.GRU`, `nn.RNN` | Recurrent layers |
| `nn.MultiheadAttention` | Multi-head attention |
| `nn.Upsample` | Upsampling |
| `nn.ZeroPad2d` | Zero padding |
| `nn.Flatten` | Flatten dimensions |
| `nn.init.*` | Weight initialization |
| `model.apply(fn)` | Apply init to all layers |

---

## 13. Key Takeaways

1. Layers are `nn.Module` subclasses with parameters and buffers.
2. `nn.Linear` computes `y = xW^T + b`; weight shape `(out, in)`.
3. `nn.Conv2d` weight shape `(out_channels, in_channels // groups, kH, kW)`.
4. Output size formula depends on kernel, stride, padding, dilation.
5. Pooling layers have no learnable parameters.
6. BatchNorm has learnable affine params and running buffers.
7. LayerNorm normalizes over last dimensions; GroupNorm divides channels.
8. Dropout is active only in training mode; `Dropout2d` zeroes channels.
9. `nn.Embedding` maps indices to vectors; `padding_idx` gives zero vector.
10. RNNs default to `(seq, batch, feature)`; use `batch_first=True` for convenience.
11. LSTM returns `(output, (h, c))`; GRU/RNN return `(output, h)`.
12. MultiheadAttention requires `embed_dim % num_heads == 0`.
13. Weight initialization can be customized with `nn.init` and `model.apply`.
14. Common mistakes: wrong input shape, forgetting mode switching, wrong loss pairing.
15. Understanding layers enables building CNNs, RNNs, and Transformers.
