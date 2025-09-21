
# 🧠 PyTorch `nn.Module` Notes

---

## 🔹 1. Definition

- **`nn.Module`** is the **base class for all neural network models** in PyTorch.  
- It provides the structure to define layers, parameters, and the **forward pass**.  
- Any custom model in PyTorch should inherit from `nn.Module`.  

👉 In short, `nn.Module` is the **blueprint for building neural networks** in PyTorch.

---

## 🔹 2. Key Features

1. 🏗️ **Model Building**  
   - Encapsulates layers and computations inside a class.  

2. 📦 **Parameter Management**  
   - Automatically registers learnable parameters (`nn.Linear`, `nn.Conv2d`, etc.).  

3. 🔄 **Forward Pass**  
   - Define how input tensors are transformed to outputs.  

4. 💾 **State Handling**  
   - Save/load model weights with `state_dict()`.  

5. 🔗 **Composability**  
   - Models can be nested (modules inside modules).  

---

## 🔹 3. Syntax

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers here
    
    def forward(self, x):
        # Define forward computation
        return x
````

---

## 🔹 4. Important Methods

1. **`__init__`** – Initialize layers.
2. **`forward(x)`** – Defines computation (called during model(x)).
3. **`parameters()`** – Returns all learnable parameters.
4. **`to(device)`** – Moves model to CPU/GPU.
5. **`state_dict()` / `load_state_dict()`** – Save and load model weights.

---

## 🔹 5. Workflow

1. 🏗️ **Define Model Class** → Inherit from `nn.Module`.
2. ⚙️ **Initialize Layers** → In `__init__`.
3. 🔄 **Define Forward Pass** → In `forward()`.
4. 🚀 **Train Model** → Pass data, compute loss, backpropagate, optimize.

---

## 🔹 6. Example – Simple Neural Network

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)   # input → hidden
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)    # hidden → output
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleNN()

# Print model architecture
print(model)

# Example forward pass
x = torch.randn(64, 784)  # batch of 64, each of size 784
output = model(x)
print(output.shape)  # torch.Size([64, 10])
```

---

## 🔹 7. Example – Using Nested Modules

```python
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(16 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

model = MyCNN()
print(model)
```

---

## 🔹 8. Advantages

✅ Provides a **clean structure** for models.
✅ Handles **automatic parameter registration**.
✅ Supports **GPU acceleration** (`.to("cuda")`).
✅ Easy to **save/load weights** with `state_dict()`.
✅ Allows **modular design** → models can be composed of multiple submodules.

---

## 🔹 9. Common Use Cases

1. 🖼️ Computer Vision (CNNs for image classification).
2. 📖 NLP (RNNs, Transformers).
3. 🎵 Speech/Audio recognition.
4. 📊 Tabular/structured data tasks.

---

## 🔹 10. Quick Comparison

| Without `nn.Module`             | With `nn.Module`              |
| ------------------------------- | ----------------------------- |
| ❌ Hard to organize layers       | ✅ Layers neatly defined       |
| ❌ Manual parameter management   | ✅ Auto parameter registration |
| ❌ No built-in save/load support | ✅ `state_dict()` support      |

---

## 🔹 11. Visual Workflow

```
Input Tensor  --->  Model (nn.Module)  --->  Forward Pass  --->  Output
                       |      |
                       |      └──> Layers (Linear, Conv, etc.)
                       └────> Parameters (Weights & Biases)
```

---

## ✅ Summary

`nn.Module` is the **foundation of all neural networks in PyTorch**.
It organizes layers, parameters, and the forward pass, making model building clean, modular, and efficient 🚀.

---

