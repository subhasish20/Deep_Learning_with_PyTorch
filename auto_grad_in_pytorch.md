

# 🔄 PyTorch Autograd Notes

---

## 🔹 1. Definition

- **Autograd** in PyTorch is the **automatic differentiation engine**.  
- It tracks operations on tensors and computes **gradients** (derivatives) automatically.  
- These gradients are essential for **backpropagation** in neural networks.  

👉 In short, Autograd powers the **learning process** by computing gradients for optimization.

---

## 🔹 2. Key Features

1. 📝 **Automatic Gradient Computation**  
   - Tracks tensor operations and builds a **computational graph**.  

2. 🌳 **Dynamic Computational Graph**  
   - Graph is built on-the-fly during execution (flexible & memory-efficient).  

3. ⚙️ **Backpropagation**  
   - `tensor.backward()` computes gradients w.r.t. all leaf tensors.  

4. 🛑 **Control Flow**  
   - Use `with torch.no_grad():` to stop tracking (e.g., inference).  

5. 🎯 **Optimizer Integration**  
   - Works seamlessly with `torch.optim` to update model parameters.  

---

## 🔹 3. Syntax / Core Functions

```python
# Create tensor with gradient tracking
x = torch.tensor(2.0, requires_grad=True)

# Perform operations
y = x ** 3 + 2 * x

# Backpropagation
y.backward()  

# Get gradient (dy/dx)
print(x.grad)
````

---

## 🔹 4. Important Attributes & Functions

1. **`requires_grad=True`** – Tells PyTorch to track operations on tensor.
2. **`backward()`** – Computes gradients using backpropagation.
3. **`.grad`** – Stores gradient of tensor.
4. **`torch.no_grad()`** – Temporarily disables autograd (for inference).
5. **`.detach()`** – Creates a tensor without gradient tracking.

---

## 🔹 5. Workflow

1. 📥 Create Tensors with `requires_grad=True`.
2. 🧮 Perform operations → PyTorch builds computation graph.
3. 🔄 Call `.backward()` on output.
4. 📊 Gradients are stored in `.grad` of leaf tensors.
5. 🚀 Optimizer updates parameters using gradients.

---

## 🔹 6. Example – Single Variable Gradient

```python
import torch

# Input tensor with gradient tracking
x = torch.tensor(5.0, requires_grad=True)

# Function: y = x^2
y = x ** 2

# Backprop
y.backward()

print("Value of x:", x.item())
print("dy/dx:", x.grad.item())  # dy/dx = 2x = 10
```

---

## 🔹 7. Example – Multiple Variables

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Function: y = a*b + b^2
y = a * b + b ** 2

y.backward()

print("dy/da:", a.grad.item())  # derivative wrt a = b = 3
print("dy/db:", b.grad.item())  # derivative wrt b = a + 2b = 2 + 6 = 8
```

---

## 🔹 8. Example – Using `torch.no_grad()`

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

# Disable gradient tracking
with torch.no_grad():
    z = x ** 3

print("Requires grad (y):", y.requires_grad)  # True
print("Requires grad (z):", z.requires_grad)  # False
```

---

## 🔹 9. Advantages

✅ No manual differentiation required.
✅ Efficient and dynamic computational graph.
✅ Works seamlessly with optimizers for training.
✅ Supports complex functions (CNNs, RNNs, Transformers).

---

## 🔹 10. Common Use Cases

1. 🎓 Training deep learning models.
2. 📉 Backpropagation in neural networks.
3. 📊 Computing gradients for optimization problems.
4. 🛠️ Research experiments requiring auto-differentiation.

---

## 🔹 11. Visual Workflow

```
Input Tensor (requires_grad=True)
        ↓
   Perform Operations
        ↓
 Computational Graph Built
        ↓
     backward()
        ↓
 Gradients stored in .grad
```

---

## ✅ Summary

**Autograd** is PyTorch’s engine for **automatic differentiation**,
allowing easy computation of gradients and enabling **backpropagation**.
It is the backbone of deep learning training 🚀.

---