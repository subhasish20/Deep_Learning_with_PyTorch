
# 🤖 Artificial Neural Network (ANN) in PyTorch

---

## 🔹 1. Definition

- An **Artificial Neural Network (ANN)** is a computational model inspired by the human brain.  
- It consists of **layers of interconnected nodes (neurons)** that transform input data into outputs.  
- In PyTorch, ANN models are built using **`torch.nn.Module`** and trained with **Autograd + Optimizers**.  

👉 ANN learns by adjusting **weights and biases** using **backpropagation** and **gradient descent**.

---

## 🔹 2. Structure of ANN

1. **Input Layer** – Accepts features from dataset.  
2. **Hidden Layers** – Perform transformations using activation functions.  
3. **Output Layer** – Produces predictions.  

Each neuron performs:  

\[
y = f(Wx + b)
\]

Where:  
- \( W \) = weights  
- \( b \) = bias  
- \( f \) = activation function  

---

## 🔹 3. Key Components in PyTorch

1. 🏗️ **Model (`nn.Module`)** – Defines layers and forward pass.  
2. ⚡ **Activation Functions** – ReLU, Sigmoid, Tanh, Softmax.  
3. 📉 **Loss Function** – e.g., `nn.CrossEntropyLoss`, `nn.MSELoss`.  
4. 🚀 **Optimizer** – e.g., SGD, Adam.  
5. 🔄 **Training Loop** – Forward pass → Loss → Backward pass → Update.  

---

## 🔹 4. Example – Simple ANN for Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define ANN
class ANNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input → Hidden
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size) # Hidden → Output
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hyperparameters
input_size = 784   # e.g., MNIST images (28x28)
hidden_size = 128
output_size = 10   # digits 0–9
learning_rate = 0.001

# Model, Loss, Optimizer
model = ANNModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy data (batch of 64 samples)
x = torch.randn(64, input_size)
y = torch.randint(0, 10, (64,))

# Training step
outputs = model(x)
loss = criterion(outputs, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())
````

---

## 🔹 5. Example – ANN Training Loop (Pseudocode)

```python
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Flatten images: [batch, 1, 28, 28] → [batch, 784]
        images = images.view(-1, 28*28)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

---

## 🔹 6. Advantages of ANN

✅ Learns complex non-linear relationships.
✅ Works across domains (vision, NLP, speech, etc.).
✅ Flexible architecture (deep, shallow, wide).
✅ Automatic gradient computation with Autograd.

---

## 🔹 7. Common Use Cases

1. 🖼️ Image Classification (e.g., MNIST, CIFAR-10).
2. 📊 Regression Problems (predicting continuous values).
3. 📖 Natural Language Processing (text classification, embeddings).
4. 🎵 Audio & Speech Recognition.

---

## 🔹 8. Visual Workflow

```
Input Data  --->  Input Layer  --->  Hidden Layers (ReLU, Sigmoid)  --->  Output Layer
                        |                          |
                        |                          └─> Produces final prediction
                        └─> Learns features via weights & biases
```

---

## 🔹 9. Quick Comparison

| Feature          | ANN in PyTorch                          |
| ---------------- | --------------------------------------- |
| Layer Definition | `nn.Linear`                             |
| Activation       | `nn.ReLU()`, `nn.Sigmoid()`             |
| Loss Functions   | `nn.CrossEntropyLoss()`, `nn.MSELoss()` |
| Optimizers       | `torch.optim.SGD`, `torch.optim.Adam`   |
| Training Support | Autograd + Backpropagation              |

---

## ✅ Summary

An **Artificial Neural Network (ANN)** in PyTorch is built using **`nn.Module`**,
trained using **Autograd + Optimizers**, and applied to various tasks like classification and regression 🚀.

---

