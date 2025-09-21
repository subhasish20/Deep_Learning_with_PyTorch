
# 📘 PyTorch DataLoader Notes

---

## 🔹 1. Definition

- **DataLoader** in PyTorch is a **utility class** that provides an efficient way to iterate over datasets.  
- It combines a dataset (`torch.utils.data.Dataset`) with a **sampler** and supports batching, shuffling, and parallel data loading.  

👉 In short, `DataLoader` helps feed data to your model during training and testing.

---

## 🔹 2. Key Features

1. 📦 **Batch Loading**  
   - Groups samples into mini-batches for training.  
   - Reduces computational overhead compared to one-by-one loading.  

2. 🔀 **Shuffling**  
   - Randomizes the order of samples each epoch to avoid model overfitting to sequence patterns.  

3. ⚡ **Parallelism (`num_workers`)**  
   - Loads data using multiple subprocesses, speeding up data loading.  

4. 🔄 **Automatic Batching (`batch_size`)**  
   - Automatically stacks tensors into batches.  

5. 🛠️ **Customizable**  
   - Works with custom datasets created by extending `torch.utils.data.Dataset`.  

---

## 🔹 3. Syntax

```python
torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    pin_memory=False
)
````

---

## 🔹 4. Important Parameters

1. **`dataset`** – The dataset object (must follow Dataset API).
2. **`batch_size`** – Number of samples per batch. (Default: `1`)
3. **`shuffle`** – Shuffle data at every epoch. (Default: `False`)
4. **`num_workers`** – Number of subprocesses for data loading (`0` = main process).
5. **`collate_fn`** – Function to merge samples into a batch (useful for variable-sized data).
6. **`drop_last`** – If `True`, drops last incomplete batch.
7. **`pin_memory`** – Speeds up transfer of data to GPU if `True`.

---

## 🔹 5. Workflow

1. 🗂️ **Dataset Creation**

   * Use built-in datasets (`torchvision.datasets`) or custom Dataset.

2. 📥 **Pass Dataset to DataLoader**

   * Wraps dataset to provide batch-wise, iterable data.

3. 🔁 **Iterate over DataLoader**

   * Returns batches of `(data, labels)` during training.

---

## 🔹 6. Example – Using Built-in Dataset

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Transformations
transform = transforms.ToTensor()

# 2. Dataset
train_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)

# 3. DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# 4. Iterating
for images, labels in train_loader:
    print(images.shape)  # torch.Size([32, 1, 28, 28])
    print(labels.shape)  # torch.Size([32])
    break
```

---

## 🔹 7. Example – Custom Dataset with DataLoader

```python
from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Data
data = torch.arange(10).float().unsqueeze(1)  # shape [10, 1]
labels = torch.arange(10)

# Dataset & DataLoader
dataset = MyDataset(data, labels)
loader = DataLoader(dataset, batch_size=3, shuffle=True)

for x, y in loader:
    print(x, y)
```

---

## 🔹 8. Advantages

✅ Simplifies data batching and shuffling.
✅ Supports **parallel data loading** → reduces bottlenecks.
✅ Flexible with **custom datasets and collate functions**.
✅ Essential for **large-scale deep learning training**.

---

## 🔹 9. Common Use Cases

1. 🖼️ Loading image datasets (e.g., MNIST, CIFAR).
2. 📖 NLP tasks with variable-length sequences (using `collate_fn`).
3. 📊 Custom datasets (e.g., CSV, medical data, time series).
4. ⚙️ Efficient GPU training pipelines.

---

## 🔹 10. Quick Comparison

| Without DataLoader                 | With DataLoader             |
| ---------------------------------- | --------------------------- |
| ❌ Manual batch creation            | ✅ Automatic batching        |
| ❌ Sequential data access           | ✅ Supports shuffling        |
| ❌ Slower training (CPU bottleneck) | ✅ Faster (parallel loading) |

---

## 🔹 11. Visual Workflow

```
Dataset  --->  DataLoader  --->  Model Training Loop
                |   |
                |   └──> Batching, Shuffling, Parallelism
                └────> Efficient Iteration
```

---

## ✅ Summary

`DataLoader` is a PyTorch utility that efficiently handles batching, shuffling, and parallel loading of data from datasets.
It is crucial for building scalable and efficient deep learning pipelines 🚀.

---

