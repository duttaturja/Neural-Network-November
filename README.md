# Neural Network November (NNN)

**Keep Your Gradients Tight and Neurons Strong** 💪🧠

Welcome to **Neural Network November**, the ultimate PyTorch playground where neurons stay disciplined, gradients don’t leak, and your models train harder than you do. This repo contains classic neural network exercises with a meme twist.

---

## 🎯 About the Repo

NNN is a collection of PyTorch projects designed to help you **master neural networks from scratch** while having fun. Each notebook focuses on a **different type of dataset and neural network**:

1. **Iris Classification** – Fully connected NN for 3-class tabular data
2. **MNIST Digits** – CNN for image classification
3. **Diabetes Regression** – Linear regression NN
4. **Breast Cancer Binary Classification** – Logistic regression NN
5. **Make Moons Classification** – Nonlinear 2D dataset with a feed-forward NN

The goal: **maximize accuracy, minimize overfitting, and survive Neural Network November**.

---

## 🧰 Repo Structure

```
NNN/
├─ README.md
├─ notebooks/
│   ├─ Classification_FNN.ipynb
│   ├─ Classification_CNN.ipynb
│   ├─ Classification_MLP.ipynb
│   ├─ Linear_Regression.ipynb
│   ├─ Logistic_Regression.ipynb

```

---

## 🚀 Features

* Step-by-step **Jupyter notebooks** for each dataset
* Fully implemented **PyTorch neural networks from scratch**
* **Training loops**, **evaluation**, and **visualizations**
* Plots for **loss curves**, **accuracy**, and **decision boundaries**
* Saved model checkpoints (`.pt`) for quick reuse
* Meme-inspired guidance for **NNN discipline**

---

## 📚 Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/Neural-Network-November.git
cd Neural-Network-November
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install torch torchvision scikit-learn matplotlib pandas numpy
```

4. **Open Jupyter Notebook**

```bash
jupyter notebook
```

Then open any notebook inside the `notebooks/` folder.

---

## 🏆 Datasets Covered

| Dataset       | Type                                  | Notebook                          |
| ------------- | ------------------------------------- | --------------------------------- |
| Iris          | Classification (3 classes)            | `iris_classification.ipynb`       |
| MNIST         | Image Classification (0-9)            | `mnist_cnn.ipynb`                 |
| Diabetes      | Regression (continuous)               | `diabetes_regression.ipynb`       |
| Breast Cancer | Binary Classification                 | `breast_cancer_logistic.ipynb`    |
| Make Moons    | Classification (2 classes, nonlinear) | `make_moons_classification.ipynb` |

---

## 📈 Tips for Neural Network November

* **Keep your gradients tight** – watch learning rate and avoid exploding gradients
* **Neurons strong** – experiment with hidden layers and activation functions
* **Normalize inputs** – standardize features for tabular datasets
* **Batch discipline** – shuffle and batch your data properly
* **Plot everything** – loss curves, accuracy curves, decision boundaries
* **Experiment** – tweak hidden layers, dropout, learning rates, and noise for make_moons

---

## 🤪 Meme Corner

> “All forward passes, no backward pleasure.”
> “Weights locked. Biases suppressed. Gradients obey.”
> “Zero Dropout. Maximum Discipline.”

NNN isn’t just about training networks — it’s about **staying disciplined, having fun, and surviving November**.

---

## 💾 Saved Models

All notebooks save their trained PyTorch models in the `saved_models/` folder. You can load any `.pt` file and evaluate on test data or continue training:

```python
import torch
# Example for Make Moons
from make_moons_model import Net  # replace with your model class

model = Net()
model.load_state_dict(torch.load('saved_models/make_moons_model.pt'))
model.eval()
```

Replace the filename with the appropriate dataset model.

---

## ⚡ Contributing

Feel free to:

* Add more datasets
* Add more neural network experiments
* Make NNN memes even funnier 😎

Fork the repo, make your changes, and open a pull request.

---

## 📜 License

MIT License — do whatever you want, but **keep your gradients tight and neurons strong** 😉

---

**Neural Network November (NNN)**
