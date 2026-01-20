# The Complete AI Engineering Interview Guide
## A Comprehensive Resource for Mastering Artificial Intelligence Engineering

---

<div align="center">

**From Foundations to Frontiers: Everything You Need to Ace Your AI Engineering Interview**

*Based on comprehensive coursework and industry best practices*

</div>

---

# Table of Contents

1. [Introduction to AI Engineering](#part-1-introduction-to-ai-engineering)
2. [Machine Learning Foundations](#part-2-machine-learning-foundations)
3. [Deep Learning with Keras & TensorFlow](#part-3-deep-learning-with-keras--tensorflow)
4. [Deep Learning with PyTorch](#part-4-deep-learning-with-pytorch)
5. [Convolutional Neural Networks & Computer Vision](#part-5-convolutional-neural-networks--computer-vision)
6. [Transformers & Attention Mechanisms](#part-6-transformers--attention-mechanisms)
7. [Generative AI & Large Language Models](#part-7-generative-ai--large-language-models)
8. [Advanced Fine-Tuning Techniques](#part-8-advanced-fine-tuning-techniques)
9. [Reinforcement Learning](#part-9-reinforcement-learning)
10. [MLOps & Model Deployment](#part-10-mlops--model-deployment)
11. [Interview Questions & Answers](#part-11-interview-questions--answers)
12. [Quick Reference Cheat Sheets](#part-12-quick-reference-cheat-sheets)

---

# Part 1: Introduction to AI Engineering

## What is AI Engineering?

AI Engineering is the discipline of building, deploying, and maintaining artificial intelligence systems in production environments. It bridges the gap between research and practical applications, combining software engineering principles with machine learning expertise.

### Key Responsibilities of an AI Engineer

1. **Model Development**: Designing and training ML/DL models
2. **Data Pipeline Engineering**: Building robust data ingestion and preprocessing systems
3. **Model Optimization**: Improving model performance and efficiency
4. **Deployment**: Moving models from development to production
5. **Monitoring**: Tracking model performance and detecting drift
6. **Infrastructure**: Managing compute resources (GPUs, TPUs, cloud services)

### The AI Engineering Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Applications                          │
│        (Chatbots, Recommendation Systems, Vision)        │
├─────────────────────────────────────────────────────────┤
│                   Model Serving                          │
│          (TensorFlow Serving, TorchServe, ONNX)         │
├─────────────────────────────────────────────────────────┤
│                 Model Training                           │
│            (PyTorch, TensorFlow, JAX, Keras)            │
├─────────────────────────────────────────────────────────┤
│                  Data Processing                         │
│           (Pandas, NumPy, Apache Spark, Dask)           │
├─────────────────────────────────────────────────────────┤
│                   Infrastructure                         │
│             (AWS, GCP, Azure, Kubernetes)               │
└─────────────────────────────────────────────────────────┘
```

---

# Part 2: Machine Learning Foundations

## 2.1 Supervised Learning

Supervised learning uses labeled data to learn a mapping from inputs to outputs.

### Linear Regression

**Purpose**: Predict continuous values by fitting a linear relationship.

**Mathematical Foundation**:
```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

**Cost Function (Mean Squared Error)**:
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Key Interview Questions**:
- *Q: What are the assumptions of linear regression?*
  - A: Linearity, independence, homoscedasticity, normality of residuals, no multicollinearity.

- *Q: How do you handle multicollinearity?*
  - A: Use VIF (Variance Inflation Factor), remove correlated features, apply PCA, or use regularization.

### Multiple Linear Regression

Extends simple linear regression to multiple features:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
r2_score = model.score(X_test, y_test)
```

### Logistic Regression

**Purpose**: Binary or multi-class classification using the sigmoid function.

**Sigmoid Function**:
```
σ(z) = 1 / (1 + e^(-z))
```

**Loss Function (Binary Cross-Entropy)**:
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Key Interview Questions**:
- *Q: Why can't we use MSE for logistic regression?*
  - A: MSE creates a non-convex loss surface with local minima, making optimization difficult. Cross-entropy provides a convex surface for binary classification.

- *Q: What's the difference between logistic regression and linear regression?*
  - A: Linear regression predicts continuous values; logistic regression predicts probabilities (0-1) for classification using the sigmoid function.

### Decision Trees

**Concept**: Hierarchical structure making decisions based on feature thresholds.

**Splitting Criteria**:
- **Gini Impurity**: `Gini = 1 - Σ(pᵢ)²`
- **Entropy/Information Gain**: `Entropy = -Σ(pᵢ·log₂(pᵢ))`

**Advantages**:
- Interpretable
- No feature scaling needed
- Handles non-linear relationships

**Disadvantages**:
- Prone to overfitting
- Sensitive to data variations
- Can create biased trees with imbalanced data

### Random Forests

**Concept**: Ensemble of decision trees using bagging (bootstrap aggregating).

**Key Hyperparameters**:
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split a node
- `max_features`: Features considered at each split

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train, y_train)
```

### Gradient Boosting (XGBoost, LightGBM, CatBoost)

**Concept**: Sequential ensemble where each model corrects errors of previous models.

**Key Differences**:

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Tree Growth | Level-wise | Leaf-wise | Level-wise |
| Speed | Fast | Faster | Fast |
| Categorical | Needs encoding | Native support | Best native support |
| Memory | High | Lower | Moderate |

### Support Vector Machines (SVM)

**Concept**: Finds optimal hyperplane maximizing margin between classes.

**Kernel Trick**: Maps data to higher dimensions for linear separability.

**Common Kernels**:
- Linear: `K(x, y) = x·y`
- Polynomial: `K(x, y) = (γ·x·y + r)^d`
- RBF/Gaussian: `K(x, y) = exp(-γ||x-y||²)`

### K-Nearest Neighbors (KNN)

**Concept**: Classifies based on majority vote of k nearest neighbors.

**Distance Metrics**:
- Euclidean: `√Σ(xᵢ - yᵢ)²`
- Manhattan: `Σ|xᵢ - yᵢ|`
- Minkowski: `(Σ|xᵢ - yᵢ|^p)^(1/p)`

**Key Consideration**: Feature scaling is essential!

## 2.2 Unsupervised Learning

### K-Means Clustering

**Algorithm**:
1. Initialize k centroids randomly
2. Assign points to nearest centroid
3. Recalculate centroids
4. Repeat until convergence

**Elbow Method**: Plot inertia vs k to find optimal clusters.

**Silhouette Score**: Measures cluster cohesion and separation (-1 to 1).

### Hierarchical Clustering

**Types**:
- **Agglomerative** (bottom-up): Start with individual points, merge
- **Divisive** (top-down): Start with one cluster, split

**Linkage Methods**:
- Single: Minimum distance
- Complete: Maximum distance
- Average: Average distance
- Ward: Minimizes variance

### DBSCAN (Density-Based Spatial Clustering)

**Key Parameters**:
- `eps`: Maximum distance between points in cluster
- `min_samples`: Minimum points to form dense region

**Advantages**: Finds arbitrary shapes, identifies outliers.

### Principal Component Analysis (PCA)

**Purpose**: Dimensionality reduction while preserving variance.

**Steps**:
1. Standardize data
2. Compute covariance matrix
3. Calculate eigenvectors and eigenvalues
4. Select top k components
5. Transform data

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Retain 95% variance
X_reduced = pca.fit_transform(X)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

## 2.3 Model Evaluation

### Classification Metrics

**Confusion Matrix**:
```
                 Predicted
                 Pos    Neg
Actual  Pos      TP     FN
        Neg      FP     TN
```

**Key Metrics**:
- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
- **Precision**: `TP / (TP + FP)` - "Of predicted positives, how many correct?"
- **Recall/Sensitivity**: `TP / (TP + FN)` - "Of actual positives, how many found?"
- **F1-Score**: `2 × (Precision × Recall) / (Precision + Recall)`
- **Specificity**: `TN / (TN + FP)`

**ROC-AUC**: Area under ROC curve (TPR vs FPR). 0.5 = random, 1.0 = perfect.

### Regression Metrics

- **MSE**: `(1/n) Σ(yᵢ - ŷᵢ)²`
- **RMSE**: `√MSE`
- **MAE**: `(1/n) Σ|yᵢ - ŷᵢ|`
- **R² Score**: `1 - (SS_res / SS_tot)` - Proportion of variance explained

### Cross-Validation

**K-Fold Cross-Validation**:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

**Stratified K-Fold**: Maintains class distribution in each fold.

## 2.4 Regularization

### L1 Regularization (Lasso)
```
Loss = MSE + λ Σ|wᵢ|
```
- Produces sparse models (feature selection)
- Can zero out coefficients

### L2 Regularization (Ridge)
```
Loss = MSE + λ Σwᵢ²
```
- Prevents large weights
- All features retained (small weights)

### Elastic Net
```
Loss = MSE + λ₁Σ|wᵢ| + λ₂Σwᵢ²
```
- Combines L1 and L2
- Good when features are correlated

---

# Part 3: Deep Learning with Keras & TensorFlow

## 3.1 Neural Network Fundamentals

### The Perceptron

The simplest neural network unit:
```
output = activation(Σ(wᵢ × xᵢ) + bias)
```

### Multi-Layer Perceptron (MLP)

**Components**:
- Input layer: Receives features
- Hidden layers: Learn representations
- Output layer: Produces predictions

### Forward Propagation

Data flows through network:
```
z¹ = W¹x + b¹     # Linear transformation
a¹ = σ(z¹)         # Activation
z² = W²a¹ + b²    # Next layer
a² = σ(z²)         # Output
```

### Backpropagation

**Chain Rule Application**:
```
∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W
```

**Key Interview Question**:
- *Q: Explain backpropagation in simple terms.*
- A: It's the algorithm that calculates how much each weight contributed to the error, then adjusts weights proportionally. It uses the chain rule to propagate gradients backward from output to input.

## 3.2 Activation Functions

### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
```
**Pros**: Fast, reduces vanishing gradient
**Cons**: Dying ReLU problem

### Leaky ReLU
```
f(x) = max(αx, x), where α ≈ 0.01
```
**Fixes**: Dying ReLU by allowing small negative values

### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
```
**Range**: (0, 1)
**Use**: Binary classification output
**Issue**: Vanishing gradient for extreme values

### Tanh
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
**Range**: (-1, 1)
**Use**: Zero-centered, better than sigmoid for hidden layers

### Softmax
```
softmax(xᵢ) = e^xᵢ / Σe^xⱼ
```
**Use**: Multi-class classification output (probability distribution)

### GELU (Gaussian Error Linear Unit)
```
f(x) = x × Φ(x)
```
**Use**: Transformers, modern architectures

## 3.3 Building Models with Keras

### Sequential API

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Functional API (For Complex Architectures)

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate

# Multiple inputs
input_a = Input(shape=(32,), name='input_a')
input_b = Input(shape=(32,), name='input_b')

# Parallel branches
x1 = Dense(64, activation='relu')(input_a)
x2 = Dense(64, activation='relu')(input_b)

# Merge
merged = concatenate([x1, x2])
output = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[input_a, input_b], outputs=output)
```

### Custom Layers

```python
from tensorflow.keras.layers import Layer

class CustomDense(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

### Custom Training Loops

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
```

## 3.4 Loss Functions

### For Regression
```python
# Mean Squared Error
loss = 'mse'  # or tf.keras.losses.MeanSquaredError()

# Mean Absolute Error
loss = 'mae'

# Huber Loss (less sensitive to outliers)
loss = tf.keras.losses.Huber(delta=1.0)
```

### For Classification
```python
# Binary classification
loss = 'binary_crossentropy'

# Multi-class (one-hot labels)
loss = 'categorical_crossentropy'

# Multi-class (integer labels)
loss = 'sparse_categorical_crossentropy'
```

## 3.5 Optimizers

### SGD with Momentum
```python
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)
```

### Adam (Adaptive Moment Estimation)
```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,      # Momentum
    beta_2=0.999,    # RMSprop
    epsilon=1e-7
)
```

**Adam combines**:
- Momentum: Exponential moving average of gradients
- RMSprop: Adaptive learning rates per parameter

### Learning Rate Schedules

```python
# Exponential decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.96
)

# Reduce on plateau (callback)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001
)
```

## 3.6 Regularization Techniques

### Dropout
```python
# During training: Randomly zeros units with probability p
# During inference: All units active, outputs scaled
Dense(256, activation='relu'),
Dropout(0.5),  # 50% dropout rate
```

### Batch Normalization
```python
# Normalizes activations: (x - μ) / σ × γ + β
Dense(256),
BatchNormalization(),
Activation('relu'),
```

**Benefits**:
- Enables higher learning rates
- Reduces internal covariate shift
- Acts as regularizer

### Early Stopping
```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

## 3.7 Hyperparameter Tuning with Keras Tuner

```python
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    
    # Tune number of units
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(hp_units, activation='relu'))
    
    # Tune dropout rate
    hp_dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
    model.add(Dropout(hp_dropout))
    
    model.add(Dense(10, activation='softmax'))
    
    # Tune learning rate
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Search strategies
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='tuning',
    project_name='my_tuning'
)

tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
```

---

# Part 4: Deep Learning with PyTorch

## 4.1 PyTorch Fundamentals

### Tensors

```python
import torch

# Create tensors
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)
randn = torch.randn(3, 4)  # Normal distribution

# Device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

# Automatic differentiation
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
```

### Autograd (Automatic Differentiation)

```python
# Computation graph
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x**2 + y**3
z.backward()

print(f"dz/dx = {x.grad}")  # 2x = 4
print(f"dz/dy = {y.grad}")  # 3y² = 27
```

## 4.2 Building Models in PyTorch

### nn.Module Pattern

```python
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = NeuralNetwork(784, 256, 10).to(device)
```

### nn.Sequential

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

## 4.3 Data Loading

### Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[idx]

# Create DataLoader
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)
```

### Image Transforms

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## 4.4 Training Loop

```python
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
```

## 4.5 Weight Initialization

### Xavier/Glorot Initialization
Best for tanh activations:
```python
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

### He/Kaiming Initialization
Best for ReLU activations:
```python
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

### Custom Initialization
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')

model.apply(init_weights)
```

---

# Part 5: Convolutional Neural Networks & Computer Vision

## 5.1 CNN Fundamentals

### Convolution Operation

```
Output[i,j] = Σₘ Σₙ Input[i+m, j+n] × Kernel[m, n]
```

**Key Parameters**:
- **Kernel/Filter Size**: Typically 3×3, 5×5, 7×7
- **Stride**: Step size (1 = pixel by pixel, 2 = skip every other)
- **Padding**: 'same' preserves dimensions, 'valid' reduces

**Output Size Formula**:
```
Output = (Input - Kernel + 2×Padding) / Stride + 1
```

### Pooling Layers

**Max Pooling**: Takes maximum value in window
**Average Pooling**: Takes average value
**Global Average Pooling**: Reduces entire feature map to single value

### CNN Architecture Pattern

```
Input → [Conv → BN → ReLU → Pool] × N → Flatten → Dense → Output
```

## 5.2 Building CNNs

### Keras CNN

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Block 2
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Block 3
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Classification head
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

### PyTorch CNN

```python
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

## 5.3 Classic CNN Architectures

### LeNet-5 (1998)
First successful CNN for digit recognition.
```
Input → Conv5 → Pool → Conv5 → Pool → FC → FC → Output
```

### AlexNet (2012)
Won ImageNet, popularized deep learning.
- 8 layers (5 conv, 3 FC)
- ReLU activation
- Dropout regularization
- Local Response Normalization

### VGGNet (2014)
Showed deeper is better with small 3×3 filters.
```
VGG-16: 13 conv layers + 3 FC layers
VGG-19: 16 conv layers + 3 FC layers
```
**Key Insight**: Two 3×3 convs have same receptive field as one 5×5, but fewer parameters.

### ResNet (2015)
Introduced residual connections to train very deep networks.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)  # Skip connection!
        return F.relu(out)
```

### Inception/GoogLeNet (2014)
Multiple parallel convolutions at different scales.

### EfficientNet (2019)
Compound scaling of depth, width, and resolution.

## 5.4 Transfer Learning

### Feature Extraction (Freeze Base)

```python
from tensorflow.keras.applications import VGG16

# Load pre-trained model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers
base_model.trainable = False

# Add custom classifier
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

### Fine-Tuning (Unfreeze Some Layers)

```python
# First train with frozen base...

# Then unfreeze top layers of base model
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
model.fit(train_data, epochs=10)
```

### PyTorch Transfer Learning

```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)

# Only train new layers
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

## 5.5 Data Augmentation

### Keras ImageDataGenerator

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

### Advanced Augmentation (Albumentations)

```python
import albumentations as A

transform = A.Compose([
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

# Part 6: Transformers & Attention Mechanisms

## 6.1 The Attention Mechanism

### Intuition

Attention allows models to focus on relevant parts of input when producing output. Like how humans focus on specific words when translating.

### Self-Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I provide?"
- **d_k**: Dimension of keys (for scaling)

### Step-by-Step Self-Attention

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    
    # 1. Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 2. Apply mask (optional, for decoder)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3. Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # 4. Weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

### Multi-Head Attention

Multiple attention heads learn different relationships:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_k
        )
        
        return self.W_o(attn_output)
```

## 6.2 Positional Encoding

Transformers have no inherent notion of position, so we add positional information:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

**Mathematical Formulation**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## 6.3 The Transformer Architecture

### Full Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
```

### Encoder-Only (BERT-style)

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)
```

### Decoder-Only (GPT-style)

```python
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # Causal mask (prevents attending to future tokens)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, ~mask)
        
        x = self.norm(x)
        return self.output(x)
```

## 6.4 Vision Transformer (ViT)

### Concept

Treat images as sequences of patches:

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, 
                 num_heads, num_layers, num_classes):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            vocab_size=0,  # Not used
            d_model=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=embed_dim * 4
        )
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer.layers[0](x)  # Simplified
        
        # Classification from [CLS] token
        return self.head(x[:, 0])
```

## 6.5 Key Transformer Variants

| Model | Type | Use Case | Key Innovation |
|-------|------|----------|----------------|
| **BERT** | Encoder | NLU, Classification | Masked language modeling |
| **GPT** | Decoder | Text generation | Autoregressive |
| **T5** | Encoder-Decoder | Text-to-text | Unified framework |
| **ViT** | Encoder | Vision | Image patches |
| **CLIP** | Multimodal | Vision-Language | Contrastive learning |

---

# Part 7: Generative AI & Large Language Models

## 7.1 Language Modeling Fundamentals

### N-gram Models

**Unigram**: `P(word)`
**Bigram**: `P(word | previous_word)`
**Trigram**: `P(word | two_previous_words)`

```python
from nltk import ngrams
from collections import Counter

# Build bigram model
def build_bigram_model(corpus):
    tokens = corpus.split()
    bigrams = list(ngrams(tokens, 2))
    bigram_freq = Counter(bigrams)
    unigram_freq = Counter(tokens)
    
    # Probability: P(w2|w1) = count(w1,w2) / count(w1)
    model = {}
    for (w1, w2), freq in bigram_freq.items():
        if w1 not in model:
            model[w1] = {}
        model[w1][w2] = freq / unigram_freq[w1]
    
    return model
```

### Neural Language Models

**RNN-based**: Sequential processing, vanishing gradients
**LSTM/GRU**: Gated mechanisms for long-term memory
**Transformer**: Parallel processing, attention mechanism

## 7.2 Tokenization

### Types of Tokenization

**Word-level**: "Hello world" → ["Hello", "world"]
- Simple but large vocabulary, OOV problem

**Character-level**: "Hello" → ["H", "e", "l", "l", "o"]
- Small vocabulary but loses semantic meaning

**Subword (BPE, WordPiece, SentencePiece)**: "unhappiness" → ["un", "happiness"]
- Balance between vocabulary size and meaning

### Byte-Pair Encoding (BPE)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Train
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=30000
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Use
output = tokenizer.encode("Hello, how are you?")
print(output.tokens)
```

### Using Hugging Face Tokenizers

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode
encoding = tokenizer(
    "Hello, how are you?",
    padding="max_length",
    max_length=128,
    truncation=True,
    return_tensors="pt"
)

print(encoding['input_ids'])
print(encoding['attention_mask'])

# Decode
decoded = tokenizer.decode(encoding['input_ids'][0])
```

## 7.3 Large Language Model Architectures

### GPT Architecture (Decoder-Only)

```
Input → Token Embedding + Position Embedding → 
[Masked Multi-Head Attention → FFN] × N → 
Linear → Softmax → Output Probabilities
```

**Key Features**:
- Autoregressive: Generates one token at a time
- Causal masking: Only attends to previous tokens
- Pre-training: Next token prediction

### BERT Architecture (Encoder-Only)

```
Input → Token + Segment + Position Embeddings → 
[Multi-Head Attention → FFN] × N → 
Output Embeddings
```

**Key Features**:
- Bidirectional: Attends to all tokens
- Pre-training: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- Fine-tuning: Add task-specific heads

### T5 Architecture (Encoder-Decoder)

All NLP tasks framed as text-to-text:
- Translation: "translate English to French: Hello" → "Bonjour"
- Summarization: "summarize: [long text]" → "[summary]"
- Classification: "classify: [text]" → "positive"

## 7.4 Pre-training Objectives

### Causal Language Modeling (CLM)
Predict next token given previous tokens.
```
P(w_t | w_1, w_2, ..., w_{t-1})
```

### Masked Language Modeling (MLM)
Predict masked tokens given context.
```
"The [MASK] sat on the mat" → "cat"
```

### Span Corruption (T5)
Replace spans with sentinel tokens, predict spans.

## 7.5 Inference Strategies

### Greedy Decoding
Always pick highest probability token.
```python
def greedy_decode(model, input_ids, max_length):
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids
```

### Beam Search
Keep top-k candidates at each step.

### Sampling Strategies

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

output = model.generate(
    input_ids,
    max_length=100,
    
    # Temperature: Higher = more random
    temperature=0.7,
    
    # Top-k: Only consider k most likely tokens
    top_k=50,
    
    # Top-p (nucleus): Consider tokens with cumulative prob > p
    top_p=0.95,
    
    # Repetition penalty
    repetition_penalty=1.2,
    
    # Number of sequences
    num_return_sequences=3
)
```

## 7.6 Working with Hugging Face

### Loading Models

```python
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoTokenizer, pipeline
)

# For embeddings
model = AutoModel.from_pretrained("bert-base-uncased")

# For generation
model = AutoModelForCausalLM.from_pretrained("gpt2")

# For classification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)

# Easy inference with pipelines
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
```

### Text Generation

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

output = generator(
    "Once upon a time",
    max_length=100,
    num_return_sequences=1,
    temperature=0.7
)

print(output[0]['generated_text'])
```

### Text Classification

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

result = classifier(
    "I want to book a flight to Paris",
    candidate_labels=["travel", "finance", "food"]
)

print(result['labels'][0])  # "travel"
```

---

# Part 8: Advanced Fine-Tuning Techniques

## 8.1 Full Fine-Tuning

Training all model parameters on task-specific data.

```python
from transformers import (
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
```

## 8.2 Parameter-Efficient Fine-Tuning (PEFT)

### Why PEFT?

- **Memory Efficient**: Only store small adapter weights
- **Fast Training**: Fewer parameters to update
- **Multi-task**: Swap adapters for different tasks
- **Prevents Catastrophic Forgetting**: Base model unchanged

### Adapter Layers

Insert small trainable modules between frozen layers:

```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # Bottleneck architecture
        down = self.activation(self.down_project(x))
        up = self.up_project(down)
        return x + up  # Residual connection
```

## 8.3 LoRA (Low-Rank Adaptation)

### Concept

Instead of updating full weight matrices, add low-rank decomposition:

```
W_new = W_frozen + BA
```

Where:
- W: d×k original weight matrix
- B: d×r matrix (r << d)
- A: r×k matrix
- BA: Low-rank update

### Implementation

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Original frozen weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x):
        # Original forward
        original = F.linear(x, self.weight)
        
        # LoRA forward
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B)
        
        return original + (self.alpha / self.rank) * lora
```

### Using PEFT Library

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=32,          # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply to model
model = get_peft_model(base_model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# "trainable params: 294,912 || all params: 124,734,464 || trainable%: 0.24%"
```

## 8.4 QLoRA (Quantized LoRA)

Combines 4-bit quantization with LoRA for even more efficiency:

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

## 8.5 Instruction Fine-Tuning

### Dataset Format

```python
# Instruction template
template = """### Instruction: {instruction}

### Input: {input}

### Response: {output}"""

# Example
example = {
    "instruction": "Translate the following sentence to French",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
}
```

### Using SFTTrainer

```python
from trl import SFTTrainer

def format_instruction(example):
    return f"""### Instruction: {example['instruction']}

### Input: {example['input']}

### Response: {example['output']}"""

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_instruction,
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="./sft_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10
    )
)

trainer.train()
```

## 8.6 RLHF (Reinforcement Learning from Human Feedback)

### Process

1. **Supervised Fine-Tuning (SFT)**: Initial fine-tuning on high-quality data
2. **Reward Model Training**: Train model to predict human preferences
3. **PPO Training**: Optimize policy using reward model

### Reward Model

```python
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels=1  # Scalar reward
)

# Training on preference pairs: (chosen, rejected)
# Loss: -log(sigmoid(reward_chosen - reward_rejected))
```

### PPO Training

```python
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=256,
    mini_batch_size=16
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer
)

# Training loop
for batch in dataloader:
    # Generate response
    response = model.generate(batch["input_ids"])
    
    # Get reward
    reward = reward_model(response)
    
    # PPO step
    stats = ppo_trainer.step(batch["input_ids"], response, reward)
```

## 8.7 DPO (Direct Preference Optimization)

Simpler alternative to RLHF without explicit reward model:

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,  # KL penalty coefficient
    learning_rate=1e-6
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,  # Contains chosen/rejected pairs
    tokenizer=tokenizer
)

dpo_trainer.train()
```

---

# Part 9: Reinforcement Learning

## 9.1 RL Fundamentals

### Key Concepts

- **Agent**: Learner/decision maker
- **Environment**: What agent interacts with
- **State (s)**: Current situation
- **Action (a)**: What agent can do
- **Reward (r)**: Feedback signal
- **Policy (π)**: Strategy for selecting actions
- **Value Function (V)**: Expected return from state
- **Q-Function (Q)**: Expected return from state-action pair

### Markov Decision Process (MDP)

```
P(s', r | s, a): Probability of next state and reward given current state and action
```

### Bellman Equation

**Value Function**:
```
V(s) = E[R + γV(s')]
```

**Q-Function**:
```
Q(s, a) = E[R + γ max_a' Q(s', a')]
```

## 9.2 Q-Learning

### Algorithm

```python
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}  # Q-table
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = max(Q.get(state, {}), key=Q[state].get, default=0)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Q-update
            old_value = Q.get(state, {}).get(action, 0)
            next_max = max(Q.get(next_state, {}).values(), default=0)
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            
            if state not in Q:
                Q[state] = {}
            Q[state][action] = new_value
            
            state = next_state
    
    return Q
```

## 9.3 Deep Q-Networks (DQN)

### Architecture

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)
```

### Key Innovations

1. **Experience Replay**: Store transitions, sample randomly
2. **Target Network**: Separate network for stable targets
3. **Epsilon Decay**: Decrease exploration over time

### Implementation

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.memory = deque(maxlen=10000)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss and update
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
```

## 9.4 Policy Gradient Methods

### REINFORCE Algorithm

```python
def reinforce(env, policy_net, optimizer, episodes=1000, gamma=0.99):
    for episode in range(episodes):
        log_probs = []
        rewards = []
        
        state = env.reset()
        done = False
        
        while not done:
            # Get action probabilities
            probs = policy_net(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient loss
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

# Part 10: MLOps & Model Deployment

## 10.1 Model Serialization

### Keras/TensorFlow

```python
# Save entire model
model.save('model.keras')

# Save weights only
model.save_weights('weights.h5')

# Load
loaded_model = tf.keras.models.load_model('model.keras')
```

### PyTorch

```python
# Save state dict (recommended)
torch.save(model.state_dict(), 'model.pth')

# Load
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save entire model (includes architecture)
torch.save(model, 'full_model.pth')
```

### ONNX Export

```python
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## 10.2 Model Optimization

### Quantization

```python
# PyTorch dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

# TensorFlow Lite quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Pruning

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)
```

### Knowledge Distillation

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    # Soft targets
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_probs = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

## 10.3 Serving Models

### FastAPI Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = torch.load('model.pth')
model.eval()

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    with torch.no_grad():
        input_tensor = torch.tensor(request.data)
        output = model(input_tensor)
        prediction = output.argmax(dim=1).tolist()
    return {"prediction": prediction}
```

### Hugging Face Inference API

```python
from transformers import pipeline

# Create pipeline
classifier = pipeline("sentiment-analysis", model="./my_fine_tuned_model")

# Serve with Gradio
import gradio as gr

def predict(text):
    result = classifier(text)
    return result[0]['label'], result[0]['score']

gr.Interface(
    fn=predict,
    inputs="text",
    outputs=["text", "number"],
    title="Sentiment Analysis"
).launch()
```

## 10.4 Monitoring & Evaluation

### Key Metrics to Track

1. **Model Performance**: Accuracy, latency, throughput
2. **Data Quality**: Distribution shifts, missing values
3. **System Health**: Memory, CPU/GPU usage
4. **Business Metrics**: User engagement, conversion

### Detecting Model Drift

```python
from scipy.stats import ks_2samp

def detect_drift(reference_data, production_data, threshold=0.05):
    statistic, p_value = ks_2samp(reference_data, production_data)
    
    if p_value < threshold:
        print(f"Drift detected! p-value: {p_value}")
        return True
    return False
```

---

# Part 11: Interview Questions & Answers

## Machine Learning Fundamentals

### Q1: What is the bias-variance tradeoff?
**A**: Bias measures how far model predictions are from true values (underfitting). Variance measures sensitivity to training data fluctuations (overfitting). High bias + low variance = simple model, consistent but wrong. Low bias + high variance = complex model, fits training but poor generalization. The goal is to find the sweet spot minimizing total error.

### Q2: Explain the difference between L1 and L2 regularization.
**A**: 
- **L1 (Lasso)**: Adds |weights| penalty. Produces sparse models, can zero coefficients. Good for feature selection.
- **L2 (Ridge)**: Adds weights² penalty. Shrinks weights but rarely zeros them. Better when all features matter.
- **Elastic Net**: Combines both for balance.

### Q3: How do you handle imbalanced datasets?
**A**:
1. **Resampling**: Oversample minority (SMOTE) or undersample majority
2. **Class weights**: Penalize misclassification of minority more
3. **Different metrics**: Use F1, AUC-ROC instead of accuracy
4. **Algorithmic**: Use algorithms robust to imbalance (XGBoost)
5. **Threshold tuning**: Adjust classification threshold
6. **Focal loss**: Down-weight easy examples

### Q4: What is cross-validation and why use it?
**A**: Cross-validation splits data into k folds, trains on k-1, validates on 1, repeating k times. Benefits:
- More reliable performance estimate
- Uses all data for both training and validation
- Detects overfitting
- Particularly useful for small datasets

## Deep Learning

### Q5: Explain vanishing and exploding gradients.
**A**: 
- **Vanishing**: Gradients become tiny during backprop (especially with sigmoid/tanh), early layers don't learn. Solutions: ReLU, batch norm, residual connections, proper initialization.
- **Exploding**: Gradients grow exponentially, causing NaN values. Solutions: Gradient clipping, batch norm, careful initialization.

### Q6: Why do we use batch normalization?
**A**: Batch norm normalizes layer inputs to zero mean, unit variance. Benefits:
1. Enables higher learning rates
2. Reduces internal covariate shift
3. Acts as regularization
4. Reduces sensitivity to initialization
5. Allows deeper networks to train

### Q7: Compare CNNs and Transformers for vision tasks.
**A**:
- **CNNs**: Local receptive fields, translation equivariance, efficient for images, strong inductive bias for visual patterns
- **ViT**: Global attention from start, requires more data, scales better, more flexible, captures long-range dependencies
- **Hybrid approaches**: Combine CNN features with transformer attention for best of both

### Q8: What is dropout and how does it work?
**A**: Dropout randomly zeros neurons during training with probability p. During inference, all neurons active, outputs scaled by (1-p). It:
- Prevents co-adaptation of neurons
- Acts as ensemble of sub-networks
- Provides regularization without additional parameters
- Typical values: 0.2-0.5

## Transformers & LLMs

### Q9: Explain the self-attention mechanism.
**A**: Self-attention allows each position to attend to all other positions:
1. Create Query (Q), Key (K), Value (V) projections
2. Compute attention scores: softmax(QK^T / √d_k)
3. Weighted sum of values based on scores
4. Multi-head attention runs multiple attention heads in parallel

Benefits: Captures long-range dependencies, parallelizable, no recurrence needed.

### Q10: What is the difference between BERT and GPT?
**A**:
| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional | Causal (left-to-right) |
| Pre-training | MLM + NSP | Next token prediction |
| Best for | Classification, NLU | Generation, completion |
| Context | Full context both directions | Only previous tokens |

### Q11: Explain LoRA and why it's useful.
**A**: LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to frozen pretrained weights: W_new = W + BA where B and A are small matrices. Benefits:
- 99%+ fewer trainable parameters
- Memory efficient (can train large models on consumer GPUs)
- Quick task switching (just swap adapters)
- No inference latency increase (merge weights)
- Preserves pretrained knowledge

### Q12: How does instruction fine-tuning differ from regular fine-tuning?
**A**: 
- **Regular fine-tuning**: Train on task-specific data (input → output)
- **Instruction fine-tuning**: Train on (instruction, input) → output format
- Instruction tuning makes models follow directions, generalizes to new tasks
- Examples: "Summarize this:", "Translate to French:", "Write code that:"

## System Design

### Q13: How would you deploy an LLM for production?
**A**:
1. **Model optimization**: Quantization (INT8/INT4), pruning, distillation
2. **Infrastructure**: GPU servers, load balancing, auto-scaling
3. **Serving**: TensorRT, vLLM, TGI for efficient inference
4. **Caching**: KV cache for generation, semantic caching for common queries
5. **Monitoring**: Latency, throughput, token usage, quality metrics
6. **Safety**: Content filtering, rate limiting, prompt injection protection

### Q14: Describe a typical ML pipeline.
**A**:
1. **Data Collection**: Gather raw data from sources
2. **Data Validation**: Check quality, distributions, schema
3. **Feature Engineering**: Transform raw data into features
4. **Model Training**: Train with hyperparameter tuning
5. **Model Evaluation**: Validate on held-out data
6. **Model Registry**: Version and store models
7. **Deployment**: Serve model via API
8. **Monitoring**: Track performance, detect drift
9. **Retraining**: Update model on new data

### Q15: How do you handle model drift in production?
**A**:
1. **Monitor**: Track prediction distributions, performance metrics
2. **Detect**: Statistical tests (KS test, PSI) comparing training vs production
3. **Alert**: Set thresholds for automatic notifications
4. **Investigate**: Analyze root cause (data change, feature drift, concept drift)
5. **Remediate**: Retrain on new data, update features, or rollback
6. **Automate**: Implement continuous training pipelines

## Coding Questions

### Q16: Implement a basic neural network from scratch.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.01
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
        
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, y, learning_rate=0.01):
        m = y.shape[0]
        y_one_hot = np.eye(self.activations[-1].shape[1])[y]
        
        delta = self.activations[-1] - y_one_hot
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = self.activations[i].T @ delta / m
            db = delta.sum(axis=0, keepdims=True) / m
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
            
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
```

### Q17: Implement attention mechanism.

```python
import torch
import torch.nn.functional as F

def attention(query, key, value, mask=None):
    """
    Scaled dot-product attention
    
    Args:
        query: (batch, seq_len, d_k)
        key: (batch, seq_len, d_k)
        value: (batch, seq_len, d_v)
        mask: optional (batch, seq_len, seq_len)
    
    Returns:
        output: (batch, seq_len, d_v)
        attention_weights: (batch, seq_len, seq_len)
    """
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

---

# Part 12: Quick Reference Cheat Sheets

## Activation Functions Summary

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| ReLU | max(0, x) | [0, ∞) | Hidden layers (default) |
| Leaky ReLU | max(αx, x) | (-∞, ∞) | Prevent dying ReLU |
| Sigmoid | 1/(1+e^-x) | (0, 1) | Binary output |
| Tanh | (e^x-e^-x)/(e^x+e^-x) | (-1, 1) | Hidden layers (RNN) |
| Softmax | e^xi/Σe^xj | (0, 1), sum=1 | Multi-class output |
| GELU | x·Φ(x) | (-0.17, ∞) | Transformers |

## Loss Functions Summary

| Task | Loss Function | When to Use |
|------|---------------|-------------|
| Binary Classification | Binary Cross-Entropy | Two classes |
| Multi-class | Categorical Cross-Entropy | One-hot labels |
| Multi-class | Sparse Categorical CE | Integer labels |
| Regression | MSE | General regression |
| Regression | MAE | Outlier-robust |
| Regression | Huber | Balance MSE/MAE |
| Imbalanced | Focal Loss | Class imbalance |

## Optimizer Comparison

| Optimizer | Key Feature | Best For |
|-----------|-------------|----------|
| SGD | Simple, momentum optional | Large-scale, well-tuned |
| Adam | Adaptive rates + momentum | Default choice |
| AdamW | Adam + weight decay | Transformers |
| RMSprop | Adaptive rates | RNNs |
| LAMB | Layer-wise adaptive | Large batch training |

## Regularization Techniques

| Technique | Effect | Typical Values |
|-----------|--------|----------------|
| Dropout | Random neuron zeroing | 0.2-0.5 |
| L1 (Lasso) | Sparse weights | 1e-4 to 1e-2 |
| L2 (Ridge) | Small weights | 1e-4 to 1e-2 |
| Batch Norm | Normalize activations | After linear layers |
| Early Stopping | Prevent overtraining | patience=5-10 |
| Data Augmentation | Increase data diversity | Task-dependent |

## Common Hyperparameter Ranges

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| Learning Rate | 1e-5 to 1e-2 | Lower for fine-tuning |
| Batch Size | 16, 32, 64, 128 | Larger = more stable |
| Hidden Units | 32-1024 | Depend on task complexity |
| Dropout Rate | 0.1-0.5 | Higher for larger models |
| Weight Decay | 1e-5 to 1e-2 | Regularization strength |
| Warmup Steps | 100-10000 | For LR schedulers |

## Model Architecture Quick Reference

### CNN for Image Classification
```
Input(224x224x3) → Conv(32)×2 → Pool → Conv(64)×2 → Pool → 
Conv(128)×3 → Pool → GlobalAvgPool → Dense(256) → Output
```

### Transformer Block
```
Input → LayerNorm → MultiHeadAttention → Residual → 
LayerNorm → FFN → Residual → Output
```

### Basic LLM Architecture
```
Embedding(vocab) + PositionalEncoding → 
[TransformerBlock] × N → LayerNorm → Linear(vocab)
```

## Framework Code Patterns

### Keras Model Training
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2,
                   callbacks=[EarlyStopping(patience=3)])
```

### PyTorch Training Step
```python
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

### Hugging Face Fine-Tuning
```python
trainer = Trainer(model=model, args=training_args, 
                 train_dataset=train, eval_dataset=val)
trainer.train()
```

---

## Study Tips for AI Engineering Interviews

### 1. **Understand Fundamentals Deeply**
Don't just memorize—understand WHY things work. Be able to explain concepts from first principles.

### 2. **Practice Coding**
Implement algorithms from scratch:
- Neural network forward/backward pass
- Attention mechanism
- Common layers (Conv, BatchNorm, etc.)

### 3. **Know Your Projects**
Be ready to discuss:
- Architecture decisions you made
- Challenges and how you solved them
- Results and metrics
- What you'd do differently

### 4. **Stay Current**
Follow:
- ArXiv papers (especially transformer variants)
- Hugging Face blog
- OpenAI, Google AI, Meta AI research

### 5. **System Design**
Practice designing:
- Recommendation systems
- Real-time ML pipelines
- Scalable inference systems

### 6. **Behavioral Questions**
Prepare stories about:
- Technical challenges overcome
- Team collaboration
- Making tradeoffs under constraints

---

## Final Checklist

Before your interview, ensure you can:

- [ ] Explain gradient descent and backpropagation clearly
- [ ] Describe attention mechanism step by step
- [ ] Compare different optimizers and when to use each
- [ ] Explain transfer learning and fine-tuning strategies
- [ ] Discuss LoRA and parameter-efficient fine-tuning
- [ ] Describe a complete ML pipeline
- [ ] Handle imbalanced datasets
- [ ] Debug common training issues
- [ ] Design a production ML system
- [ ] Write clean PyTorch/TensorFlow code

---

*This guide was compiled based on comprehensive AI engineering coursework covering machine learning, deep learning, computer vision, NLP, transformers, and generative AI. Regular review and hands-on practice with the concepts presented here will prepare you for success in AI engineering interviews.*

**Good luck with your interviews! 🚀**
