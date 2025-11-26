# Essential AI Engineering Terms
## Comprehensive Glossary Based on AI Engineer Projects

This document compiles all essential terms and concepts found across your AI engineering projects, organized by category for easy reference.

---

## Table of Contents

1. [Neural Network Fundamentals](#neural-network-fundamentals)
2. [Model Architecture Terms](#model-architecture-terms)
3. [Training & Optimization](#training--optimization)
4. [Regularization Techniques](#regularization-techniques)
5. [Activation Functions](#activation-functions)
6. [Loss Functions](#loss-functions)
7. [Optimizers](#optimizers)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
10. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
11. [Transformers & Vision Transformers](#transformers--vision-transformers)
12. [Advanced Architectures](#advanced-architectures)
13. [Transfer Learning & Fine-tuning](#transfer-learning--fine-tuning)
14. [Reinforcement Learning](#reinforcement-learning)
15. [Machine Learning Algorithms](#machine-learning-algorithms)
16. [Hyperparameter Tuning](#hyperparameter-tuning)
17. [Framework-Specific Terms](#framework-specific-terms)
18. [Data Loading & Management](#data-loading--management)
19. [Model Evaluation & Validation](#model-evaluation--validation)
20. [Common Problems & Solutions](#common-problems--solutions)

---

## Neural Network Fundamentals

### **Neural Network**
A computational model inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers.

### **Neuron/Unit**
A single processing element in a neural network that receives inputs, applies weights and biases, and produces an output through an activation function.

### **Layer**
A collection of neurons that process data at the same level. Types include:
- **Input Layer**: Receives raw data
- **Hidden Layer**: Processes data between input and output
- **Output Layer**: Produces final predictions

### **Weight**
A learnable parameter that determines the strength of connection between neurons. Updated during training via backpropagation.

### **Bias**
A learnable parameter added to the weighted sum of inputs, allowing the model to shift the activation function.

### **Forward Propagation (Forward Pass)**
The process of passing input data through the network layers to generate predictions. Data flows from input → hidden layers → output.

### **Backpropagation (Backward Pass)**
The algorithm that calculates gradients by propagating errors backward through the network, enabling weight updates.

### **Gradient**
The partial derivative of the loss function with respect to each weight. Indicates the direction and magnitude of weight adjustments needed.

### **Gradient Descent**
An optimization algorithm that minimizes the loss function by iteratively moving in the direction of the steepest descent (negative gradient).

### **Stochastic Gradient Descent (SGD)**
A variant of gradient descent that uses a single random sample (or small batch) per iteration, making training faster but noisier.

### **Mini-Batch Gradient Descent**
A compromise between batch and stochastic gradient descent, using a small subset of data (batch) per iteration.

### **Batch Gradient Descent**
Uses the entire dataset to compute gradients in each iteration. More accurate but slower.

---

## Model Architecture Terms

### **Sequential Model**
A linear stack of layers where data flows sequentially from one layer to the next. Common in Keras for simple architectures.

### **Functional API**
A more flexible way to build models in Keras that allows for complex architectures with multiple inputs/outputs and shared layers.

### **Dense Layer (Fully Connected Layer)**
A layer where every neuron is connected to every neuron in the previous layer. Used for final classification/regression.

### **Flatten Layer**
Converts multi-dimensional tensors (e.g., 2D feature maps) into 1D vectors for input to dense layers.

### **Embedding Layer**
Maps discrete categorical inputs to dense vector representations. Common in NLP and recommendation systems.

### **Residual Connection (Skip Connection)**
A connection that adds the input of a layer directly to its output, helping with gradient flow in deep networks.

### **Architecture**
The overall structure and design of a neural network, including layer types, sizes, and connections.

### **Model Depth**
The number of layers in a neural network. Deeper networks can learn more complex patterns but are harder to train.

### **Model Width**
The number of neurons in each layer. Wider networks have more capacity but require more parameters.

---

## Training & Optimization

### **Epoch**
One complete pass through the entire training dataset. Training typically involves multiple epochs.

### **Batch**
A subset of training data processed together in one forward/backward pass. Batch size affects training stability and memory usage.

### **Batch Size**
The number of samples processed in one iteration. Common values: 16, 32, 64, 128.

### **Iteration/Step**
One update of model weights. For a dataset of N samples with batch size B, one epoch = N/B iterations.

### **Learning Rate**
A hyperparameter that controls the step size during weight updates. Too high: unstable training; too low: slow convergence.

### **Learning Rate Schedule**
A strategy to adjust the learning rate during training (e.g., reduce on plateau, exponential decay).

### **Convergence**
The state when the model's loss stops decreasing significantly, indicating training completion.

### **Training Loss**
The error computed on the training dataset. Measures how well the model fits the training data.

### **Validation Loss**
The error computed on a held-out validation set. Indicates generalization performance.

### **Overfitting**
When a model performs well on training data but poorly on new data. The model has memorized training patterns rather than learning generalizable features.

### **Underfitting**
When a model is too simple to capture underlying patterns, performing poorly on both training and validation data.

### **Generalization**
The ability of a model to perform well on unseen data, not just the data it was trained on.

### **Early Stopping**
A regularization technique that stops training when validation loss stops improving, preventing overfitting.

### **Model Checkpointing**
Saving model weights at regular intervals or when validation performance improves, allowing recovery of best models.

---

## Regularization Techniques

### **Dropout**
Randomly sets a fraction of neurons to zero during training, forcing the network to learn redundant representations and preventing overfitting.

### **Batch Normalization**
Normalizes the inputs to each layer by adjusting and scaling activations. Stabilizes training and allows higher learning rates.

### **Layer Normalization**
Similar to batch normalization but normalizes across features rather than batches. Common in transformers.

### **L1 Regularization (Lasso)**
Adds penalty proportional to the absolute value of weights, encouraging sparsity (feature selection).

### **L2 Regularization (Ridge)**
Adds penalty proportional to the square of weights, preventing weights from becoming too large.

### **Elastic Net**
Combines L1 and L2 regularization for balanced feature selection and weight shrinkage.

### **Weight Decay**
A form of L2 regularization that decays weights during optimization.

### **Data Augmentation**
Artificially expands the training dataset by applying transformations (rotation, flipping, scaling) to existing samples.

---

## Activation Functions

### **Activation Function**
A non-linear function applied to neuron outputs, enabling neural networks to learn complex patterns.

### **ReLU (Rectified Linear Unit)**
f(x) = max(0, x). Most common activation function. Fast computation, helps with vanishing gradient problem.

### **Leaky ReLU**
f(x) = max(αx, x) where α is a small positive constant. Addresses the "dying ReLU" problem.

### **Sigmoid**
f(x) = 1 / (1 + e^(-x)). Outputs values between 0 and 1. Used for binary classification outputs.

### **Tanh (Hyperbolic Tangent)**
f(x) = (e^x - e^(-x)) / (e^x + e^(-x)). Outputs values between -1 and 1. Zero-centered version of sigmoid.

### **Softmax**
Converts raw scores (logits) into probability distributions. Used for multi-class classification outputs.

### **Swish**
f(x) = x * sigmoid(x). Self-gated activation function that can outperform ReLU.

### **GELU (Gaussian Error Linear Unit)**
f(x) = x * Φ(x) where Φ is the CDF of standard normal distribution. Used in transformers.

---

## Loss Functions

### **Loss Function (Cost Function)**
A function that measures the difference between predicted and actual values, guiding the training process.

### **Mean Squared Error (MSE)**
Average of squared differences between predictions and targets. Used for regression tasks.

### **Mean Absolute Error (MAE)**
Average of absolute differences. Less sensitive to outliers than MSE.

### **Binary Crossentropy**
Loss function for binary classification. Measures the difference between predicted probabilities and true binary labels.

### **Categorical Crossentropy**
Loss function for multi-class classification with one-hot encoded labels.

### **Sparse Categorical Crossentropy**
Categorical crossentropy for integer-encoded labels (no one-hot encoding needed).

### **Huber Loss**
Combines MSE and MAE, less sensitive to outliers than MSE.

### **Focal Loss**
Addresses class imbalance by down-weighting easy examples and focusing on hard examples.

---

## Optimizers

### **Optimizer**
An algorithm that updates model weights to minimize the loss function.

### **SGD (Stochastic Gradient Descent)**
Basic optimizer that updates weights using gradients. Can be enhanced with momentum.

### **Momentum**
Accumulates gradients from previous steps, helping overcome local minima and speed up convergence.

### **Adam (Adaptive Moment Estimation)**
Combines momentum and adaptive learning rates. Most popular optimizer, works well for most tasks.

### **RMSprop**
Adaptive learning rate optimizer that divides learning rate by exponentially decaying average of squared gradients.

### **AdamW**
Adam with decoupled weight decay, often performs better than Adam.

### **Adagrad**
Adapts learning rate per parameter based on historical gradients. Good for sparse data.

### **Adadelta**
Extension of Adagrad that reduces aggressive learning rate decay.

---

## Evaluation Metrics

### **Accuracy**
Percentage of correct predictions: (TP + TN) / Total. Simple but can be misleading with imbalanced datasets.

### **Precision**
Of positive predictions, how many are correct: TP / (TP + FP). Measures prediction quality.

### **Recall (Sensitivity)**
Of actual positives, how many were found: TP / (TP + FN). Measures coverage.

### **F1-Score**
Harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall). Balanced metric.

### **Confusion Matrix**
A table showing true vs predicted classifications. Shows TP, TN, FP, FN.

### **ROC Curve (Receiver Operating Characteristic)**
Plots true positive rate vs false positive rate at different classification thresholds.

### **AUC (Area Under ROC Curve)**
Measures overall classifier quality. Higher AUC (closer to 1.0) indicates better performance.

### **Precision-Recall Curve**
Plots precision vs recall at different thresholds. Better than ROC for imbalanced datasets.

### **Mean Absolute Error (MAE)**
Average absolute difference between predictions and targets. Regression metric.

### **Root Mean Squared Error (RMSE)**
Square root of average squared differences. Penalizes large errors more than MAE.

### **R² Score (Coefficient of Determination)**
Measures how well the model explains variance in the data. Range: -∞ to 1 (higher is better).

---

## Data Preprocessing & Augmentation

### **Normalization**
Scaling data to a standard range (typically 0-1 or -1 to 1). Helps with training stability.

### **Standardization (Z-score Normalization)**
Transforms data to have mean=0 and std=1. Formula: (x - μ) / σ.

### **One-Hot Encoding**
Converts categorical variables into binary vectors. Each category becomes a binary feature.

### **Label Encoding**
Converts categorical labels to integers. Used for ordinal data or when categories have inherent order.

### **Feature Scaling**
Normalizing or standardizing features to similar ranges, preventing some features from dominating.

### **Data Augmentation**
Techniques to artificially expand training data:
- **Rotation**: Rotate images by various angles
- **Flipping**: Horizontal/vertical flips
- **Scaling**: Zoom in/out
- **Translation**: Shift images
- **Brightness/Contrast**: Adjust image properties
- **Noise Injection**: Add random noise

### **ImageDataGenerator (Keras)**
A utility that generates batches of augmented image data on-the-fly during training.

### **Data Pipeline**
A sequence of data transformations applied before feeding data to the model.

---

## Convolutional Neural Networks (CNNs)

### **Convolutional Layer (Conv2D)**
Applies filters (kernels) to input images to detect features like edges, textures, and patterns.

### **Filter/Kernel**
A small matrix (e.g., 3×3, 5×5) that slides across the input to detect specific features.

### **Stride**
The number of pixels the filter moves each step. Stride=1 moves pixel by pixel; stride=2 moves 2 pixels.

### **Padding**
Adding zeros around image edges to preserve spatial dimensions after convolution. Types:
- **Valid Padding**: No padding, output smaller than input
- **Same Padding**: Padding to keep output same size as input

### **Feature Map**
The output of a convolutional layer, showing where detected features appear in the input.

### **Pooling Layer**
Reduces spatial dimensions while preserving important features. Types:
- **Max Pooling**: Takes maximum value in each region
- **Average Pooling**: Takes average value in each region
- **Global Average Pooling**: Reduces entire feature map to single value

### **Pool Size**
The size of the pooling window (e.g., 2×2, 3×3).

### **Depth/Channels**
The number of feature maps (filters) in a convolutional layer. More channels = more feature detectors.

### **Receptive Field**
The region of input space that affects a particular neuron. Larger receptive fields capture more context.

### **Transpose Convolution (Deconvolution)**
Upsamples feature maps, used in segmentation and generative models.

### **Depthwise Convolution**
Applies a single filter per input channel, reducing parameters compared to standard convolution.

### **Pointwise Convolution**
1×1 convolution used to change the number of channels efficiently.

---

## Transformers & Vision Transformers

### **Transformer**
An attention-based architecture that processes sequences in parallel, revolutionizing NLP and now used in vision.

### **Self-Attention**
Mechanism allowing each position in a sequence to attend to all other positions, capturing relationships.

### **Multi-Head Attention**
Runs multiple attention mechanisms in parallel, each learning different types of relationships.

### **Query (Q), Key (K), Value (V)**
Three matrices in attention mechanism:
- **Query**: "What am I looking for?"
- **Key**: "What do I contain?"
- **Value**: "What information do I provide?"

### **Attention Score**
Measures how much one position should attend to another. Computed as Q·K^T / √d_k.

### **Positional Encoding**
Adds information about position in sequence to embeddings, since transformers have no inherent order.

### **Encoder**
The part of a transformer that processes input sequences and extracts representations.

### **Decoder**
The part of a transformer that generates output sequences based on encoder outputs.

### **Vision Transformer (ViT)**
Applies transformer architecture to images by splitting images into patches and treating them as sequences.

### **Patch Embedding**
Converts image patches into vector representations for input to transformer.

### **CLS Token**
A special classification token added to sequences in transformers, used for final predictions.

### **Layer Normalization**
Normalizes activations across features (not batches), stabilizing transformer training.

### **Feed-Forward Network (FFN)**
A two-layer MLP applied after attention in transformer blocks.

### **Transformer Block**
A complete transformer unit: Multi-head attention → Add & Norm → FFN → Add & Norm.

---

## Advanced Architectures

### **Autoencoder**
An unsupervised learning architecture that learns to compress and reconstruct data. Components:
- **Encoder**: Compresses input to latent representation
- **Decoder**: Reconstructs input from latent representation

### **Variational Autoencoder (VAE)**
An autoencoder that learns a probabilistic latent space, enabling generation of new samples.

### **GAN (Generative Adversarial Network)**
A framework with two competing networks:
- **Generator**: Creates fake data
- **Discriminator**: Distinguishes real from fake data

### **Deep Q-Network (DQN)**
A reinforcement learning algorithm that uses a neural network to approximate Q-values (action values).

### **Q-Learning**
A reinforcement learning algorithm that learns optimal action-selection policies by estimating Q-values.

### **ResNet (Residual Network)**
A CNN architecture using residual connections to enable training of very deep networks (100+ layers).

### **VGG (Visual Geometry Group)**
A deep CNN architecture known for its simplicity and effectiveness, using small 3×3 filters.

### **Hybrid Model**
Combines multiple architectures (e.g., CNN-ViT hybrid) to leverage strengths of each.

---

## Transfer Learning & Fine-tuning

### **Transfer Learning**
Using a model trained on one task as a starting point for a different but related task.

### **Pre-trained Model**
A model trained on a large dataset (e.g., ImageNet) that has learned useful features.

### **Feature Extraction**
Using pre-trained layers as fixed feature extractors, only training new classification layers.

### **Fine-tuning**
Unfreezing and training some layers of a pre-trained model on new data, adapting it to the new task.

### **Frozen Layers**
Layers whose weights are not updated during training. Used in transfer learning to preserve learned features.

### **Trainable Layers**
Layers whose weights are updated during training.

### **Base Model**
The pre-trained model used as the foundation in transfer learning.

### **Top Layers**
The final classification/regression layers of a model, typically replaced in transfer learning.

---

## Reinforcement Learning

### **Reinforcement Learning (RL)**
A learning paradigm where an agent learns to make decisions by interacting with an environment and receiving rewards.

### **Agent**
The learning system that takes actions in an environment.

### **Environment**
The world or system the agent interacts with.

### **State**
The current situation or configuration of the environment.

### **Action**
A decision or move the agent can take.

### **Reward**
Feedback signal indicating how good an action was. The agent's goal is to maximize cumulative reward.

### **Policy**
The strategy the agent uses to select actions. Can be deterministic or stochastic.

### **Q-Value (Action Value)**
The expected cumulative reward of taking an action in a state and following optimal policy thereafter.

### **Replay Buffer**
Stores past experiences (state, action, reward, next state) for training in DQN.

### **Epsilon-Greedy**
An exploration strategy: with probability ε, take random action; otherwise, take best action.

### **Exploration vs Exploitation**
Trade-off between trying new actions (exploration) and using known good actions (exploitation).

---

## Machine Learning Algorithms

### **Supervised Learning**
Learning from labeled data. Types:
- **Classification**: Predicting discrete categories
- **Regression**: Predicting continuous values

### **Unsupervised Learning**
Learning patterns from unlabeled data. Types:
- **Clustering**: Grouping similar data points
- **Dimensionality Reduction**: Reducing feature space

### **Linear Regression**
Models relationship between features and target using a linear equation.

### **Logistic Regression**
Classification algorithm that models probability using logistic function.

### **Decision Tree**
A tree-like model that makes decisions by splitting data based on feature values.

### **Random Forest**
An ensemble method combining multiple decision trees, reducing overfitting.

### **Gradient Boosting**
Sequentially builds models, each correcting errors of previous ones.

### **XGBoost**
Extreme Gradient Boosting, an optimized gradient boosting implementation.

### **LightGBM**
Light Gradient Boosting Machine, faster and more memory-efficient than XGBoost.

### **CatBoost**
Gradient boosting optimized for categorical features.

### **K-Nearest Neighbors (KNN)**
Classifies/predicts based on the k most similar training examples.

### **Support Vector Machine (SVM)**
Finds optimal hyperplane to separate classes, maximizing margin.

### **Naive Bayes**
Probabilistic classifier based on Bayes' theorem with independence assumptions.

### **K-Means Clustering**
Partitions data into k clusters by minimizing within-cluster variance.

### **PCA (Principal Component Analysis)**
Reduces dimensionality by finding principal components that capture most variance.

### **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
Non-linear dimensionality reduction for visualization, preserving local structure.

### **UMAP (Uniform Manifold Approximation and Projection)**
Modern dimensionality reduction technique preserving both local and global structure.

---

## Hyperparameter Tuning

### **Hyperparameter**
A configuration setting not learned during training (e.g., learning rate, batch size, number of layers).

### **Hyperparameter Tuning**
The process of finding optimal hyperparameter values.

### **Grid Search**
Systematically searches through a predefined grid of hyperparameter values.

### **Random Search**
Randomly samples hyperparameter combinations from defined distributions.

### **Bayesian Optimization**
Uses probabilistic models to guide hyperparameter search more efficiently.

### **Keras Tuner**
A library for hyperparameter tuning in Keras, supporting various search strategies.

### **Trial**
A single hyperparameter configuration tested during tuning.

### **Hyperband**
A bandit-based approach that allocates resources to promising hyperparameter configurations.

---

## Framework-Specific Terms

### **Keras**
High-level neural network API, now integrated into TensorFlow.

### **TensorFlow**
Google's open-source machine learning framework.

### **PyTorch**
Facebook's deep learning framework, known for dynamic computation graphs.

### **Tensor**
A multi-dimensional array, the fundamental data structure in deep learning frameworks.

### **Computation Graph**
A directed graph representing mathematical operations. Can be:
- **Static** (TensorFlow 1.x): Defined before execution
- **Dynamic** (PyTorch, TensorFlow 2.x): Built during execution

### **Eager Execution**
Immediate execution of operations (PyTorch default, TensorFlow 2.x default).

### **Model Compilation**
In Keras, configuring the model with optimizer, loss, and metrics before training.

### **Model Fitting**
The process of training a model on data.

### **Callback**
A function executed at specific points during training (e.g., after each epoch).

### **Model Checkpoint**
Saving model state (weights, optimizer state) to disk.

### **State Dict (PyTorch)**
A dictionary containing model parameters (weights and biases).

### **Device**
The hardware where computation occurs (CPU, GPU, TPU).

### **CUDA**
NVIDIA's parallel computing platform for GPU acceleration.

---

## Data Loading & Management

### **DataLoader (PyTorch)**
A utility that provides batches of data, handles shuffling, and multi-process data loading.

### **Dataset (PyTorch)**
An abstract class representing a dataset, must implement `__len__` and `__getitem__`.

### **ImageDataGenerator (Keras)**
Generates batches of augmented image data with real-time data augmentation.

### **Data Generator**
A Python generator that yields batches of data, useful for large datasets that don't fit in memory.

### **Memory-Based Loading**
Loading entire dataset into memory before training. Fast but memory-intensive.

### **Generator-Based Loading**
Loading data on-the-fly during training. Memory-efficient but slower.

### **Data Pipeline**
A sequence of data transformations and loading steps.

### **Transform**
A function applied to data (e.g., normalization, augmentation) before feeding to model.

### **Data Splitting**
Dividing data into:
- **Training Set**: Used to train the model
- **Validation Set**: Used to tune hyperparameters and monitor training
- **Test Set**: Used for final evaluation (only touched once)

### **Stratified Split**
Maintains class distribution when splitting data, important for imbalanced datasets.

---

## Model Evaluation & Validation

### **Cross-Validation**
Splitting data into k folds, training on k-1 folds and validating on remaining fold. Repeats k times.

### **K-Fold Cross-Validation**
Dividing data into k equal parts, using each part as validation set once.

### **Holdout Validation**
Simple train/validation/test split. Most common approach.

### **Train-Test Split**
Dividing data into training and testing sets.

### **Validation Set**
A held-out set used during training to monitor performance and prevent overfitting.

### **Test Set**
A held-out set used only for final evaluation, never used during training or tuning.

### **Baseline**
A simple model or heuristic used as a reference point to compare against.

### **Model Comparison**
Evaluating multiple models to select the best one.

### **Performance Metrics**
Quantitative measures of model performance (accuracy, precision, recall, etc.).

---

## Common Problems & Solutions

### **Vanishing Gradient Problem**
Gradients become extremely small in deep networks, preventing weight updates in early layers. Solutions:
- ReLU activation
- Batch normalization
- Residual connections
- Proper weight initialization

### **Exploding Gradient Problem**
Gradients become extremely large, causing unstable training. Solutions:
- Gradient clipping
- Proper weight initialization
- Batch normalization

### **Weight Initialization**
Setting initial weights before training. Methods:
- **Random**: Random small values
- **Xavier/Glorot**: For tanh/sigmoid activations
- **He Initialization**: For ReLU activations
- **Zero Initialization**: Usually avoided (causes symmetry)

### **Class Imbalance**
When one class has significantly more samples than others. Solutions:
- Class weights
- Oversampling minority class
- Undersampling majority class
- SMOTE (Synthetic Minority Oversampling)
- Focal loss

### **Data Leakage**
When information from test set leaks into training, causing overly optimistic results.

### **Mode Collapse (GANs)**
When generator produces limited variety of samples, failing to capture full data distribution.

### **Training Instability**
Model loss becomes NaN or training becomes erratic. Causes:
- Learning rate too high
- Poor weight initialization
- Numerical instability

### **Overfitting Solutions**
- Increase training data
- Data augmentation
- Dropout
- Regularization (L1/L2)
- Reduce model complexity
- Early stopping

### **Underfitting Solutions**
- Increase model complexity
- Train longer (more epochs)
- Reduce regularization
- Feature engineering
- Better architecture

---

## Additional Important Terms

### **Ensemble**
Combining predictions from multiple models to improve performance.

### **Bagging**
Training multiple models on different subsets of data and averaging predictions.

### **Boosting**
Sequentially training models, each focusing on previous errors.

### **Stacking**
Training a meta-model on predictions from base models.

### **Feature Engineering**
Creating new features from existing data to improve model performance.

### **Feature Selection**
Choosing the most relevant features for the model.

### **Dimensionality Reduction**
Reducing the number of features while preserving important information.

### **One-Hot Encoding**
Converting categorical variables to binary vectors.

### **Label Encoding**
Converting categorical labels to integers.

### **Imputation**
Filling missing values in data.

### **Outlier Detection**
Identifying and handling anomalous data points.

### **Normalization vs Standardization**
- **Normalization**: Scaling to [0,1] range
- **Standardization**: Scaling to mean=0, std=1

### **Model Serialization**
Saving trained models to disk for later use.

### **Model Deployment**
Making trained models available for production use.

### **Inference**
Using a trained model to make predictions on new data.

### **Batch Inference**
Processing multiple samples simultaneously for efficiency.

### **Real-time Inference**
Processing samples one at a time as they arrive.

---

## Quick Reference: Common Code Patterns

### **Keras Model Definition**
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

### **PyTorch Model Definition**
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### **Model Compilation (Keras)**
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### **Model Training (Keras)**
```python
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, model_checkpoint]
)
```

### **Model Training (PyTorch)**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Summary

This glossary covers essential terms from:
- **Deep Learning Fundamentals**: Neural networks, training, optimization
- **CNN Architecture**: Convolution, pooling, feature extraction
- **Transformers**: Attention, self-attention, ViT
- **Advanced Models**: GANs, Autoencoders, DQN
- **Transfer Learning**: Pre-trained models, fine-tuning
- **Machine Learning**: Traditional algorithms, evaluation
- **Data Management**: Loading, preprocessing, augmentation
- **Framework Terms**: Keras, PyTorch, TensorFlow specifics

**Key Takeaways:**
1. Understanding these terms is essential for AI engineering
2. Many terms are interconnected (e.g., dropout prevents overfitting)
3. Framework-specific terms may differ but concepts are similar
4. Practice implementing these concepts to truly understand them

---

*This document was generated by analyzing all projects in the AI Engineer folder. Keep it as a reference guide while working on AI/ML projects.*

