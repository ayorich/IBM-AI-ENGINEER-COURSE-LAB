# Supervised Learning

## Regression

1. **Linear Regression**
   - **Task**: Predicting house prices based on features like the number of rooms and distance to employment centers.
   - **Example**: Estimate the median value of owner-occupied homes in a Boston suburb, focusing on understanding the relationship between house price and number of rooms.

2. **Ridge Regression**
   - **Task**: Predict house prices with better handling of multicollinearity among predictor variables.
   - **Example**: Improve prediction accuracy when features like the number of rooms and property tax rates are highly correlated.

3. **Lasso Regression**
   - **Task**: Predict house prices while performing feature selection.
   - **Example**: Identify the most significant predictors of house price, potentially excluding less impactful features like proximity to the Charles River.

4. **Polynomial Regression**
   - **Task**: Capture non-linear relationships in predicting house prices.
   - **Example**: Model the relationship between house prices and the age of houses, which may not be linear.

5. **Support Vector Regression (SVR)**
   - **Task**: Predict house prices with a focus on minimizing prediction error in high-dimensional space.
   - **Example**: Use a kernel trick to model complex relationships between features and house prices.

6. **Decision Trees**
   - **Task**: Predict house prices via a series of decision rules.
   - **Example**: Determine house prices by recursively splitting data based on features like crime rate and number of rooms.

7. **Random Forest Regression**
   - **Task**: Improve prediction accuracy and stability of house prices.
   - **Example**: Aggregate predictions from multiple decision trees to improve accuracy and reduce overfitting.

8. **Gradient Boosting Machines (GBM)**
   - **Task**: Enhance prediction accuracy by sequentially building models.
   - **Example**: Iteratively refine house price predictions to minimize errors by focusing on previously mispredicted examples.

9. **XGBoost**
   - **Task**: Efficiently predict house prices with high accuracy and speed.
   - **Example**: Use advanced boosting techniques to produce highly accurate house price predictions quickly.

10. **LightGBM**
    - **Task**: Predict house prices using a gradient boosting framework optimized for speed.
    - **Example**: Handle large datasets with missing values gracefully and predict house prices efficiently.

11. **CatBoost**
    - **Task**: Predict house prices with categorical feature support.
    - **Example**: Leverage categorical variables like neighborhood or proximity to highways for better house price predictions.

12. **K-Nearest Neighbors Regression (KNN)**
    - **Task**: Predict house prices based on proximity to similar historical data points.
    - **Example**: Estimate the price of a house by averaging the prices of the most similar houses in the dataset.

13. **Neural Networks (e.g., Multi-layer Perceptrons)**
    - **Task**: Predict house prices using a model capable of learning complex patterns.
    - **Example**: Use neural networks to capture intricate relationships between diverse features like crime rate and number of rooms.

## Classification

1. **Logistic Regression**
   - **Task**: Classify houses as above or below median price.
   - **Example**: Predict whether a house is in the upper price half based on features.

2. **Support Vector Machines (SVM)**
   - **Task**: Classify houses into high or low price categories.
   - **Example**: Use a hyperplane to separate houses into high and low price classes based on features.

3. **Decision Trees**
   - **Task**: Classify houses as expensive or affordable.
   - **Example**: Use decision rules to classify houses based on features like room count and distance to employment centers.

4. **Random Forest**
   - **Task**: Classify houses with increased accuracy and reduced overfitting.
   - **Example**: Aggregate multiple decision trees to classify houses as expensive or affordable.

5. **Gradient Boosting Machines (GBM)**
   - **Task**: Improve classification accuracy of house prices.
   - **Example**: Sequentially build decision trees to classify houses more accurately.

6. **XGBoost**
   - **Task**: Efficiently classify houses into price categories.
   - **Example**: Use advanced boosting techniques for fast and accurate classification.

7. **LightGBM**
   - **Task**: Classify houses quickly using gradient boosting.
   - **Example**: Handle large datasets and classify houses efficiently.

8. **CatBoost**
   - **Task**: Classify houses with categorical feature support.
   - **Example**: Leverage categorical features for better classification accuracy.

9. **K-Nearest Neighbors (KNN)**
   - **Task**: Classify houses based on similarity to others.
   - **Example**: Classify a house as expensive or affordable based on nearby similar houses.

10. **Naive Bayes**
    - **Task**: Classify houses using probabilistic models.
    - **Example**: Predict the probability of a house being expensive based on features.

11. **Neural Networks (e.g., Convolutional Neural Networks, Recurrent Neural Networks)**
    - **Task**: Classify houses using complex pattern recognition.
    - **Example**: Use neural networks to classify houses as expensive or affordable based on multiple features.

12. **AdaBoost**
    - **Task**: Boost the classification performance of a weak learner.
    - **Example**: Combine multiple weak classifiers to improve accuracy in classifying houses.

13. **Bagging**
    - **Task**: Reduce variance in house classification.
    - **Example**: Use ensemble methods to stabilize house classification predictions.

# Unsupervised Learning

## Clustering

1. **K-Means Clustering**
   - **Task**: Group houses into clusters based on feature similarities.
   - **Example**: Identify neighborhoods with similar house characteristics, like price and number of rooms.

2. **Hierarchical Clustering**
   - **Task**: Create a hierarchy of house clusters.
   - **Example**: Build a dendrogram to show how houses group together based on features.

3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
   - **Task**: Identify dense regions of similar houses.
   - **Example**: Discover clusters of houses in densely populated areas with similar prices.

4. **Gaussian Mixture Models**
   - **Task**: Model the distribution of house features and classify them into clusters.
   - **Example**: Assume a Gaussian distribution for house features to group similar houses together.

## Dimensionality Reduction

1. **Principal Component Analysis (PCA)**
   - **Task**: Reduce the dimensionality of the dataset while preserving variance.
   - **Example**: Simplify the Boston Housing dataset by reducing feature space, capturing the most variance with fewer components.

2. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
   - **Task**: Visualize high-dimensional house data in two dimensions.
   - **Example**: Create a 2D visualization of houses to explore hidden patterns in their features.

3. **Linear Discriminant Analysis (LDA)**
   - **Task**: Reduce dimensionality while enhancing class separability.
   - **Example**: Project the house data into a lower-dimensional space to maximize separation between expensive and affordable houses.

4. **Autoencoders**
   - **Task**: Learn efficient representations of house features.
   - **Example**: Use neural networks to encode the essential characteristics of houses into a compressed form and then reconstruct them.
