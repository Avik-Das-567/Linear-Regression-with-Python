# Additional Exercise - Linear Regression with Python

This sub-project is an extension of the main **Linear Regression with Python** project and focuses on applying a "**from-scratch linear regression model**" to a real-world style dataset.
It reinforces both conceptual understanding and practical implementation of supervised learning using pure NumPy, without relying on external machine learning libraries.

Instead of synthetic data, this exercise uses a structured dataset and emphasizes building reusable model code through a standalone Python class.

---

## Objective

The goal of this exercise is to:

- Apply linear regression to a real dataset

- Strengthen understanding of model training mechanics

- Practice separating **model logic** from **experimentation code**

- Visualize training behavior and prediction accuracy

This exercise helps bridge the gap between **toy examples** and **practical data modeling**.

---

## Folder Contents

This sub-repository includes the following components:

- A **Jupyter Notebook** for experimentation and visualization
- A **dataset** file (`chirps.xls`)
- A standalone **Python model implementation** (`linreg.py`)
- Training and prediction **output plots**

---

## Model Architecture

The core model is implemented in a separate Python file as a reusable class called `LinearModel`.

Key components of the model include:

- Random initialization of **weights (W)** and **bias (b)**

- Forward propagation using: $y = b + XW$

- Mean Squared Error (MSE) loss computation

- Manual backpropagation to compute:
  - Gradient of weights (**dW**)
  - Gradient of bias (**db**)

- Gradient descent based parameter updates

All of these are implemented manually using NumPy operations.

---

## Training Process

The training pipeline is built around a custom training loop that performs:

- Forward pass

- Loss calculation

- Backward pass (gradient computation)

- Parameter updates

- Loss tracking per iteration

The model is trained for multiple iterations and prints periodic loss updates to monitor convergence.

---

## Results and Visualizations

### Training Loss Plot :

  ![Training Loss Plot](https://github.com/Avik-Das-567/Linear-Regression-with-Python/blob/main/Additional%20Exercise/images/Training_Loss_Plot.png "Training Loss Plot")

The training loss plot shows a **rapid and smooth decrease in loss values** over time, indicating that:

- Gradients are computed correctly
- Learning rate is stable
- The model successfully minimizes prediction error

This confirms that the optimization pipeline is functioning as expected.

### Predictions vs Actual Values :

  ![Predictions Plot](https://github.com/Avik-Das-567/Linear-Regression-with-Python/blob/main/Additional%20Exercise/images/Predictions_Plot.png "Predictions Plot")

The prediction plot visualizes:
- Actual target values
- Model predicted values

The close clustering of points demonstrates that the trained model learns the relationship between input features and output values effectively.

This validates the correctness of:
- Forward pass logic
- Gradient descent updates
- Model convergence behavior

---

## Dataset Description

This project uses a small Excel dataset (`chirps.xls`) containing two numerical columns:
- **X**: Number of cricket chirps per minute  
- **Y**: Ambient room temperature  

The dataset contains 15 samples and is used to model the linear relationship between chirping frequency and temperature using gradient descent-based optimization.

---

## Tools & Technologies Used

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Key Highlights

- Implemented reusable linear regression model in a standalone Python class
- Applied regression to a real-world inspired dataset
- Visualized training convergence and prediction accuracy

---
