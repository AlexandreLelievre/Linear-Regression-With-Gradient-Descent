# Linear Regression with Gradient Descent

This project implements a **Python class** for **linear regression** using **gradient descent** as the optimization algorithm. The class is designed to fit a linear model to a dataset, minimize the cost function, and make predictions. It also includes **regularization** techniques to prevent overfitting.

## Table of Contents
- [Overview](#overview)
- [Mathematical Concepts](#mathematical-concepts)
  - [Linear Regression](#linear-regression)
  - [Gradient Descent](#gradient-descent)
  - [Regularization](#regularization)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

The goal of this project is to create a **Python class** that performs **linear regression** using **gradient descent**. The class handles feature normalization, computes the cost function, updates the model parameters using gradient descent, and supports regularization methods like L1 and L2.

This project aims to provide a clear and concise implementation of the basic mathematical concepts behind linear regression, and it's also a good starting point for anyone looking to understand and experiment with optimization algorithms like gradient descent.

## Mathematical Concepts

### Linear Regression

Linear regression is a method used to model the relationship between a dependent variable \( y \) and one or more independent variables \( X \). In simple linear regression, the relationship is modeled as:

\[
y = \theta_0 + \theta_1 X
\]

Where:
- \( y \) is the predicted output,
- \( \theta_0 \) is the intercept,
- \( \theta_1 \) is the slope (or coefficient) of the independent variable \( X \).

The goal is to find the parameters \( \theta_0 \) and \( \theta_1 \) that minimize the difference between the predicted \( y \) and the actual target values.

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function. The cost function for linear regression (Mean Squared Error) is:

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(X^{(i)}) - y^{(i)} \right)^2
\]

Where:
- \( m \) is the number of training examples,
- \( h_\theta(X) \) is the prediction (hypothesis) using the current parameters \( \theta \),
- \( y \) is the actual target value.

Gradient descent works by updating the model parameters in the direction that reduces the cost function:

\[
\theta_j = \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
\]

Where:
- \( \alpha \) is the learning rate (step size),
- \( \frac{\partial J(\theta)}{\partial \theta_j} \) is the gradient of the cost function with respect to \( \theta_j \).

### Regularization

To prevent overfitting, regularization techniques are introduced, such as:
- **L2 regularization (Ridge)**: Adds a penalty proportional to the sum of the squared values of the parameters.
  
  \[
  J(\theta) = \text{MSE} + \lambda \sum_{j=1}^{n} \theta_j^2
  \]

- **L1 regularization (Lasso)**: Adds a penalty proportional to the absolute values of the parameters.

  \[
  J(\theta) = \text{MSE} + \lambda \sum_{j=1}^{n} |\theta_j|
  \]

In this implementation, a combination of L1 and L2 regularization is supported using the **l1_ratio** parameter.

## Features

- Linear regression model using gradient descent.
- Normalization of features and target values.
- Regularization: L1 (Lasso), L2 (Ridge), or a combination.
- Visualization of the cost function evolution.
- Ability to make predictions on new data.

## Installation

To use this project, you need to have **Python 3** installed along with the following packages:

```bash
pip install numpy matplotlib

