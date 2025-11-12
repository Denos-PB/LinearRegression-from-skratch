# Linear Regression from Scratch – Project Overview

This project demonstrates a **fully functional linear regression model built entirely from scratch** using only NumPy—no high-level libraries like scikit-learn or TensorFlow.

##  What It Shows

- **Core ML Understanding**: Implements gradient descent, loss computation, and parameter updates manually—proving deep knowledge of how linear regression actually works under the hood.
- **Real-World Ready Features**:
  - Supports **L1 (Lasso)** and **L2 (Ridge)** regularization to prevent overfitting
  - Includes **early stopping** when the model converges
  - Uses efficient **vectorized operations** (no Python loops over samples or features)
- **Production-Quality Code**: Clean, well-structured, and includes input validation and a built-in scoring method (`score()`).

## Performance

- Trained and evaluated on synthetic regression data (1,000 samples, 10 features)
- Achieves **strong R² scores** comparable to scikit-learn’s implementation
- Includes visualizations (actual vs. predicted plots) to validate model behavior

## Why It Matters

Instead of just calling `sklearn.linear_model.LinearRegression()`, this project proves the ability to **build, debug, and optimize ML algorithms from first principles**—a critical skill for roles involving custom model development, research, or performance-critical applications.

> **Built with**: Python, NumPy, Matplotlib  
> **No external ML libraries used** — just pure math and code.
