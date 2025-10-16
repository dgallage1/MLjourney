# Linear Regression with Gradient Descent — Notes

---

## Overview

What the code is doing overall

This whole program teaches the computer how to draw the best-fit straight line through two data points —
in this case, houses of size 1 (1000 sq ft) and 2 (2000 sq ft) priced at $300 k and $500 k.

It does this automatically using a process called gradient descent, which keeps adjusting the line until the errors between predicted and actual prices are as small as possible.

This code trains a **linear regression model** using **gradient descent**.  
It learns the best values for \( w \) (slope) and \( b \) (intercept) so that the line:

$$
f_{w,b}(x) = w x + b
$$

fits the data points with minimal error.

---

## 1. Importing Libraries

We import:

- `math` → basic math functions  
- `numpy` → handles numerical data (arrays, vectorized math)  
- `matplotlib.pyplot` → for plotting graphs  
- `plt.style.use('default')` → ensures a clean plot style  

---

## 2. Loading Data

We define the training data:

$$
x = [1, 2], \quad y = [300, 500]
$$

- \( x \): house size (in 1000 sqft)  
- \( y \): house price (in thousand dollars)

**Goal:** learn a linear function that predicts \( y \) for any \( x \).

---

## 3. Cost Function \( J(w,b) \)


This function checks how bad the line currently is.
It compares predicted prices with real prices and measures the average squared difference.

If the predictions are close, the cost will be small.
If they’re far off, the cost will be large.

So, the lower the cost, the better your model is performing.

You can think of the cost function as a “score of wrongness.”

### Formula for Cost Function

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} ( f_{w,b}(x^{(i)}) - y^{(i)} )^2
$$

where  

$$
f_{w,b}(x^{(i)}) = w x^{(i)} + b
$$

- \( m \): number of training examples  
- Lower \( J(w,b) \) means a better model fit  

**Purpose:** tells us how well the model is doing overall.

---

## 4. Gradient Function

The gradient function calculates the **slope (direction)** of the cost function with respect to \( w \) and \( b \).  
These values show how much to adjust \( w \) and \( b \) to reduce cost.

### Derivative with respect to \( w \)

$$
\frac{\partial J(w,b)}{\partial w} =
\frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}
$$

### Derivative with respect to \( b \)

$$
\frac{\partial J(w,b)}{\partial b} =
\frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$

**Purpose:** finds how each parameter affects the cost — these gradients guide the updates in gradient descent.

---

## 5. Gradient Descent Function

Gradient descent repeatedly adjusts \( w \) and \( b \) using the gradients above.  
Each iteration moves a small step in the direction that reduces \( J(w,b) \).

### Update Rules

$$
w := w - \alpha \frac{\partial J(w,b)}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J(w,b)}{\partial b}
$$

where  
\( \alpha \) = learning rate (controls step size)

- If \( \alpha \) is too small → learning is slow  
- If \( \alpha \) is too large → cost may diverge (overshoot)

**Purpose:** gradually finds the best \( w \) and \( b \) that minimize cost.

---

## 6. Training (Running Gradient Descent)

We start with:

$$
w = 0, \quad b = 0, \quad \alpha = 0.01
$$

and run for **10,000 iterations**.

Each iteration:
1. Computes gradients  
2. Updates \( w \) and \( b \)  
3. Tracks cost history  

Over time, cost decreases and the model learns the best parameters.

---

## 7. Model Results

After training:

$$
w \approx 200, \quad b \approx 100
$$

So the model becomes:

$$
\hat{y} = 200x + 100
$$

**Interpretation:**
- Each extra 1000 sqft adds \$200k to the price  
- The base (intercept) price is \$100k  

---

## 8. Plotting the Cost Function

A cost vs. iteration graph shows:
- Rapid drop in cost early on (fast learning)
- Gradual flattening (approaching minimum)

This confirms that the model is **converging properly**.

---

## 9. Making Predictions

We use the learned model:

$$
\hat{y} = w x + b
$$

Example predictions:

$$
x = 1.0 \Rightarrow 200(1) + 100 = 300
$$

$$
x = 2.0 \Rightarrow 200(2) + 100 = 500
$$

Predicted prices match the training data, showing that the model fits well.

---

## 10. Testing Learning Rate Effects

If we increase the learning rate too much (e.g., \( \alpha = 0.8 \)):

- Updates become too large  
- Cost function oscillates or diverges  
- Model fails to converge  

This demonstrates why **choosing an appropriate learning rate** is important.

---

## ✅ Summary

- **Cost function** → measures prediction error  
- **Gradients** → tell us which direction to move  
- **Gradient descent** → updates parameters iteratively  
- **Learning rate** → controls how big each update step is  
- **Goal** → minimize \( J(w,b) \) to find the best line of fit  

### Final Equation

$$
\boxed{\hat{y} = w x + b}
$$

with approximately:

$$
w = 200, \quad b = 100
$$
