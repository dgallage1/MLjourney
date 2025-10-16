# ML Notes - Supervised Machine Learning : Regression and Classification
# Week One

## What is machine learning 

Field of study that gives computers the ability to learn without being explicity programmed.
eg: checkers game - computer played against themselves to build experience 
more games the better.

## Two types of ML algorithms

Course 1, 2:

**Supervised Learning** - used most in real world applications, most rapid advancements and innovation

Course 3: 

**Unsupervied Learning** 
**recommended systems**
**reinforcement learning**

### supervised learning 1 - regression

algorithms that learn 
Input -> Desired output mapping 
X -> Y

you give the learning algo examples that includes the correct label Y for given input X, seeing correct pairs of Input X and desired output label that the learning algo takes just input alone with output label and gives reasonablly accurate prediction of output.

Input (X) ->    Output (Y)      Application
email           spam? (0/1)     spam filtering 
audio           text transcipts speech recognition 
English         spanish         machine translation
ad              click (0/1)     online advertising
image,radar     position        self driving car
image of phone  defect (0/1)    visual inspection 

In all applications you would first train model with examples of input X and the right answers Label Y. After model learns this they will take brand new input X to the corresponding output Y 

**Example question**  - Regression : Housing price prediction 
I want to predict housing prices based on the size of the house in feet 
y axis - prices in 1000's, x axis is house size in feet 

could draw a straight line through the data but there could be a better solution
how about a curved line does that fit better 
need to find what the best way of fitting the data and that is by finding an algorithm

this is a good exmaple of supervised learning : gave the algo a data set in which the right answer that is the label or the correct price y is given for every house on the plot.vThe task of the learning algorithm is to produce more of these right answers. 

This housing price prediction is the particular type of supervised learning called regression - predict number from infinitely many possible outcomes

### supervised learning 2 - classification 

**Examples question** - Breast cancer detection
Doctors need a tool to diagnose cancer - check to see if the lump is malignant (cancerous - 1 ) or benign (non  - 0)
Tumours of different sizes need to classified into the two categories 

plot axis: 
y axis is either 0 or 1 to see if it is cancerous or benign
x axis is the tumour size 

only small finite number of possible categories as the outcome different from regression that has so many different possibilities 

so after all the points are plotted if a new patient comes with a certain tumour size between two exisiting results they ask to classfiy if you think its benign or malignant?

- can have two possible outcomes 
- even the malignant can have type one or type two 
- output classes and output categories mean the same thing 

Predict categories - can be non numeric eg: whether a tumor is benign or malignant 

now let's say we have two or more inputs : tumor size and age, if we plot them against each other and break it into circles for benign and X for malignant then have more of a higher chance to see a better value and can even have a boundary line to break the two categories 

**unsupervised learning**

Data only comes with inputs x, but not output labels y. Algorithm has to find structure in the data. 

### unsupervised learning 1 - clustering

Takes data without labels and try to automatically group the, into clusters. Find some strcuture or pattern or just find something interesting in the data - not trying to supervise the algorithm. 

**example : clustering in google news** : Giant panda gives birth to rare twin cubs at Japan's oldest zoo - is the main headline, the articles under are related to it with similar words. The algorithm has to figure out own its own which cluster or articles everyday to put there.

**example : clustering DNA microarray** : group into three types of categoires, not telling algorithm in advance that there is a type 1 person with certain characteristics etc. All we are doing is giving the data and get the model automatically find a strcture in the data and see if there are any major types for the individuals.

**example 3: grouping customers - clustering** : Companies have huge databases of customer information, can you automatically group customers into different market segments to more efficiently serve customers.  

### unsupervised learning 2 - Anomaly detection

Anomaly detection : Find unusual data points. Good for fraud detection, unusual transactions could mean a sign of fraud. 

### unsupervised learning 2 - Dimenstionality reduction

Compress data using fewer numbers. Take a big data set and compress it to smaller dataset without loosing as little information as possible. 


### jupyter notebook 

Used to code up and experiment to try things out.

- use shift + enter to exit out of an edit mode back to like a viewing mode 
- double click into box to access and edit the text 
- different types of cells - markdown for like writing and text and code cell for code
- can just add cell blocks above or below previous ones
- use hashtags for titles and things like this 
- press d twice fast to delete a code block
- pretty easy to work around 

## Linear regression model 1

**Example question**  - linear Regression : Housing price prediction 
I want to predict housing prices based on the size of the house in feet 
y axis - prices in 1000's, x axis is house size in feet 

build a linear regression model will fit a straight line to the data.
- supervised learning model : first training model with the right answers
- regression model becuase it predicts numbers as the output
- infintie many possible outputs 

different to classifcation model - predicts the categories - small finite number of outputs 

##### Terminology 

**Training set** -> data set used to train the model 
input variable -> **input feature** -> x
output variable -> **target variable** -> y 
**m** -> total number of training samples 
(x,y) -> single training example 

## Training Example Notation

Each training example is represented as a pair:

(x<sub>i</sub>, y<sub>i</sub>)

For example, the first training example is:

(x<sub>1</sub>, y<sub>1</sub>) = (2104, 400)

Here:
- **x<sub>i</sub>** represents the input features (e.g., house size in square feet)
- **y<sub>i</sub>** represents the corresponding output value (e.g., house price in $1000s)
- **i** is the index of the training example (1 ≤ i ≤ m)


## Linear regression model 2

training set with features and targets -> supervised learning algorithm -> produces function **f**, takes in input x that goes into f - model, and output y-hat which is the prediction for y (esmtimated y) where y is the target. 

House example: give size as an input goes into model f and outputs an estimated price

**How to represent the function f?**

f<sub>w,b</sub>(x) = wx + b 
f<sub>w,b</sub>(x) -> f(x)

Linear regression with one variable (single feature x) -> Univariate linear regression
**f(x) = wx + b**

## Notation
Here is a summary of some of the notation you will encounter.  

| General <img width=70/> <br /> Notation <img width=70/> | Description <img width=350/> | Python (if applicable) |
|:--------------------------|:------------------------------------------------------------|:------------------------|
| $a$ | Scalar, non-bold |  |
| $\mathbf{a}$ | Vector, bold |  |
| **Regression** |  |  |
| $\mathbf{x}$ | Training example feature values (in this lab – Size (1000 sqft)) | `x_train` |
| $\mathbf{y}$ | Training example targets (in this lab – Price (1000s of dollars)) | `y_train` |
| $(x_i, y_i)$ | $i$-th training example | `x_i`, `y_i` |
| $m$ | Number of training examples | `m` |
| $w$ | Parameter: weight | `w` |
| $b$ | Parameter: bias | `b` |
| $f_{w,b}(x_i)$ | Model output at $x_i$ parameterized by $w,b$: $f_{w,b}(x_i) = wx_i + b$ | `f_wb` |


Common tools 
- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data

### Understanding `x_train.shape[1]` in a Machine Learning Context

```python
import numpy as np

# 1D array example (only one dimension)
x_train = np.array([1.0, 2.0])
print(x_train.shape)      # (2,)
# This has only one dimension → 2 elements total
# There is no second dimension, so the next line will raise an error:
# print(x_train.shape[1])  ❌ IndexError: tuple index out of range
```

Explanation:

For a 1D array like x_train = [1.0, 2.0], x_train.shape returns (2,), meaning one dimension with two values.

Since the tuple (2,) has only one element, trying to access x_train.shape[1] fails because there is no second dimension. x_train[0] = 2, 1d with 2 elements 

NumPy doesn’t treat this as a table, just a single list of data points.


### Cost Function 

For the model : f<sub>w,b</sub>(x) = wx + b 
w, b are the parameters of the model - variables that we can change or adjust to improve the model 
They can also be known as coeffcients or weights 

**How to find values for w and b?** so that y hat i is close to y i for all values 

### Cost function : squared error cost function

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

- takes prediction y hat and predicts it to the target y -> **error** (how far off prediction from the target)
- then do the square of the error
-  compute different training examples for i
- sum of the sqaured errors from i =1 to however many training examples we have (m) 
- m is the number of training examples, if more examples m is larger, cost function will be summing over several more examples.
- so divide it to get the average sqaured error by dividing by m, for ML divide by 2 to make later calculations look neater 
- Capital J(w,b) -> the cost function output

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \\[6pt]
\begin{aligned}
f_{w,b}(x^{(i)}) &= w x^{(i)} + b : \text{ predicted value (hypothesis)} \\
y^{(i)} &= \text{ true target value} \\
m &= \text{ number of training examples}
\end{aligned}
$$

#### Goal
Find \( w \) and \( b \) that minimize the cost function:

$$
\min_{w,\,b} J(w, b)
$$
**Example simplified:**

$$
f_{w}(x) = w x
$$

- there is no y intercept 
- cost function below is simplified to this 

$$
J(w) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w}(x^{(i)}) - y^{(i)})^2
$$

$$
\min_{w} J(w)
$$


for this, when w is fixed (always constant value) then fw is only a function on input x. 
$$
f_{w}(x) = w x
$$

### Cost Function Example (No Intercept)

Given data points:  
\[
(1,1),\ (2,2),\ (3,3)
\]

Model:  
\[
f_w(x) = w x
\]

Cost function:  
$$
J(w) = \frac{1}{2m} \sum_{i=1}^{m} (f_w(x^{(i)}) - y^{(i)})^2
$$
Substitute \( f_w(x^{(i)}) = w x^{(i)} \) and \( m = 3 \):  
$$
J(w) = \frac{1}{6}\left[(w(1)-1)^2 + (w(2)-2)^2 + (w(3)-3)^2\right]
$$


### When \( w = 1 \)

$$
J(1) = \frac{1}{6}\left[(1(1)-1)^2 + (1(2)-2)^2 + (1(3)-3)^2\right]
$$

$$
J(1) = \frac{1}{6}\left[(1-1)^2 + (2-2)^2 + (3-3)^2\right]
$$

$$
J(1) = \frac{1}{6}[0 + 0 + 0] = 0
$$


**Result:**  
When \( w = 1 \), every term in the brackets equals zero because the line \( y = x \) passes exactly through all data points.  
Therefore, the cost function is:

$$
\boxed{J(1) = 0}
$$

For different values of w corresponds to a different straight line fit f(x), you can trace out what the cost function J(w) looks like and from our example we can see that when w = 1 that is the minimum point in the curved line. 

**how to choose vlaue of w** - to minimise J to be as small as possible, minimises sqaured errors as much as possible -> minimise the cost 

### visualising the cost function 

3d can be used with contours to show 2d models

### Gradient Descent 

Have cost function J(w,b) -> we want to minimise this for linear regression but can use gradient descent for anything. Work with cost function that works with more than two parameters. 

Outline 
- start setting w and b to zero (common)
- Keep changing w and b to reduce J(w,b)
- Until we settle at or near a minimum 
- for some functions of J there may have more than one minimum 

linear regression owuld always be a bow shaped or hammock shaped visual, also the example is not sqaured error cost. Think of it like a golf course, high points are mountains, low points are valleys. Goal is to start at the top and get to bottom as most efficiently as possible. Look around 360 and go downhill as quikc as possible to a valley, look for the steepest descent with a little step. Then again after moving a step do another 360 degree to see which way would get the steepest and you get a new path. Keep doing steps till your at the bottom of the valley. We just went through multiple steps of gradient descent. Now try again use a different starting point a couple steps to the right of the original, repeat gradient descent process and you end up in another valley different to the first time - you get to another local minimum. 

### Implementing gradient descent algorithm

### Gradient Descent Formula

The general update rule for gradient descent is:

$$
w = w - \alpha \frac{\partial}{\partial w} J(w, b)
$$

assingment code
- equal sign is the assignment operator
- a = c -> take value c and store in computers variable a 
- alpha is the learning rate : controls how big of a step you take down the hill, if alpha is too large you are trying to take huge steps down the hill.
- derivative team of the cost function J

### Gradient Descent Formula for Bias


The update rule for the bias term is:

$$
b = b - \alpha \frac{\partial}{\partial b} J(w, b)
$$

**Where:**
- `b` → bias parameter  
- `α` → learning rate  
- `∂/∂b J(w, b)` → derivative (gradient) of the cost function with respect to `b`

- Two parameters w and b need to be updated simultaneously. 

**example question** Correct : Simultaneous update

$$
tempw = w - \alpha \frac{\partial}{\partial w} J(w, b)
$$

$$
tempb = b - \alpha \frac{\partial}{\partial b} J(w, b)
$$

where:
- `w` = weight parameter  
- `b` = bias parameter  
- `alpha` = learning rate  
- `∂J(w, b)/∂w` = partial derivative of the cost function with respect to `w`  
- `∂J(w, b)/∂b` = partial derivative of the cost function with respect to `b`

w = tempw 
b = tempb

- make sure to update both at the same time not one after the other
- repeat until convergence

### gradient descent intuition J(w) visualised

- take a point w on the curve and draw a tangent line to the curve on that point
- eg: 2/1 -> gradient, line go up to the right - slope is a positive and the derivative part of that formulae is greater than zero 
- w = w - alpha * (postive number) 
- learning rate is always positive
- w minus a positive number new value willl mean that w will be assigned a smaller value and moves to the left 

- if it is a negative slope
- w = w - alpha*(negative number)
- w = w--above -> w will be assigned a higher number so will move to the right to get closer to the minimum 

### Learning rate

If alpha is too small - eg: 0.000001 then you only take really small steps the learning rate is so small and the outcome of the process is you do reduce the cost J but the process is very small. Gradient descent is too slow.

If alpha is too large  - but its already pretty close to the minimum already, it increases and the step might go way higher than the minimum and keep overshooting. Gradient descent may overshoot and never reach minimum, it may fail to converge and my even diverge. 

**example question** : if w is already at the local minimum. So the derivative term is equal to zero so w is updated to w - alpha * 0 which is just w = w. It won't change the parameters as it will keep it at local minimum 

- can reach local minimum with fixed learning rate alpha 
- large to not as alrge to smaller and automatically take smaller and smaller steps, aproaching a local minimum the update steps become smaller, can reach minimum without decreasing learning rate. 

### Gradient descent for linear regression 

### Derivative of Cost Function with respect to \( w \)

$$
\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}
$$

### Derivative of Cost Function with respect to \( b \)

$$
\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$





