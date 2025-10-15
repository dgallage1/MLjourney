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

(x<sup>(i)</sup>, y<sup>(i)</sup>) -> i<sup>(th) training example -> NOT sqaured number
eg x<sup>(1), y<sup>(1) -> (2104,400)

## Linear regression model 2

training set with features and targets -> supervised learning algorithm -> produces function **f**, takes in input x that goes into f - model, and output y-hat which is the prediction for y (esmtimated y) where y is the target. 

House example: give size as an input goes into model f and outputs an estimated price

**How to represent the function f?**

f<sub>w,b</sub>(x) = w





