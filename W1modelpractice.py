# Making a model to predict house prices based on size using linear regression

H1Size = 1000
H2Size = 2000
H1Price = 300
H2Price = 500

# Function to generate linear regression model parameters (w, b)

def generate_linear_regression_model(size1, size2, price1, price2):
    w = (price2 - price1) / (size2 - size1)
    b = price1 - w * size1
    return w, b

# fucntion to generate the model
generate_linear_regression_model(H1Size, H2Size, H1Price, H2Price)
w, b = generate_linear_regression_model(H1Size, H2Size, H1Price, H2Price)
print("w =", w)
print("b =", b)

# Function to predict cost based on the model parameters (w, b)
def predict_cost(new_size, w, b):
    """
    Predicts cost using the linear model parameters (w, b).
    """
    predicted = w * new_size + b
    print(f"For a house size of {new_size}, predicted cost = {predicted:.2f}")
    return predicted

# Predicting costs for new house sizes
predict_cost(1500, w, b)
predict_cost(2500, w, b)