"""Simple feature engineering and optimization helpers for Unit IV."""


# Makes one new column for each category found in the chosen column.
def make_category_columns(data, column):
    from pandas import get_dummies

    return get_dummies(data, columns=[column])


# Changes number columns to a similar scale so large values do not dominate.
def scale_numbers(data):
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError(
            "Install scikit-learn first: python -m pip install scikit-learn"
        ) from error
    scaler = StandardScaler()
    return scaler.fit_transform(data)


# Finds the best straight line through data using least squares.
def fit_linear_model(features, answers):
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError as error:
        raise ImportError(
            "Install scikit-learn first: python -m pip install scikit-learn"
        ) from error
    model = LinearRegression()
    model.fit(features, answers)
    return model


# Repeatedly adjusts a number to get closer to a minimum.
# The gradient function should return the slope at the current number.
def gradient_descent(gradient, start, step_size=0.1, steps=100):
    number = start
    for _ in range(steps):
        number = number - step_size * gradient(number)
    return number

