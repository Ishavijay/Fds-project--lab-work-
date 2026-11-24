# Unit II: basic probability and statistics calculations.
# Functions cover averages, probability, correlation, and normal distributions.

import math


def average(numbers):
    # Return the arithmetic mean of the numbers.
    if not numbers:
        raise ValueError("Give at least one number.")

    total = 0
    count = 0
    for number in numbers:
        total += number
        count += 1

    return total / count


def conditional_probability(chance_of_both, chance_of_b):
    # Return P(A given B) from P(A and B) and P(B).
    if chance_of_b <= 0:
        raise ValueError("The chance of B must be greater than zero.")

    return chance_of_both / chance_of_b


def correlation(first, second):
    # Return the Pearson correlation between two equally sized number lists.
    if len(first) != len(second) or len(first) < 2:
        raise ValueError(
            "Give two lists with the same number of at least two values."
        )

    first_mean = average(first)
    second_mean = average(second)
    covariance_total = 0
    first_square_total = 0
    second_square_total = 0

    for first_value, second_value in zip(first, second):
        first_difference = first_value - first_mean
        second_difference = second_value - second_mean
        covariance_total += first_difference * second_difference
        first_square_total += first_difference**2
        second_square_total += second_difference**2

    denominator = math.sqrt(first_square_total * second_square_total)
    if denominator == 0:
        raise ValueError("Both lists need some variation in their values.")

    return covariance_total / denominator


def normal_probability(value, mean=0, standard_deviation=1):
    # Return the probability that a normal value is at most the given value.
    if standard_deviation <= 0:
        raise ValueError("Standard deviation must be greater than zero.")

    # Approximate the normal CDF with a polynomial and exponential calculation.
    z = (value - mean) / standard_deviation
    sign = 1 if z >= 0 else -1
    absolute_z = abs(z) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * absolute_z)
    coefficients = (
        1.061405429,
        -1.453152027,
        1.421413741,
        -0.284496736,
        0.254829592,
    )

    polynomial = coefficients[0]
    for coefficient in coefficients[1:]:
        polynomial = polynomial * t + coefficient

    erf_value = 1 - polynomial * t * math.exp(-(absolute_z**2))

    return (1 + sign * erf_value) / 2

