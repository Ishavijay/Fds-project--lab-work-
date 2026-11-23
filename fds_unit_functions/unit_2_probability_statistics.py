"""Simple probability and statistics helpers for Unit II."""

import math
import statistics


# Finds the average of a list of numbers.
def average(numbers):
    if not numbers:
        raise ValueError("Give at least one number.")
    return sum(numbers) / len(numbers)


# Finds the chance of A happening when B has already happened.
def conditional_probability(chance_of_both, chance_of_b):
    if chance_of_b <= 0:
        raise ValueError("The chance of B must be greater than zero.")
    return chance_of_both / chance_of_b


# Finds how closely two lists of numbers move together.
def correlation(first, second):
    if len(first) != len(second) or len(first) < 2:
        raise ValueError(
            "Give two lists with the same number of at least two values."
        )
    first_sd = statistics.stdev(first)
    second_sd = statistics.stdev(second)
    if first_sd == 0 or second_sd == 0:
        raise ValueError(
            "Both lists need some variation in their values."
        )
    return statistics.correlation(first, second)


# Finds the probability that a normal value is below the given value.
def normal_probability(value, mean=0, standard_deviation=1):
    if standard_deviation <= 0:
        raise ValueError("Standard deviation must be greater than zero.")
    z = (value - mean) / (standard_deviation * math.sqrt(2))
    return (1 + math.erf(z)) / 2

