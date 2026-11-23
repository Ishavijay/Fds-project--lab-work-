"""Simple helpers for Unit I: Introduction to Data Science."""


# Explains the four main kinds of data analytics.
def analytics_types():
    return {
        "descriptive": "What happened?",
        "diagnostic": "Why did it happen?",
        "predictive": "What might happen next?",
        "prescriptive": "What should we do?",
    }


# Gives the main steps to follow in a CRISP-DM data science project.
def crisp_dm_steps():
    return [
        "Understand the problem",
        "Understand the data",
        "Prepare the data",
        "Build a model",
        "Check the results",
        "Use and monitor the result",
    ]


# Shows the five simple steps of the SEMMA data process.
def semma_steps():
    return ["Sample", "Explore", "Modify", "Model", "Assess"]

