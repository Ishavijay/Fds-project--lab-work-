# Unit I: introductory data science ideas.
# This module explains analytics types and the CRISP-DM and SEMMA workflows.


def analytics_types():
    # Return four kinds of analytics and the question each one answers.
    return {
        "descriptive": "What happened?",
        "diagnostic": "Why did it happen?",
        "predictive": "What might happen next?",
        "prescriptive": "What should we do?",
    }


def crisp_dm_steps():
    # Return the six steps in the CRISP-DM data science process.
    return [
        "Understand the problem",
        "Understand the data",
        "Prepare the data",
        "Build a model",
        "Check the results",
        "Use and monitor the result",
    ]


def semma_steps():
    # Return the five steps in the SEMMA data process.
    return ["Sample", "Explore", "Modify", "Model", "Assess"]
