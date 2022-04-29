# Background

We have a machine learning pipeline which runs a batch scoring process.

As part of that process we want to check our data quality before it's fed
to the machine learning model.

We have an existing script that runs some checks on the data quality which
we want to improve and integrate into the pipeline.

# The Task

Refactor the script `quality_check.py` as you see appropriate for it to be
used in a machine learning pipeline.

You are free to change things as you see fit to meet the requirements.

Requirements:

- Validation:
	- Check for NaNs in the data.
	- Check for duplication issues in the data.
	- Check the account types in the data.
	- We expect many more quality checks to be added in the future (50+).
- How it will be used:
	- The validation should be callable from an existing Python application.
	- The pipeline should stop if there are data issues.


We expect no more than two hours to be spent on this task. If you
run over then please switch to writing what you would plan to do.

# Testing

Sample data has been provided for testing:

- `data-clean.csv` should pass validation
- `data-dirty.csv` should fail the validation

# How We Evaluate

- Correctness
- Clean code and Idiomatic Python
