# Fake News Detection — Learning Project

A small, explainable starter project for experimenting with text classification and misinformation signals. This is an educational prototype, not a fact-checking authority.

## What it will do

- Train a baseline text classifier on a labeled dataset.
- Evaluate it on held-out data using precision, recall, F1-score, and a confusion matrix.
- Let a user enter a headline or article text and see the model's predicted class and probability scores.
- Keep limitations visible: a model can learn writing style and dataset bias rather than truth.

## Important limitation

**A model prediction is not proof that a story is true or false.** Use trustworthy sources and independent evidence to verify claims. Do not use this prototype for consequential decisions about people.

## Planned stack

- Python
- scikit-learn
- pandas
- Streamlit (for a lightweight UI, in a later step)

## Project status

This repository started empty. We are building it in small, testable steps. The first milestone is a reproducible baseline model with a clear dataset format and evaluation report.

## Quick start

The first implementation is being added next. Once the code is in place, follow the install and run commands in this README.

## Dataset expectations

Use a dataset you have permission to use, with a text column and a label column. Keep the original source and license noted. Avoid evaluating on examples used during training, and watch for duplicate or near-duplicate articles that can inflate scores.

## Roadmap

1. Add a simple training pipeline and data validation.
2. Add evaluation metrics and tests.
3. Add a local prediction interface.
4. Improve reliability with source/context signals and human review rather than treating text style as truth.
