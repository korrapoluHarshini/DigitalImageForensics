"""Notebook helpers for the automated Colab workflow.

The Colab notebook imports this module so that the fragile parts (dataset
layout detection, input validation, resume planning, artefact packaging) are
ordinary, testable Python instead of untested notebook code.
"""
