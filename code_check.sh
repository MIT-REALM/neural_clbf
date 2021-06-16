#!/bin/bash
brunette .  # auto-format code
flake8      # catch any errors that the first tool missed
mypy .      # check types
