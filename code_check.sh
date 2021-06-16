#!/bin/bash
echo "Auto-formatting code..."
brunette .  # auto-format code
echo "Checking format..."
flake8      # catch any errors that the first tool missed
echo "Checking types..."
mypy .      # check types
echo "Done!"
