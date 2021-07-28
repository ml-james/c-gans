#!/bin/bash

pip install flake8
flake8 .
python3 -m unittest tests/*test*