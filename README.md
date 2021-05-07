### cGANs

#### Introduction 

This repository is a direct result of my Masters Physics research project at *The University of Cambridge* entitled: **Fast Simulation of Particle Physics Detectors Using Machine Learning Tools**. Herein lies the final report, and the code used to generate the containing plots. I shall defer to the `pdf` to explain the motivation, the aims of the study and the results. 

#### Running the scripts

In order to set-up the environment to run the scripts in this project, please execute the following commands. You will need to run these commands in the root directory of this repository as they will require the `Pipfile` and `Pipfile.lock` files:
 
 1) `python -m pip install --upgrade pip`
 2) `pip install pipenv`
 3) `pipenv install`

To run `c-dcgan.py` please type the following into your terminal: `pipenv run python c-dcgan.py <path-to-results-directory> <number-of-epochs>`

#### What's up next?

* Tidy up the remaining scripts.
* Commit the .h5 files of the trained models for re-usability.
* Train the models in AWS...?
