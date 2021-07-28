### cGANs

#### Introduction 

This repository is a direct result of my Masters Physics research project at the *University of Cambridge* entitled: **Fast Simulation of Particle Physics Detectors using Machine Learning Tools**. Herein lies the final report, and the code used to generate the containing plots. I shall defer to the `.pdf` to explain the motivation, the aims of the study and the results. 

#### Running the scripts

In order to set-up the environment to run the scripts in this project, please download `docker`. Then from the `root` of this repository run the following commands.
 
 1) `docker build -t c-gans .`
 2) `docker run --rm -it --entrypoint bash c-gans`

Now running interactively within the docker container you can run the scripts, for example, to run `c-dcgan.py` please type the following into your terminal: `python3 c_gans/c-dcgan.py <path-to-results-directory> <number-of-epochs-to-train>`

#### What's up next?

* Tidy up the remaining scripts.
* Commit the `.h5` files of the trained models for re-usability.
* Train the models in AWS...?
