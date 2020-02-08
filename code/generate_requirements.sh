#!/bin/sh
conda env export > conda_environment.yml
conda list > conda_list.txt
conda list --explicit > conda_spec-file.txt
pip freeze > requirements_pip.txt