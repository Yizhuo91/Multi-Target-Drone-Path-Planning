#!/bin/bash

case ${1:-'cuda'} in

    'cuda')
    python3.8 -m venv venvs/cuda
    source venvs/cuda/Scripts/activate
    pip install -r requirements/cuda.txt
    ;;

    'directml')
    python3.8 -m venv venvs/directml
    source venvs/directml/Scripts/activate
    pip install -r requirements/directml.txt
    ;;

    *)
    echo 'usage: bash setup.sh {cuda,directml}'
    ;;

esac
