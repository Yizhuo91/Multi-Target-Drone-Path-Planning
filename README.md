# Drone Path Planning

This is the GitHub repository for the drone path planning project. It aims to apply deep reinforcement learning to the problem of path planning for drones.

## Setup

In your working directory, clone this repository.

```bash
git clone https://github.com/jayasting98/drone-path-planning.git
```

This project is intended to run on Python 3.8 (the version of Python in the NUS School of Computing Compute Cluster as of 2 January 2023); ensure you have Python 3.8. If you are using Ubuntu, you can use [this Stack Exchange answer](https://askubuntu.com/questions/682869/how-do-i-install-a-different-python-version-using-apt-get#answer-682875) as a guide to install Python 3.8 especially if you already have other Python versions installed. After that, run `setup.sh`. If you want to run the code with CUDA, use the `cuda` argument. Otherwise, if you want to run the code with DirectML, use the `directml` argument. If no argument is provided, the setup will default to `cuda`.

```bash
bash setup.sh {cuda,directml}
```

For example, for `directml`, run the following command.

```bash
bash setup.sh directml
```

This will create a Python virtual environment with `venv` and install the required Python packages using pip with reference to the respective pip requirements file in the `requirements` folder according to what option was chosen. The Python virtual environment will be named according to the argument used when running `setup.sh`. Alternatively, you can set up the Python virtual environment yourself (or not, but it is recommended that you do) and you can install the required packages yourself, which can be found in the pip requirements files in the `requirements` folder.

## Usage

Activate the Python virtual environment using the following command.

```bash
source venvs/VENV_NAME/bin/activate
```

For example, the `cuda` virtual environment can be activated using the following command.

```bash
source venvs/cuda/bin/activate
```

Then you can run the code using the following command.

```bash
python -m drone_path_planning.main ROUTINE SCENARIO [options]
```

For example, if you want to train an agent in the single-chaser single-moving-target scenario, use the `train` routine and the `single-chaser_single-moving-target` scenario. If you want to save the model to the `data/single-chaser_single-moving-target/0/saves/checkpoint` folder, use the `--save_dir` parameter. If you want training logs recorded into the `data/single-chaser_single-moving-target/0/logs` directory, use the `--logs_dir` argument.

```bash
python -m drone_path_planning.main train single-chaser_single-moving-target --save_dir=data/single-chaser_single-moving-target/0/saves/checkpoint --logs_dir=data/single-chaser_single-moving-target/0/logs
```

OR

```bash
export SCENARIO="single-chaser_single-moving-target"
export RUN="0"
export SAVE_DIR="data/${SCENARIO}/${RUN}/saves/checkpoint"
export LOGS_DIR="data/${SCENARIO}/${RUN}/logs"
python -m drone_path_planning.main train ${SCENARIO} --save_dir=${SAVE_DIR} --logs_dir=${LOGS_DIR}
```

You can use the following command for help and more information.

```bash
python -m drone_path_planning.main -h
```

## Examples

The following is a video showing multiple trajectories of the trained neural network controlling the agent chaser drones to intercept the target drone.

https://user-images.githubusercontent.com/65202977/229480327-a89084d9-bad2-43b0-990c-d9e3591e0432.mp4
