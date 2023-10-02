# Collision Predictor NMPC

This repository contains the framework using in our paper called: "Nonlinear Model Predictive Control for Deep Neural Network-Based
Collision Avoidance exploiting Depth Images", submitted to ICRA 2023.
The associated (temporary) video can be seen at https://youtu.be/aELjlfwAjfk. 


## Installation

First, install:

* [ml-casadi](https://github.com/TUM-AAS/ml-casadi)
* [acados_template](https://github.com/acados/acados)

Other Python requirements are listed in `requirements.txt` and can be installed with `pip`:
```
pip install -r requirements.txt
```

Clone the repo and navigate to main folder. Then, the package can be install using `pip`:
```
pip install -e .
```

NOTE: For convenience, the training data base folder is defined in `colpred_nmpc/__init__.py`, as the variable `COLPREDMPC_DATA_DIR`.
For retraining on your own dataset, please update this global variable or set the correct path in the training/testing scripts.


## Repo architecture

The repo folder is organized as follow:

1. **colpred_nmpc**: Main module folder, including Neural Nets, Casadi+Acados based controller, and ROS wrapper.
1. **config**: Yaml files for missions configurations and robot parameters.
1. **gazebo**: A handful of Gazebo world files used in simulations. NOTE: spawns robot module which uses [mrsim-gazebo](https://git.openrobots.org/projects/mrsim-gazebo) for simulations using GenoM. Should be replaced/removed for using with another simulation pipeline.
1. **logs**: Contains execution logs and Neural Nets weight files. This is where the scripts are looking for weights by default.
1. **scripts**: Scripts folder, including NN training/testing and main ROS controller (`nmpc_ros.py`).


## Usage

TODO upload weights + data somewhere?

### Training

Use `depth_state_check_train.py` for training the Neural Network.
The class `Colpred` encompasses the network definition as well as convenience functions.

### Testing

The script `depth_state_check_train.py` allows quantitative and qualitative evaluation of the trained network.
In particular, it allows to display classfication results (2D slices, or full 3D frustrum volume).

### Flying

The Neural NMPC controller is defined in `controller.py`, and the corresponging ROS interface is defined in `ros_wrapper.py`.
The main script `nmpc_ros.py` encapsulates the controller initialization, and the actual control loop.

Following ROS services are defined:
* startstop: starts the control loop
* goto: sets the desired waypoint relative to current pose, according the `p_des` in the yaml files.
* toggle_colpred: enable collision avoidance objective and constraint?

NOTE: When using ml-casadi, PyTorch builds the graph on first execution. Thus, the first call(s) to the CasADi function will be slow.
The main script thus processes a fake image once to prevent the first control iteration to be slow.


## Cite

When using this work in your research, please cite the following publication:

```
@INPROCEEDINGS{jacquet2023cpnmpc,
      author={Martin Jacquet and Kostas Alexis},
      title={TODO},
      year={2023},
      eprint={TODO},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```


## Ackowledgements

We would like to acknowledge [Mihir Kulkarni](mailto:mihir.kulkarni@ntnu.no) for the code snippets, in particular for `inflate_images.py` and `collision_checker.py`.


## Contact

You can contact us for any question:
* [Martin Jacquet](mailto:martin.jacquet@ntnu.no)
* [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no)
