# ATenH: Accelerated Tensor Handling for Physics-Informed DeepONets

This repository contains the implementation and experiments associated with the paper:

**"The importance of tensor handling and implementation of physics-informed DeepONets"**

We introduce **Accelerated Tensor Handling (ATenH)**, a reformulation of the physics-informed DeepONet (PI-DeepONet) architecture designed to reduce computational cost and memory requirements while maintaining predictive accuracy.

## Overview

Physics-informed DeepONets combine operator learning with physics-informed constraints to approximate solutions of parametric partial differential equations (PDEs). However, the standard implementation of PI-DeepONets (referred to here as the **Vanilla architecture**) can become computationally expensive for high-dimensional problems due to inefficient tensor handling and repeated evaluation of neural networks.

ATenH addresses these limitations by restructuring the input handling and tensor operations of the Branch and Trunk networks. This reduces unnecessary computations and memory usage without modifying the underlying operator-learning framework.

## Key differences
Individual function samples paired with collocation points in Vanilla, while ATenH re-uses Branch network passes for all collcoation points.
The inner dot product between subnetworks is managed via broadcasting in ATenH (this can be done efficiently in PyTorch using einsum).
ATenH has reduced computational scaling w.r.t collocation points compared to Vanilla.

## Results
The proposed ATenH implementation achieves significant reductions in computational cost and memory usage compared with the Vanilla architecture. In particular, ATenH enables training on higher-dimensional problems where the Vanilla implementation becomes limited by GPU memory.

## How to use the code
Each folder contains a ```Call_Training.py``` file where variables and hyperparameters can be specified for PI-DeepONet training. Run ```Call_Training.py``` to train the PI-DeepONet. This can be done for the Vanilla or ATenH architecture for one-, two- and three- dimensional linear advection problems. Trained models and relevant data will be saved to uniquely named folders within the ```Trained_Models``` folder. Model validation/testing can be performed by running ```Call_Testing.py``` with the folder name of the desired trained model data.


