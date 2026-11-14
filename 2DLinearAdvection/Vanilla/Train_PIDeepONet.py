import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import zipfile
import io
from Functions.PIDeepONet import *

def Call_PIDeepONet_Training(learning_rate, num_epochs, branch1_layers, branch2_layers, trunk_layers,
                            train_num, test_num, SampleNum_0, SampleNum_xb, SampleNum_yb, SampleNum_I,
                            lambda1, lambda2, lambda3, lambda4, c, omega_x, omega_y, omega_t, max_A, min_A, testing_collocation_ns, input_function_ns, limits, seed,
                            Processor, Save_model=False, Save_Losses=False, Save_test_data=False, Save_mesh=False, Set_NN_Seed=False,
                            Branch1_Activation="tanh", Branch2_Activation="tanh", Trunk_Activation="tanh"):

    # Convert input data to correct types.
    learning_rate = torch.tensor(learning_rate, dtype=torch.float32)
    c = torch.tensor(c, dtype=torch.float32)
    limits = np.array(limits)
    input_function_ns = np.array(input_function_ns)
    testing_collocation_ns = np.array(testing_collocation_ns)

    # Starting timer.
    start_time = time.time()

    #-----------------------------------------------------------------------------------------
    # Creating the mesh.
    #-----------------------------------------------------------------------------------------
    input_function_sensor_points = mesh_building(input_function_ns, limits)
    collocation_points = mesh_building(testing_collocation_ns, limits, tensor=True, for_input_function=False)

    #-----------------------------------------------------------------------------------------
    # Generating the training and test lower boundary conditions.
    #-----------------------------------------------------------------------------------------
    np.random.seed(seed)
    train_As_xb = np.random.uniform(min_A, max_A, train_num)
    test_As_xb = np.random.uniform(min_A, max_A, test_num)
    train_As_yb = np.random.uniform(min_A, max_A, train_num)
    test_As_yb = np.random.uniform(min_A, max_A, test_num)
    # Generate the matrix of data using the lower boundary function.
    train_matrix_xb = np.array([lower_xboundary(input_function_sensor_points["ylinspace"], input_function_sensor_points["tlinspace"], A, limits, [omega_y, omega_t], 'mesh') for A in train_As_xb])
    test_matrix_xb = np.array([lower_xboundary(input_function_sensor_points["ylinspace"], input_function_sensor_points["tlinspace"], A, limits, [omega_y, omega_t], 'mesh') for A in test_As_xb])
    train_matrix_yb = np.array([lower_yboundary(input_function_sensor_points["xlinspace"], input_function_sensor_points["tlinspace"], A, limits, [omega_x, omega_t], 'mesh') for A in train_As_yb])
    test_matrix_yb = np.array([lower_yboundary(input_function_sensor_points["xlinspace"], input_function_sensor_points["tlinspace"], A, limits, [omega_x, omega_t], 'mesh') for A in test_As_yb])
    # Convert the data matrix to a torch tensor.
    train_tensor_xb = torch.tensor(train_matrix_xb, dtype=torch.float32)
    train_tensor_yb = torch.tensor(train_matrix_yb, dtype=torch.float32)
    test_tensor_xb = torch.tensor(test_matrix_xb, dtype=torch.float32)
    test_tensor_yb = torch.tensor(test_matrix_yb, dtype=torch.float32)

    # Flatten the mesh grid to be of shape [samples, space_dim * time_dim].
    train_tensor_xb = train_tensor_xb.reshape(train_tensor_xb.size(0), -1)
    train_tensor_yb = train_tensor_yb.reshape(train_tensor_yb.size(0), -1)
    test_tensor_xb  = test_tensor_xb.reshape(test_tensor_xb.size(0), -1)
    test_tensor_yb  = test_tensor_yb.reshape(test_tensor_yb.size(0), -1)

    #-----------------------------------------------------------------------------------------
    # Sampling the collocation points. Need to generate sample collocation initial, boundary
    # and internal collocation points for every boundary condition. Also get corresponding ICs
    # and BCs.
    #-----------------------------------------------------------------------------------------
    # Function to scale the collocation points to the limits of the domain.
    def Scale_CollocationPoints(limits, collocation_points):
        # Scale the collocation points to the limits of the domain.
        scaled_collocation_points = np.zeros_like(collocation_points)
        scaled_collocation_points[:, :, 0] = limits[0, 0] + (limits[0, 1] - limits[0, 0]) * collocation_points[:, :, 0]
        scaled_collocation_points[:, :, 1] = limits[1, 0] + (limits[1, 1] - limits[1, 0]) * collocation_points[:, :, 1]
        scaled_collocation_points[:, :, 2] = limits[2, 0] + (limits[2, 1] - limits[2, 0]) * collocation_points[:, :, 2]
        return scaled_collocation_points

    # Function to sample the collocation points.
    def Sample_CollocationPoints(limits, train_num, SampleNum_0, SampleNum_xb, SampleNum_yb, SampleNum_I):
        #---------------------------------------------------------------------------------------
        ## Generating the collocation points randomly.
        # initial points, need last dimension to be zero, first two vary between 0 and 1.
        initial_collocation_points = np.random.rand(train_num, SampleNum_0, 3)
        initial_collocation_points = Scale_CollocationPoints(limits, initial_collocation_points)
        initial_collocation_points[:, :, 2] = limits[2,0]

        # boundary points, need first dimension to be zero, last two vary between 0 and 1.
        xboundary_collocation_points = np.random.rand(train_num, SampleNum_xb, 3)
        xboundary_collocation_points = Scale_CollocationPoints(limits, xboundary_collocation_points)
        xboundary_collocation_points[:, :, 0] = limits[0,0]

        # boundary points, need first dimension to be zero, last two vary between 0 and 1.
        yboundary_collocation_points = np.random.rand(train_num, SampleNum_yb, 3)
        yboundary_collocation_points = Scale_CollocationPoints(limits, yboundary_collocation_points)
        yboundary_collocation_points[:, :, 1] = limits[1,0]

        # internal points, all three dimensions vary between 0 and 1.
        internal_collocation_points = np.random.rand(train_num, SampleNum_I, 3)
        internal_collocation_points = Scale_CollocationPoints(limits, internal_collocation_points)

        sampled_collocation_points = {
                "internal": internal_collocation_points,
                "xboundary": xboundary_collocation_points,
                "yboundary": yboundary_collocation_points,
                "initial": initial_collocation_points
            }

        return sampled_collocation_points

    sampled_collocation_points = Sample_CollocationPoints(limits, train_num, SampleNum_0, SampleNum_xb, SampleNum_yb, SampleNum_I)

    # Generating the exact initial and boundary conditions for the sampled collocation points.
    # x boundary.
    sampled_xboundary_conditions = np.full_like(sampled_collocation_points["xboundary"][:,:,0], 0.0)
    for i in range(train_num):
        # Get the exact initial conditions for the sampled collocation points.
        sampled_xboundary_conditions[i] = lower_xboundary(sampled_collocation_points["xboundary"][i,:,1], sampled_collocation_points["xboundary"][i,:,2], train_As_xb[i], limits, [omega_y, omega_t], 'collocation')
    # y boundary.
    sampled_yboundary_conditions = np.full_like(sampled_collocation_points["yboundary"][:,:,0], 0.0)
    for i in range(train_num):
        # Get the exact initial conditions for the sampled collocation points.
        sampled_yboundary_conditions[i] = lower_yboundary(sampled_collocation_points["yboundary"][i,:,0], sampled_collocation_points["yboundary"][i,:,2], train_As_yb[i], limits, [omega_x, omega_t], 'collocation')

    sampled_collocation_points = {
        "internal": torch.tensor(sampled_collocation_points["internal"], dtype=torch.float32),
        "xboundary": torch.tensor(sampled_collocation_points["xboundary"], dtype=torch.float32),
        "yboundary": torch.tensor(sampled_collocation_points["yboundary"], dtype=torch.float32),
        "initial": torch.tensor(sampled_collocation_points["initial"], dtype=torch.float32)
    }

    sampled_exact_solutions = {
        "xboundary": torch.tensor(sampled_xboundary_conditions, dtype=torch.float32),
        "yboundary": torch.tensor(sampled_yboundary_conditions, dtype=torch.float32),
        "initial": torch.full_like(sampled_collocation_points["initial"][:,:,0], 0.0)
    }

    #-----------------------------------------------------------------------------------------
    # Duplicate the input functions to match the number of collocation points.
    #-----------------------------------------------------------------------------------------
    # In this case we do a single branch pass for each trunk pass.
    # Add a new dimension in the second position and expand it
    train_tensors_xb = {"initial":train_tensor_xb.unsqueeze(1).expand(-1, SampleNum_0, -1),
                    "xboundary":train_tensor_xb.unsqueeze(1).expand(-1, SampleNum_xb, -1),
                    "yboundary":train_tensor_xb.unsqueeze(1).expand(-1, SampleNum_yb, -1),
                    "internal":train_tensor_xb.unsqueeze(1).expand(-1, SampleNum_I, -1)}
    train_tensors_yb = {"initial":train_tensor_yb.unsqueeze(1).expand(-1, SampleNum_0, -1),
                    "xboundary":train_tensor_yb.unsqueeze(1).expand(-1, SampleNum_xb, -1),
                    "yboundary":train_tensor_yb.unsqueeze(1).expand(-1, SampleNum_yb, -1),
                    "internal":train_tensor_yb.unsqueeze(1).expand(-1, SampleNum_I, -1)}

    #-----------------------------------------------------------------------------------------
    # Reshape all model inputs to the form: [n*p, m or d].
    #-----------------------------------------------------------------------------------------
    # Reshaping input data to [batch_num, input_dim] format.
    def reshape_tensors_3Dto2D(tensors):
        reshaped_tensors = {}
        # Reshape the tensor to [batch_num, input_dim]
        reshaped_tensors["initial"] = tensors["initial"].reshape(tensors["initial"].shape[0]*tensors["initial"].shape[1], tensors["initial"].shape[2])
        reshaped_tensors["xboundary"] = tensors["xboundary"].reshape(tensors["xboundary"].shape[0]*tensors["xboundary"].shape[1], tensors["xboundary"].shape[2])
        reshaped_tensors["yboundary"] = tensors["yboundary"].reshape(tensors["yboundary"].shape[0]*tensors["yboundary"].shape[1], tensors["yboundary"].shape[2])
        reshaped_tensors["internal"] = tensors["internal"].reshape(tensors["internal"].shape[0]*tensors["internal"].shape[1], tensors["internal"].shape[2])
        return reshaped_tensors

    def reshape_tensor_2Dto1D(tensor):
        # Reshape the tensor to [batch_num, input_dim]
        reshaped_tensor = tensor.reshape(tensor.shape[0]*tensor.shape[1])
        return reshaped_tensor

    # Reshaping the branch inputs to [n*p, m] format.
    train_tensors_xb = reshape_tensors_3Dto2D(train_tensors_xb)
    train_tensors_yb = reshape_tensors_3Dto2D(train_tensors_yb)

    sampled_collocation_points = reshape_tensors_3Dto2D(sampled_collocation_points)
    sampled_collocation_points["internal"] = sampled_collocation_points["internal"].requires_grad_(True)

    sampled_exact_solutions["initial"] = reshape_tensor_2Dto1D(sampled_exact_solutions["initial"])
    sampled_exact_solutions["xboundary"] = reshape_tensor_2Dto1D(sampled_exact_solutions["xboundary"])
    sampled_exact_solutions["yboundary"] = reshape_tensor_2Dto1D(sampled_exact_solutions["yboundary"])

    #-----------------------------------------------------------------------------------------
    # Single training run.
    #-----------------------------------------------------------------------------------------
    if Set_NN_Seed:
        set_seed(seed, Processor)

    model = DeepONet(branch1_input_dim=train_tensors_xb["initial"].shape[-1], branch1_layers=branch1_layers, branch1_activation=Branch1_Activation,
                     branch2_input_dim=train_tensors_yb["initial"].shape[-1], branch2_layers=branch2_layers, branch2_activation=Branch2_Activation,
                     trunk_input_dim=sampled_collocation_points["internal"].shape[-1], trunk_layers=trunk_layers, trunk_activation=Trunk_Activation)

    #-----------------------------------------------------------------------------------------
    # Move tensors to GPU.
    #-----------------------------------------------------------------------------------------
    if Processor == 'NvidiaGPU':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sampled_exact_solutions["initial"] = sampled_exact_solutions["initial"].to(device)
        sampled_exact_solutions["xboundary"] = sampled_exact_solutions["xboundary"].to(device)
        sampled_exact_solutions["yboundary"] = sampled_exact_solutions["yboundary"].to(device)
        sampled_collocation_points['initial'] = sampled_collocation_points['initial'].to(device)
        sampled_collocation_points['xboundary'] = sampled_collocation_points['xboundary'].to(device)
        sampled_collocation_points['yboundary'] = sampled_collocation_points['yboundary'].to(device)
        sampled_collocation_points['internal'] = sampled_collocation_points['internal'].to(device)
        train_tensors_xb["initial"] = train_tensors_xb["initial"].to(device)
        train_tensors_xb["xboundary"] = train_tensors_xb["xboundary"].to(device)
        train_tensors_xb["yboundary"] = train_tensors_xb["yboundary"].to(device)
        train_tensors_xb["internal"] = train_tensors_xb["internal"].to(device)
        train_tensors_yb["initial"] = train_tensors_yb["initial"].to(device)
        train_tensors_yb["xboundary"] = train_tensors_yb["xboundary"].to(device)
        train_tensors_yb["yboundary"] = train_tensors_yb["yboundary"].to(device)
        train_tensors_yb["internal"] = train_tensors_yb["internal"].to(device)
        model = model.to(device)
        c = c.to(device)
        learning_rate = learning_rate.to(device)

    #-----------------------------------------------------------------------------------------
    # Train the model.
    #-----------------------------------------------------------------------------------------
    # Define the optimizer
    LR = learning_rate.item()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()  # Create an instance of MSELoss

    # Lists to store epoch numbers and loss values
    epochs = []
    losses = []
    losses_0 = []
    losses_xb = []
    losses_yb = []
    losses_I = []

    # Timer for training.
    start_train_time = time.time()

    print("start training")
    losses, losses_0, losses_xb, losses_yb, losses_I, epochs, model = training_loop(model, optimizer, mse_loss, train_tensors_xb, train_tensors_yb,
                                                                                    sampled_collocation_points, sampled_exact_solutions, c, num_epochs,
                                                                                    lambda1, lambda2, lambda3, lambda4, losses, losses_0, losses_xb,
                                                                                    losses_yb, losses_I, epochs)


    # Ending timer for training.
    train_time = time.time() - start_train_time
    print(f"Traintime: {train_time:.6f} seconds")

    # Ending timer.
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.6f} seconds")

    if Processor == 'NvidiaGPU':
        model = model.cpu()
        c = c.cpu()
        learning_rate = learning_rate.cpu()

    #-----------------------------------------------------------------------------------------
    # Save out the data.
    #-----------------------------------------------------------------------------------------
    if Save_Losses or Save_model or Save_test_data or Save_mesh:
        base_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(base_path, "Trained_Models")

        save_data(base_path, branch1_layers, branch2_layers, trunk_layers, Branch1_Activation, Branch2_Activation, Trunk_Activation,
            num_epochs, lambda1, lambda2, lambda3, lambda4,
            learning_rate, c, omega_x, omega_y, omega_t, input_function_ns, testing_collocation_ns,
            SampleNum_0, SampleNum_xb, SampleNum_yb, SampleNum_I,
            train_num, test_num, Processor,
            model, epochs, losses, losses_0, losses_xb, losses_yb, losses_I,
            test_tensor_xb, test_tensor_yb, test_As_xb, test_As_yb,
            limits, collocation_points,
            runtime, train_time,
            Save_model=Save_model, Save_Losses=Save_Losses, Save_test_data=Save_test_data, Save_mesh=Save_mesh)
