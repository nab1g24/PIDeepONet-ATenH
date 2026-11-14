import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from Functions.PIDeepONet import *


def Call_PIDeepONet_Training(learning_rate, num_epochs, branch_layers, trunk_layers,
                            train_num, test_num, SampleNum_0, SampleNum_b, SampleNum_I,
                            lambda1, lambda2, lambda3, c, omega, max_A, min_A, testing_collocation_ns, input_function_ns, limits, seed,
                            Processor, Save_model=False, Save_Losses=False, Save_test_data=False, Save_mesh=False, Set_NN_Seed=False,
                            Branch_Activation="tanh", Trunk_Activation="tanh"):

    # Convert input data to correct types.
    learning_rate = torch.tensor(learning_rate, dtype=torch.float32)
    c = torch.tensor([c], dtype=torch.float32)
    testing_collocation_ns = np.array(testing_collocation_ns)
    input_function_ns = np.array(input_function_ns)
    limits = np.array(limits)

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
    train_As = np.random.uniform(min_A, max_A, train_num)
    test_As = np.random.uniform(min_A, max_A, test_num)
    # Generate the matrix of data using the lower boundary function.
    train_matrix = np.array([lower_boundary(input_function_sensor_points["tlinspace"], A, limits, omega) for A in train_As])
    test_matrix = np.array([lower_boundary(input_function_sensor_points["tlinspace"], A, limits, omega) for A in test_As])
    # Convert the data matrix to a torch tensor.
    train_tensor = torch.tensor(train_matrix, dtype=torch.float32)
    test_tensor = torch.tensor(test_matrix, dtype=torch.float32)

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
        return scaled_collocation_points

    # Function to sample the collocation points.
    def Sample_CollocationPoints(limits, train_num, SampleNum_0, SampleNum_b, SampleNum_I):
        #---------------------------------------------------------------------------------------
        ## Generating the collocation points randomly.
        # initial points, need last dimension to be zero, first two vary between 0 and 1.
        initial_collocation_points = np.random.rand(train_num, SampleNum_0, 2)
        initial_collocation_points = Scale_CollocationPoints(limits, initial_collocation_points)
        initial_collocation_points[:, :, 1] = limits[1,0]

        # boundary points, need first dimension to be zero, last two vary between 0 and 1.
        boundary_collocation_points = np.random.rand(train_num, SampleNum_b, 2)
        boundary_collocation_points = Scale_CollocationPoints(limits, boundary_collocation_points)
        boundary_collocation_points[:, :, 0] = limits[0,0]

        # internal points, all three dimensions vary between 0 and 1.
        internal_collocation_points = np.random.rand(train_num, SampleNum_I, 2)
        internal_collocation_points = Scale_CollocationPoints(limits, internal_collocation_points)

        sampled_collocation_points = {
                "internal": internal_collocation_points,
                "boundary": boundary_collocation_points,
                "initial": initial_collocation_points
            }

        return sampled_collocation_points

    sampled_collocation_points = Sample_CollocationPoints(limits, train_num, SampleNum_0, SampleNum_b, SampleNum_I)

    # Generating the exact initial and boundary conditions for the sampled collocation points.
    sampled_boundary_conditions = np.full_like(sampled_collocation_points["boundary"][:,:,0], 0.0)
    for i in range(train_num):
        # Get the exact initial conditions for the sampled collocation points.
        sampled_boundary_conditions[i] = lower_boundary(sampled_collocation_points["boundary"][i,:,1], train_As[i], limits, omega)

    sampled_collocation_points = {
        "internal": torch.tensor(sampled_collocation_points["internal"], dtype=torch.float32),
        "boundary": torch.tensor(sampled_collocation_points["boundary"], dtype=torch.float32),
        "initial": torch.tensor(sampled_collocation_points["initial"], dtype=torch.float32)
    }

    sampled_exact_solutions = {
        "boundary": torch.tensor(sampled_boundary_conditions, dtype=torch.float32),
        "initial": torch.full_like(sampled_collocation_points["initial"][:,:,0], 0.0)
    }

    #-----------------------------------------------------------------------------------------
    # Duplicate the input functions to match the number of collocation points.
    #-----------------------------------------------------------------------------------------
    # In this case we do a single branch pass for each trunk pass.
    # Add a new dimension in the second position and expand it
    train_tensors = {"initial":train_tensor.unsqueeze(1).expand(-1, SampleNum_0, -1),
                    "boundary":train_tensor.unsqueeze(1).expand(-1, SampleNum_b, -1),
                    "internal":train_tensor.unsqueeze(1).expand(-1, SampleNum_I, -1)}

    N_initial = collocation_points["initial"].shape[0]
    N_boundary = collocation_points["boundary"].shape[0]
    N_internal = collocation_points["internal"].shape[0]
    # Add a new dimension in the second position and expand it
    test_tensors = {"initial":test_tensor.unsqueeze(1).expand(-1, N_initial, -1),
                    "boundary":test_tensor.unsqueeze(1).expand(-1, N_boundary, -1),
                    "internal":test_tensor.unsqueeze(1).expand(-1, N_internal, -1)}

    #-----------------------------------------------------------------------------------------
    # Reshape all model inputs to the form: [n*p, m or d].
    #-----------------------------------------------------------------------------------------
    # Reshaping input data to [batch_num, input_dim] format.
    def reshape_tensor_3Dto2D(tensor):
        # Reshape the tensor to [batch_num, input_dim]
        reshaped_tensor = tensor.reshape(tensor.shape[0]*tensor.shape[1], tensor.shape[2])
        return reshaped_tensor

    def reshape_tensor_2Dto1D(tensor):
        # Reshape the tensor to [batch_num, input_dim]
        reshaped_tensor = tensor.reshape(tensor.shape[0]*tensor.shape[1])
        return reshaped_tensor

    train_tensors["initial"] = reshape_tensor_3Dto2D(train_tensors["initial"])
    train_tensors["boundary"] = reshape_tensor_3Dto2D(train_tensors["boundary"])
    train_tensors["internal"] = reshape_tensor_3Dto2D(train_tensors["internal"])
    test_tensors["initial"] = reshape_tensor_3Dto2D(test_tensors["initial"])
    test_tensors["boundary"] = reshape_tensor_3Dto2D(test_tensors["boundary"])
    test_tensors["internal"] = reshape_tensor_3Dto2D(test_tensors["internal"])

    sampled_collocation_points["initial"] = reshape_tensor_3Dto2D(sampled_collocation_points["initial"])
    sampled_collocation_points["boundary"] = reshape_tensor_3Dto2D(sampled_collocation_points["boundary"])
    sampled_collocation_points["internal"] = reshape_tensor_3Dto2D(sampled_collocation_points["internal"]).requires_grad_(True)

    sampled_exact_solutions["initial"] = reshape_tensor_2Dto1D(sampled_exact_solutions["initial"])
    sampled_exact_solutions["boundary"] = reshape_tensor_2Dto1D(sampled_exact_solutions["boundary"])

    #-----------------------------------------------------------------------------------------
    # Single training run
    #-----------------------------------------------------------------------------------------
    if Set_NN_Seed:
        set_seed(seed, Processor)

    model = DeepONet(branch_input_dim=train_tensors["initial"].shape[-1], branch_layers=branch_layers, Branch_Activation=Branch_Activation,
                     trunk_input_dim=sampled_collocation_points["internal"].shape[-1], trunk_layers=trunk_layers, Trunk_Activation=Trunk_Activation)

    #-----------------------------------------------------------------------------------------
    # Move tensors to GPU.
    #-----------------------------------------------------------------------------------------
    if Processor == 'NvidiaGPU':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sampled_exact_solutions["initial"] = sampled_exact_solutions["initial"].to(device)
        sampled_exact_solutions["boundary"] = sampled_exact_solutions["boundary"].to(device)
        sampled_collocation_points['initial'] = sampled_collocation_points['initial'].to(device)
        sampled_collocation_points['boundary'] = sampled_collocation_points['boundary'].to(device)
        sampled_collocation_points['internal'] = sampled_collocation_points['internal'].to(device)
        train_tensors["initial"] = train_tensors["initial"].to(device)
        train_tensors["boundary"] = train_tensors["boundary"].to(device)
        train_tensors["internal"] = train_tensors["internal"].to(device)
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
    losses_b = []
    losses_I = []

    start_train_time = time.time()

    print("start training")
    losses, losses_0, losses_b, losses_I, epochs, model = training_loop(model, optimizer, mse_loss, train_tensors,
                                                                        sampled_collocation_points, sampled_exact_solutions,
                                                                        c, num_epochs, lambda1, lambda2, lambda3, losses, losses_0,
                                                                        losses_b, losses_I, epochs)

    # Ending timer for training.
    train_time = time.time() - start_train_time
    print(f"Traintime: {train_time:.6f} seconds")

    # Ending timer.
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.6f} seconds")

    if Processor == "NvidiaGPU":
        model = model.cpu()
        c = c.cpu()

    #-----------------------------------------------------------------------------------------
    # Save out the data.
    #-----------------------------------------------------------------------------------------
    if Save_Losses or Save_model or Save_test_data or Save_mesh:
        base_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(base_path, "Trained_Models")

        save_data(base_path, branch_layers, trunk_layers, Branch_Activation, Trunk_Activation, num_epochs,
                    lambda1, lambda2, lambda3,
                    learning_rate, c, omega, input_function_ns, testing_collocation_ns,
                    SampleNum_0, SampleNum_b, SampleNum_I,
                    train_num, test_num, Processor,
                    model, epochs, losses, losses_0, losses_b, losses_I,
                    test_tensors, test_As,
                    limits, collocation_points,
                    runtime, train_time,
                    Save_model=Save_model, Save_Losses=Save_Losses, Save_test_data=Save_test_data, Save_mesh=Save_mesh)
