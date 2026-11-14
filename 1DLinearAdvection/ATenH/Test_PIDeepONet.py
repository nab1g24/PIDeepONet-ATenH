import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn as nn
import numpy as np
import os
import gzip
from Functions.PIDeepONet import build_mlp, DeepONet
from plot_defaults import set_plot_defaults
set_plot_defaults()

def Call_PIDeepONet_Testing(file_name):
    #-----------------------------------------------------------------------------------------
    # Set the path for data of interest.
    #-----------------------------------------------------------------------------------------
    data_location = file_name

    #-------------------------------------------------------------------------------------------
    # Load the mesh points from the training script.
    #-------------------------------------------------------------------------------------------
    base_path = os.getcwd()
    # Change the folder path to the location of desired data.
    base_path = os.path.join(base_path, 'Trained_Models')
    folder_path = os.path.join(base_path, data_location)

    Cpoints_save_path = os.path.join(folder_path, "collocation_points.pkl")
    with open(Cpoints_save_path, 'rb') as f:
        collocation_points = pickle.load(f)

    input_tensor_0 = collocation_points["initial"]
    input_tensor_b = collocation_points["boundary"]
    input_tensor_I = collocation_points["internal"]

    #-------------------------------------------------------------------------------------------
    # Load the testing input functions data from the training script.
    #-------------------------------------------------------------------------------------------
    TestData_save_path = os.path.join(folder_path, "test_data.pkl.gz")  # Make sure to match the compressed filename
    with gzip.open(TestData_save_path, 'rb') as f:
        test_data = pickle.load(f)

    test_tensor = test_data["BC"]
    test_As = test_data["A"]
    omega = test_data["omega"]
    c = test_data["c"]
    domain_limits = test_data["domain_limits"]
    branch_layers = test_data["branch_layers"]
    trunk_layers = test_data["trunk_layers"]
    Branch_Activation = test_data["Branch_Activation"]
    Trunk_Activation = test_data["Trunk_Activation"]
    test_num = test_tensor.shape[0]

    # Expand the trunk input tensors.
    input_tensor_0 = input_tensor_0.unsqueeze(0).expand(test_num, -1, -1) # Shape: [n, p_0, d]
    input_tensor_b = input_tensor_b.unsqueeze(0).expand(test_num, -1, -1) # Shape: [n, p_b, d]
    input_tensor_I = input_tensor_I.unsqueeze(0).expand(test_num, -1, -1) # Shape: [n, p_I, d]

    #-----------------------------------------------------------------------------------------
    # Load the trained model.
    #-----------------------------------------------------------------------------------------
    # Initialise the neural network.
    model = DeepONet(branch_input_dim=test_tensor.shape[1], branch_layers=branch_layers, Branch_Activation=Branch_Activation, trunk_input_dim=input_tensor_I.shape[2], trunk_layers=trunk_layers, Trunk_Activation=Trunk_Activation)

    # Apply the trained model weights.
    trained_deeponet_save_path = os.path.join(folder_path, "trained_DeepONet.pth")
    state_dict_DeepONet = torch.load(trained_deeponet_save_path)
    model.load_state_dict(state_dict_DeepONet)
    model.eval()

    # Forward pass through the model.
    with torch.no_grad():
        G_0 = model(test_tensor, input_tensor_0)
        G_b = model(test_tensor, input_tensor_b)
        G_I = model(test_tensor, input_tensor_I)

    # Function for converting the collocation points which were input into the model into a mesh.
    def CollocationToMesh(InitialPoints, BoundaryPoints, InternalPoints, internal_x_len, internal_t_len, test_num):
        if InternalPoints.shape[2] != 1: # This option works for manipulating the 2D input points (collocation points).
            # First getting the internal collocation points.
            mesh_x = InternalPoints[:, :, 0].view(test_num, internal_t_len, internal_x_len) # [n, p/nx - 1, p/nt - 1]
            mesh_t = InternalPoints[:, :, 1].view(test_num, internal_t_len, internal_x_len) # [n, p/nx - 1, p/nt - 1]

            # Now we need to add the initial points to the mesh.
            mesh_0 = InitialPoints.permute(0,2,1) # [n, 1, p/nt]

            mesh_x = torch.cat((mesh_0[:, 0, :].unsqueeze(1), mesh_x), 1) # [n, p/nx, p/nt - 1]
            mesh_t = torch.cat((mesh_0[:, 1, :].unsqueeze(1), mesh_t), 1) # [n, p/nx, p/nt - 1]

            # Now we need to add the boundary points to the mesh.
            mesh_x = torch.cat((BoundaryPoints[:,:,0].unsqueeze(2), mesh_x), 2) # [n, p/nx, p/nt]
            mesh_t = torch.cat((BoundaryPoints[:,:,1].unsqueeze(2), mesh_t), 2) # [n, p/nx, p/nt]

            return mesh_x, mesh_t

        else: # This option works for manipulating the 1D output points (model solution).
            # First getting the internal points.
            reshaped_G = InternalPoints[:,:,0].view(test_num,internal_t_len,internal_x_len) # [n, p/nx - 1, p/nt - 1]

            # Now we need to add the initial points to the mesh.
            G_0 = InitialPoints.permute(0,2,1) # [n, 1, p/nt]
            reshaped_G = torch.cat((G_0, reshaped_G), 1) # [n, p/nx, p/nt - 1]

            # Now we need to add the boundary points to the mesh.
            reshaped_G = torch.cat((BoundaryPoints, reshaped_G), 2) # [n, p/nx, p/nt]
            return reshaped_G
        

    internal_x_len = input_tensor_0.shape[1]
    internal_t_len = input_tensor_b.shape[1] - 1 

    # Reshape the model output and collocation points into a mesh.
    reshaped_G = CollocationToMesh(G_0, G_b, G_I, internal_x_len, internal_t_len, test_num)
    mesh_x, mesh_t = CollocationToMesh(input_tensor_0, input_tensor_b, input_tensor_I, internal_x_len, internal_t_len, test_num)   

    #-----------------------------------------------------------------------------------------
    # Generating the exact solutions.
    #-----------------------------------------------------------------------------------------
    # Function to output the exact solution across the mesh.
    def exact_f(X, T, A, domain_limits, omega, c, lower_b):
        # Compute the characteristic condition
        characteristic_condition = T - (X - lower_b)/c >= 0  # Condition: t >= (x - lower_b/c)

        solution = A * np.exp(-((T.detach().numpy() - ((domain_limits[1,1] + domain_limits[1,0]) / 2) - ((X.detach().numpy() - lower_b.detach().numpy())/c.detach().numpy())) ** 2) / (2 * omega ** 2))

        # Set solution to zero where the condition does not hold
        solution = np.where(characteristic_condition.detach().numpy(), solution, 0)
        
        return solution

    # Preallocate the output tensor to store solutions for all 50 cases
    exact_solutions = np.zeros_like(mesh_x.detach().numpy())

    # Loop over each A value
    for i in range(test_num):
        # Compute the solution for the i-th A
        exact_solutions[i] = exact_f(mesh_x[i], mesh_t[i], test_As[i], domain_limits, omega, c, mesh_x[i,0,0])

    #-----------------------------------------------------------------------------------------
    # Save out the model solution and the exact solutions.
    #-----------------------------------------------------------------------------------------
    trained_model_save_path = os.path.join(folder_path, "trained_model_solution.pkl")
    with open(trained_model_save_path, 'wb') as f:
        pickle.dump({'model_output': reshaped_G, 'x_mesh': mesh_x, 't_mesh': mesh_t, 'exact_solutions': exact_solutions}, f)


