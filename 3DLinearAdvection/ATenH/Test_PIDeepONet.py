import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn as nn
import numpy as np
import os
import zipfile
import io
from Functions.PIDeepONet import build_mlp, DeepONet
from plot_defaults import set_plot_defaults
set_plot_defaults()

def Call_PIDeepONet_Testing(file_name):
    #-----------------------------------------------------------------------------------------
    # Set the path for data of interest.
    #-----------------------------------------------------------------------------------------
    # Change this for different data.
    data_location = file_name

    # Load the mesh points from the training script.
    base_path = os.getcwd()
    # Change the folder path to the location of desired data.
    base_path = os.path.join(base_path, 'Trained_Models')
    folder_path = os.path.join(base_path, data_location)

    # Load pickle file directly from the ZIP without extracting
    Cpoints_zip_path = os.path.join(folder_path, "collocation_points.zip")
    with zipfile.ZipFile(Cpoints_zip_path, "r") as zipf:
        with zipf.open("collocation_points.pkl") as f:
            collocation_points = pickle.loads(f.read())
    # print(collocation_points.keys())

    input_tensor_0 = collocation_points["initial"]
    input_tensor_xb = collocation_points["xboundary"]
    input_tensor_yb = collocation_points["yboundary"]
    input_tensor_zb = collocation_points["zboundary"]
    input_tensor_I = collocation_points["internal"]

    TestData_save_path = os.path.join(folder_path, "test_data.pkl")
    with open(TestData_save_path, 'rb') as f:
        test_data = pickle.load(f)

    test_tensor_xb = test_data["xBC"]
    test_tensor_yb = test_data["yBC"]
    test_tensor_zb = test_data["zBC"]
    test_As_xb = test_data["A_xb"]
    test_As_yb = test_data["A_yb"]
    test_As_zb = test_data["A_zb"]
    c = test_data["c"]
    branch1_layers = test_data["branch1_layers"]
    branch2_layers = test_data["branch2_layers"]
    branch3_layers = test_data["branch3_layers"]
    trunk_layers = test_data["trunk_layers"]
    branch1_activation = test_data["Branch1_Activation"]
    branch2_activation = test_data["Branch2_Activation"]
    branch3_activation = test_data["Branch3_Activation"]
    trunk_activation = test_data["Trunk_Activation"]
    domain_limits = test_data["domain_limits"]
    omega_x = test_data["omega_x"]
    omega_y = test_data["omega_y"]
    omega_z = test_data["omega_z"]
    omega_t = test_data["omega_t"]
    test_num = test_tensor_xb.shape[0]

    # Expand the input tensors.
    input_tensor_0 = input_tensor_0.unsqueeze(0).expand(test_num, -1, -1)  # Shape: [train_num, SampleNum_0, input_dim]
    input_tensor_xb = input_tensor_xb.unsqueeze(0).expand(test_num, -1, -1)
    input_tensor_yb = input_tensor_yb.unsqueeze(0).expand(test_num, -1, -1)
    input_tensor_zb = input_tensor_zb.unsqueeze(0).expand(test_num, -1, -1)
    input_tensor_I = input_tensor_I.unsqueeze(0).expand(test_num, -1, -1)

    # Load the trained model.
    model = DeepONet(branch1_input_dim=test_tensor_xb.shape[-1], branch1_layers = branch1_layers, branch1_activation=branch1_activation,
                    branch2_input_dim=test_tensor_yb.shape[-1], branch2_layers=branch2_layers, branch2_activation=branch2_activation,
                    branch3_input_dim=test_tensor_zb.shape[-1], branch3_layers=branch3_layers, branch3_activation=branch3_activation,
                        trunk_input_dim=input_tensor_I.shape[-1], trunk_layers=trunk_layers, trunk_activation=trunk_activation)

    trained_deeponet_save_path = os.path.join(folder_path, "trained_DeepONet.pth")
    state_dict_DeepONet = torch.load(trained_deeponet_save_path)
    model.load_state_dict(state_dict_DeepONet)
    model.eval()

    with torch.no_grad():
        G_0 = model(test_tensor_xb, test_tensor_yb, test_tensor_zb, input_tensor_0)
        G_xb = model(test_tensor_xb, test_tensor_yb, test_tensor_zb, input_tensor_xb)
        G_yb = model(test_tensor_xb, test_tensor_yb, test_tensor_zb, input_tensor_yb)
        G_zb = model(test_tensor_xb, test_tensor_yb, test_tensor_zb, input_tensor_zb)
        G_I = model(test_tensor_xb, test_tensor_yb, test_tensor_zb, input_tensor_I)

    # Function for converting the collocation points which were input into the model into a mesh.
    def CollocationToMesh(InitialPoints, xBoundaryPoints, yBoundaryPoints, zBoundaryPoints, InternalPoints, internal_x_len, internal_y_len, internal_z_len, internal_t_len, test_num):
        if InternalPoints.shape[-1] != 1: # This option works for manipulating the 4D input points.
            # First getting the internal collocation points.
            mesh_x = InternalPoints[:, :, 0].view(test_num, internal_x_len, internal_y_len, internal_z_len, internal_t_len)
            mesh_y = InternalPoints[:, :, 1].view(test_num, internal_x_len, internal_y_len, internal_z_len, internal_t_len)
            mesh_z = InternalPoints[:, :, 2].view(test_num, internal_x_len, internal_y_len, internal_z_len, internal_t_len)
            mesh_t = InternalPoints[:, :, 3].view(test_num, internal_x_len, internal_y_len, internal_z_len, internal_t_len)
            # print(mesh_x.shape, mesh_y.shape, mesh_t.shape)
            # Now getting initial points and combining with mesh.
            mesh_0_x = InitialPoints[:, :, 0].view(test_num, internal_x_len, internal_y_len, internal_z_len).unsqueeze(4)
            mesh_0_y = InitialPoints[:, :, 1].view(test_num, internal_x_len, internal_y_len, internal_z_len).unsqueeze(4)
            mesh_0_z = InitialPoints[:, :, 2].view(test_num, internal_x_len, internal_y_len, internal_z_len).unsqueeze(4)
            mesh_0_t = InitialPoints[:, :, 3].view(test_num, internal_x_len, internal_y_len, internal_z_len).unsqueeze(4)
            # print(mesh_0_x.shape, mesh_0_y.shape, mesh_0_t.shape)
            # print(mesh_x.shape, mesh_y.shape, mesh_t.shape)

            # Combine mesh and initial points.
            mesh_x = torch.cat((mesh_0_x, mesh_x), 4)
            mesh_y = torch.cat((mesh_0_y, mesh_y), 4)
            mesh_z = torch.cat((mesh_0_z, mesh_z), 4)
            mesh_t = torch.cat((mesh_0_t, mesh_t), 4)
            # Now getting the zboundary points.
            mesh_zb_x = zBoundaryPoints[:, :, 0].view(test_num, internal_x_len, internal_y_len, internal_t_len+1).unsqueeze(3)
            mesh_zb_y = zBoundaryPoints[:, :, 1].view(test_num, internal_x_len, internal_y_len, internal_t_len+1).unsqueeze(3)
            mesh_zb_z = zBoundaryPoints[:, :, 2].view(test_num, internal_x_len, internal_y_len, internal_t_len+1).unsqueeze(3)
            mesh_zb_t = zBoundaryPoints[:, :, 3].view(test_num, internal_x_len, internal_y_len, internal_t_len+1).unsqueeze(3)
            # Combine mesh and zboundary points.
            mesh_x = torch.cat((mesh_zb_x, mesh_x), 3)
            mesh_y = torch.cat((mesh_zb_y, mesh_y), 3)
            mesh_z = torch.cat((mesh_zb_z, mesh_z), 3)
            mesh_t = torch.cat((mesh_zb_t, mesh_t), 3)
            # Now getting the yboundary points.
            mesh_yb_x = yBoundaryPoints[:, :, 0].view(test_num, internal_x_len, internal_z_len+1, internal_t_len+1).unsqueeze(2)
            mesh_yb_y = yBoundaryPoints[:, :, 1].view(test_num, internal_x_len, internal_z_len+1, internal_t_len+1).unsqueeze(2)
            mesh_yb_z = yBoundaryPoints[:, :, 2].view(test_num, internal_x_len, internal_z_len+1, internal_t_len+1).unsqueeze(2)
            mesh_yb_t = yBoundaryPoints[:, :, 3].view(test_num, internal_x_len, internal_z_len+1, internal_t_len+1).unsqueeze(2)
            # print(mesh_b_x.shape, mesh_b_y.shape, mesh_b_t.shape)
            # Combine mesh and yboundary points.
            mesh_x = torch.cat((mesh_yb_x, mesh_x), 2)
            mesh_y = torch.cat((mesh_yb_y, mesh_y), 2)
            mesh_z = torch.cat((mesh_yb_z, mesh_z), 2)
            mesh_t = torch.cat((mesh_yb_t, mesh_t), 2)
            # print(mesh_x.shape, mesh_y.shape, mesh_t.shape)
            # Now getting the xboundary points.
            mesh_xb_x = xBoundaryPoints[:, :, 0].view(test_num, internal_y_len+1, internal_z_len+1, internal_t_len+1).unsqueeze(1)
            mesh_xb_y = xBoundaryPoints[:, :, 1].view(test_num, internal_y_len+1, internal_z_len+1, internal_t_len+1).unsqueeze(1)
            mesh_xb_z = xBoundaryPoints[:, :, 2].view(test_num, internal_y_len+1, internal_z_len+1, internal_t_len+1).unsqueeze(1)
            mesh_xb_t = xBoundaryPoints[:, :, 3].view(test_num, internal_y_len+1, internal_z_len+1, internal_t_len+1).unsqueeze(1)
            # print(mesh_xb_x.shape, mesh_xb_y.shape, mesh_xb_t.shape)
            # Combine mesh and xboundary points.
            mesh_x = torch.cat((mesh_xb_x, mesh_x), 1)
            mesh_y = torch.cat((mesh_xb_y, mesh_y), 1)
            mesh_z = torch.cat((mesh_xb_z, mesh_z), 1)
            mesh_t = torch.cat((mesh_xb_t, mesh_t), 1)

            return mesh_x, mesh_y, mesh_z, mesh_t

        else: # This option works for manipulating the 1D output points.
            reshaped_G = InternalPoints[:, :, 0].view(test_num, internal_x_len, internal_y_len, internal_z_len, internal_t_len)
            # Now getting initial points and combining with mesh.
            G_0 = InitialPoints[:,:,0].view((test_num, internal_x_len, internal_y_len, internal_z_len)).unsqueeze(4)
            reshaped_G = torch.cat((G_0, reshaped_G), 4)
            # Now getting the zboundary points.
            G_zb = zBoundaryPoints[:, :, 0].view(test_num, internal_x_len, internal_y_len, internal_t_len+1).unsqueeze(3)
            # Combine internal and yboundary points.
            reshaped_G = torch.cat((G_zb, reshaped_G), 3)
            # Now getting the yboundary points.
            G_yb = yBoundaryPoints[:, :, 0].view(test_num, internal_x_len, internal_z_len+1, internal_t_len+1).unsqueeze(2)
            # Combine internal and yboundary points.
            reshaped_G = torch.cat((G_yb, reshaped_G), 2)
            # Now getting the xboundary points.
            G_xb = xBoundaryPoints[:, :, 0].view(test_num, internal_y_len+1, internal_z_len+1, internal_t_len+1).unsqueeze(1)
            # Combine internal, yboundary, and xboundary points.
            reshaped_G = torch.cat((G_xb, reshaped_G), 1)

            return reshaped_G

    internal_x_len = int(collocation_points["nx"]) - 1
    internal_y_len = int(collocation_points["ny"]) - 1
    internal_z_len = int(collocation_points["nz"]) - 1
    internal_t_len = int(collocation_points["nt"]) - 1

    reshaped_G = CollocationToMesh(G_0, G_xb, G_yb, G_zb, G_I, internal_x_len, internal_y_len, internal_z_len, internal_t_len, test_num)
    mesh_x, mesh_y, mesh_z, mesh_t = CollocationToMesh(input_tensor_0, input_tensor_xb, input_tensor_yb, input_tensor_zb, input_tensor_I, internal_x_len, internal_y_len, internal_z_len, internal_t_len, test_num)

    # Function to output the exact solution but with constraint on the part of the domain impacted by boundary condition.
    def exact_f(X, Y, Z, T, A_xb, A_yb, A_zb, omegas, c, lower_xb, lower_yb, lower_zb, domain_limits):
        # Unpack spatial and temporal Gaussian widths
        omega_x, omega_y, omega_z, omega_t = omegas[0], omegas[1], omegas[2], omegas[3]

        T = T.detach().numpy()
        X = X.detach().numpy()
        Y = Y.detach().numpy()
        Z = Z.detach().numpy()
        lower_xb = lower_xb.detach().numpy()
        lower_yb = lower_yb.detach().numpy()
        lower_zb = lower_zb.detach().numpy()
        c = c.detach().numpy()

        # Time and space centres
        t0 = (domain_limits[3,1] + domain_limits[3,0]) / 2
        x0 = (domain_limits[0,1] + domain_limits[0,0]) / 2
        y0 = (domain_limits[1,1] + domain_limits[1,0]) / 2
        z0 = (domain_limits[2,1] + domain_limits[2,0]) / 2

        # Compute characteristic condition: only include points influenced by boundary
        characteristic_condition_x = T - (X - lower_xb) / c[0] >= 0
        characteristic_condition_y = T - (Y - lower_yb) / c[1] >= 0
        characteristic_condition_z = T - (Z - lower_zb) / c[2] >= 0

        # Compute shifted time and space for the x boundary condition impact
        shifted_T_x = T - ((X - lower_xb) / c[0])
        shifted_Y_x = Y - c[1] * ((X - lower_xb) / c[0])
        shifted_Z_x = Z - c[2] * ((X - lower_xb) / c[0])

        # Compute characteristic condition for y boundary condition impact
        shifted_T_y = T - (Y - lower_yb) / c[1]
        shifted_X_y = X - c[0] * (Y - lower_yb) / c[1]
        shifted_Z_y = Z - c[2] * (Y - lower_yb) / c[1]

        # Compute characteristic condition for z boundary condition impact
        shifted_T_z = T - (Z - lower_zb) / c[2]
        shifted_X_z = X - c[0] * (Z - lower_zb) / c[2]
        shifted_Y_z = Y - c[1] * (Z - lower_zb) / c[2]
        # print(lower_zb)


        # Calculate the gaussians of the x boundary condition
        gaussian_T_x = np.exp(-((shifted_T_x - t0) ** 2) / (2 * omega_t ** 2))
        gaussian_Y_x = np.exp(-((shifted_Y_x - y0) ** 2) / (2 * omega_y ** 2))
        gaussian_Z_x = np.exp(-((shifted_Z_x - z0) ** 2) / (2 * omega_z ** 2))
        sol_x = A_xb * gaussian_T_x * gaussian_Y_x * gaussian_Z_x
        sol_x = np.where(characteristic_condition_x, sol_x, 0)

        # Calculate the gaussians of the y boundary condition
        gaussian_T_y = np.exp(-((shifted_T_y - t0) ** 2) / (2 * omega_t ** 2))
        gaussian_X_y = np.exp(-((shifted_X_y - x0) ** 2) / (2 * omega_x ** 2))
        gaussian_Z_y = np.exp(-((shifted_Z_y - z0) ** 2) / (2 * omega_z ** 2))
        sol_y = A_yb * gaussian_T_y * gaussian_X_y * gaussian_Z_y
        sol_y = np.where(characteristic_condition_y, sol_y, 0)
        
        # Calculate the gaussians of the z boundary condition
        gaussian_T_z = np.exp(-((shifted_T_z - t0) ** 2) / (2 * omega_t ** 2))
        gaussian_X_z = np.exp(-((shifted_X_z - x0) ** 2) / (2 * omega_x ** 2))
        gaussian_Y_z = np.exp(-((shifted_Y_z - y0) ** 2) / (2 * omega_y ** 2))
        sol_z = A_zb * gaussian_T_z * gaussian_X_z * gaussian_Y_z
        sol_z = np.where(characteristic_condition_z, sol_z, 0)

        # Combine both boundary contributions
        solution = sol_x + sol_y + sol_z

        return solution

    # Preallocate the output tensor to store solutions for all 50 cases
    exact_solutions = np.zeros_like(mesh_x.detach().numpy())

    # Loop over each A value
    for i in range(test_num):
        # Compute the solution for the i-th A
        exact_solutions[i] = exact_f(mesh_x[i], mesh_y[i], mesh_z[i], mesh_t[i], test_As_xb[i], test_As_yb[i], test_As_zb[i], [omega_x, omega_y, omega_z, omega_t], c, mesh_x[i,0,0,0,0], mesh_y[i,0,0,0,0], mesh_z[i,0,0,0,0], domain_limits)
    # print(mesh_x.shape)
    #-----------------------------------------------------------------------------------------
    # Save out the model solution and the exact solutions.
    #-----------------------------------------------------------------------------------------
    # Convert tensors to float16 before saving.
    reshaped_G = reshaped_G.half()
    mesh_x = mesh_x.half()
    mesh_y = mesh_y.half()
    mesh_z = mesh_z.half()
    mesh_t = mesh_t.half()
    exact_solutions = exact_solutions.astype(np.float16)

    trained_model_save_path = os.path.join(folder_path, "trained_model_solution.zip")
    # Save directly into ZIP without creating an intermediate .pkl file
    with zipfile.ZipFile(trained_model_save_path, "w", compression=zipfile.ZIP_LZMA) as zipf:
        with io.BytesIO() as buffer:  # Create in-memory binary stream
            pickle.dump({'model_output': reshaped_G, 'x_mesh': mesh_x, 'y_mesh': mesh_y, 'z_mesh': mesh_z, 't_mesh': mesh_t, 'exact_solutions': exact_solutions}, buffer)  # Save to buffer
            buffer.seek(0)  # Reset buffer position
            zipf.writestr("trained_model_solution.pkl", buffer.read())  # Save buffer to ZIP
