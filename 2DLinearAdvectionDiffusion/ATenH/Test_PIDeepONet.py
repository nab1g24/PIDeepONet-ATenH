import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn as nn
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
    input_tensor_I = collocation_points["internal"]

    TestData_save_path = os.path.join(folder_path, "test_data.pkl")
    with open(TestData_save_path, 'rb') as f:
        test_data = pickle.load(f)

    test_tensor_xb = test_data["xBC"]
    test_tensor_yb = test_data["yBC"]
    branch1_layers = test_data["branch1_layers"]
    branch2_layers = test_data["branch2_layers"]
    trunk_layers = test_data["trunk_layers"]
    Branch1_Activation = test_data["Branch1_Activation"]
    Branch2_Activation = test_data["Branch2_Activation"]
    Trunk_Activation = test_data["Trunk_Activation"]
    test_num = test_tensor_xb.shape[0]

    # Expand the input tensors.
    input_tensor_0 = input_tensor_0.unsqueeze(0).expand(test_num, -1, -1)  # Shape: [train_num, SampleNum_0, input_dim]
    input_tensor_xb = input_tensor_xb.unsqueeze(0).expand(test_num, -1, -1)
    input_tensor_yb = input_tensor_yb.unsqueeze(0).expand(test_num, -1, -1)
    input_tensor_I = input_tensor_I.unsqueeze(0).expand(test_num, -1, -1)

    # Load the trained model.
    model = DeepONet(branch1_input_dim=test_tensor_xb.shape[-1], branch1_layers = branch1_layers, branch1_activation=Branch1_Activation,
                    branch2_input_dim=test_tensor_yb.shape[-1], branch2_layers=branch2_layers, branch2_activation=Branch2_Activation,
                    trunk_input_dim=input_tensor_I.shape[-1], trunk_layers=trunk_layers, trunk_activation=Trunk_Activation)

    trained_deeponet_save_path = os.path.join(folder_path, "trained_DeepONet.pth")
    state_dict_DeepONet = torch.load(trained_deeponet_save_path)
    model.load_state_dict(state_dict_DeepONet)
    model.eval()

    with torch.no_grad():
        G_0 = model(test_tensor_xb, test_tensor_yb, input_tensor_0)
        G_xb = model(test_tensor_xb, test_tensor_yb, input_tensor_xb)
        G_yb = model(test_tensor_xb, test_tensor_yb, input_tensor_yb)
        G_I = model(test_tensor_xb, test_tensor_yb, input_tensor_I)

    # Function for converting the collocation points which were input into the model into a mesh.
    def CollocationToMesh(InitialPoints, xBoundaryPoints, yBoundaryPoints, InternalPoints, internal_x_len, internal_y_len, internal_t_len, test_num):
        if InternalPoints.shape[2] != 1: # This option works for manipulating the 3D input points.
            # First getting the internal collocation points.
            mesh_x = InternalPoints[:, :, 0].view(test_num, internal_x_len, internal_y_len, internal_t_len)
            mesh_y = InternalPoints[:, :, 1].view(test_num, internal_x_len, internal_y_len, internal_t_len)
            mesh_t = InternalPoints[:, :, 2].view(test_num, internal_x_len, internal_y_len, internal_t_len)
            # print(mesh_x.shape, mesh_y.shape, mesh_t.shape)
            # Now getting initial points and combining with mesh.
            mesh_0_x = InitialPoints[:, :, 0].view(test_num, internal_x_len, internal_y_len).unsqueeze(3)
            mesh_0_y = InitialPoints[:, :, 1].view(test_num, internal_x_len, internal_y_len).unsqueeze(3)
            mesh_0_t = InitialPoints[:, :, 2].view(test_num, internal_x_len, internal_y_len).unsqueeze(3)
            # Combine mesh and initial points.
            mesh_x = torch.cat((mesh_0_x, mesh_x), 3)
            mesh_y = torch.cat((mesh_0_y, mesh_y), 3)
            mesh_t = torch.cat((mesh_0_t, mesh_t), 3)
            # Now getting the yboundary points.
            mesh_yb_x = yBoundaryPoints[:, :, 0].view(test_num, internal_x_len, internal_t_len+1).unsqueeze(2)
            mesh_yb_y = yBoundaryPoints[:, :, 1].view(test_num, internal_x_len, internal_t_len+1).unsqueeze(2)
            mesh_yb_t = yBoundaryPoints[:, :, 2].view(test_num, internal_x_len, internal_t_len+1).unsqueeze(2)
            # print(mesh_b_x.shape, mesh_b_y.shape, mesh_b_t.shape)
            # Combine mesh and yboundary points.
            mesh_x = torch.cat((mesh_yb_x, mesh_x), 2)
            mesh_y = torch.cat((mesh_yb_y, mesh_y), 2)
            mesh_t = torch.cat((mesh_yb_t, mesh_t), 2)
            # print(mesh_x.shape, mesh_y.shape, mesh_t.shape)
            # Now getting the xboundary points.
            mesh_xb_x = xBoundaryPoints[:, :, 0].view(test_num, internal_y_len+1, internal_t_len+1).unsqueeze(1)
            mesh_xb_y = xBoundaryPoints[:, :, 1].view(test_num, internal_y_len+1, internal_t_len+1).unsqueeze(1)
            mesh_xb_t = xBoundaryPoints[:, :, 2].view(test_num, internal_y_len+1, internal_t_len+1).unsqueeze(1)
            # print(mesh_xb_x.shape, mesh_xb_y.shape, mesh_xb_t.shape)
            # Combine mesh and xboundary points.
            mesh_x = torch.cat((mesh_xb_x, mesh_x), 1)
            mesh_y = torch.cat((mesh_xb_y, mesh_y), 1)
            mesh_t = torch.cat((mesh_xb_t, mesh_t), 1)

            return mesh_x, mesh_y, mesh_t

        else: # This option works for manipulating the 1D output points.
            reshaped_G = InternalPoints[:, :, 0].view(test_num, internal_x_len, internal_y_len, internal_t_len)
            # Now getting initial points and combining with mesh.
            G_0 = InitialPoints[:,:,0].view((test_num, internal_x_len, internal_y_len)).unsqueeze(3)
            reshaped_G = torch.cat((G_0, reshaped_G), 3)
            # Now getting the yboundary points.
            G_yb = yBoundaryPoints[:, :, 0].view(test_num, internal_x_len, internal_t_len+1).unsqueeze(2)
            # Combine internal and yboundary points.
            reshaped_G = torch.cat((G_yb, reshaped_G), 2)
            # Now getting the xboundary points.
            G_xb = xBoundaryPoints[:, :, 0].view(test_num, internal_y_len+1, internal_t_len+1).unsqueeze(1)
            # Combine internal, yboundary, and xboundary points.
            reshaped_G = torch.cat((G_xb, reshaped_G), 1)

            return reshaped_G

    internal_x_len = int(collocation_points["nx"]) - 1
    internal_y_len = int(collocation_points["ny"]) - 1
    internal_t_len = int(collocation_points["nt"]) - 1

    reshaped_G = CollocationToMesh(G_0, G_xb, G_yb, G_I, internal_x_len, internal_y_len, internal_t_len, test_num)
    mesh_x, mesh_y, mesh_t = CollocationToMesh(input_tensor_0, input_tensor_xb, input_tensor_yb, input_tensor_I, internal_x_len, internal_y_len, internal_t_len, test_num)

    #-----------------------------------------------------------------------------------------
    # Save out the model solution and mesh points.
    #-----------------------------------------------------------------------------------------
    # Convert tensors to float16 before saving.
    reshaped_G = reshaped_G.half()
    mesh_x = mesh_x.half()
    mesh_y = mesh_y.half()
    mesh_t = mesh_t.half()

    trained_model_save_path = os.path.join(folder_path, "trained_model_solution.zip")
    # Save directly into ZIP without creating an intermediate .pkl file
    with zipfile.ZipFile(trained_model_save_path, "w", compression=zipfile.ZIP_LZMA) as zipf:
        with io.BytesIO() as buffer:  # Create in-memory binary stream
            pickle.dump({'model_output': reshaped_G, 'x_mesh': mesh_x, 'y_mesh': mesh_y, 't_mesh': mesh_t}, buffer)  # Save to buffer
            buffer.seek(0)  # Reset buffer position
            zipf.writestr("trained_model_solution.pkl", buffer.read())  # Save buffer to ZIP
