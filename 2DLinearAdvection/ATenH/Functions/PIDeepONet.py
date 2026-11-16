import numpy as np
import torch
import torch.nn as nn
import random
import os
import pickle
import zipfile
import io

#-----------------------------------------------------------------------------------------
# Function to generate the lower boundary conditions.
#-----------------------------------------------------------------------------------------
def lower_xboundary(y, t, A, domain_limits, omegas, point_type):
    if point_type == 'mesh':
        T, Y = np.meshgrid(t, y)
        BC = A * np.exp(-((T - ((domain_limits[2,1] + domain_limits[2,0]) / 2)) ** 2) / (2 * omegas[1] ** 2)) * np.exp(-((Y - ((domain_limits[1,1] + domain_limits[1,0]) / 2)) ** 2) / (2 * omegas[0] ** 2))
    elif point_type == 'collocation':
        BC = A * np.exp(-((t - ((domain_limits[2,1] + domain_limits[2,0]) / 2)) ** 2) / (2 * omegas[1] ** 2)) * np.exp(-((y - ((domain_limits[1,1] + domain_limits[1,0]) / 2)) ** 2) / (2 * omegas[0] ** 2))
    return BC

def lower_yboundary(x, t, A, domain_limits, omegas, point_type):
    if point_type == 'mesh':
        T, X = np.meshgrid(t, x)
        BC = A * np.exp(-((T - ((domain_limits[2,1] + domain_limits[2,0]) / 2)) ** 2) / (2 * omegas[1] ** 2)) * np.exp(-((X - ((domain_limits[0,1] + domain_limits[0,0]) / 2)) ** 2) / (2 * omegas[0] ** 2))
    elif point_type == 'collocation':
        BC = A * np.exp(-((t - ((domain_limits[2,1] + domain_limits[2,0]) / 2)) ** 2) / (2 * omegas[1] ** 2)) * np.exp(-((x - ((domain_limits[0,1] + domain_limits[0,0]) / 2)) ** 2) / (2 * omegas[0] ** 2))
    return BC

#-----------------------------------------------------------------------------------------
# Set the seed value for reproducibility. Only seems to fix the learning process if ran on the CPU
#-----------------------------------------------------------------------------------------
def set_seed(seed_value, Processor):
    # Set seed for CPU
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    if Processor == 'CPU':
        print("Using CPU. Seeding set.")
        
    elif Processor == 'NvidiaGPU':
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("Using NVIDIA GPU. Seeding set.")
        else:
            print("CUDA not available. Falling back to CPU.")

#-----------------------------------------------------------------------------------------
# Function for randomly sampling collocation points.
#-----------------------------------------------------------------------------------------
def rand_sample(array, N):
    array_len = np.size(array,0)
    # Randomly sampling locations. 
    random_idxs = np.random.choice(array_len, N, replace=False)
    # Sort the numbers.
    sorted_random_idxs = np.sort(random_idxs)
    # Select the values from the array corresponding to the sampled indexes.
    random_array = array[sorted_random_idxs]
    return random_array

#-----------------------------------------------------------------------------------------
# Function for creating the neural networks.
#-----------------------------------------------------------------------------------------
# Making Branch Net inline with DeepONet pape. No bias in last layer.
def build_mlp(input_dim, layer_dims, activation=nn.Tanh, last_layer_has_activation=True, last_layer_has_bias=True):
    layers = []
    for i, dim in enumerate(layer_dims):
        is_last = (i == len(layer_dims) - 1)

        layers.append(
            nn.Linear(
                input_dim if i == 0 else layer_dims[i - 1],
                dim,
                bias=last_layer_has_bias if is_last else True
            )
        )

        if not is_last:
            layers.append(activation())
        elif last_layer_has_activation:
            layers.append(activation())

    return nn.Sequential(*layers)

class DeepONet(nn.Module):
    def __init__(self, 
                 branch1_input_dim, branch1_layers, branch1_activation,
                 branch2_input_dim, branch2_layers, branch2_activation,
                 trunk_input_dim, trunk_layers, trunk_activation):
        super().__init__()

        if branch1_activation == "tanh":
            Branch1_ActFunc=nn.Tanh
        elif branch1_activation == "gelu":
            Branch1_ActFunc=nn.GELU

        # Branch for the x boundary condition: no last activation, no last bias
        self.branch1_net = build_mlp(
            branch1_input_dim,
            branch1_layers,
            activation=Branch1_ActFunc,
            last_layer_has_activation=False,
            last_layer_has_bias=True
        )

        if branch2_activation == "tanh":
            Branch2_ActFunc=nn.Tanh
        elif branch2_activation == "gelu":
            Branch2_ActFunc=nn.GELU

        # Branch for the y boundary condition: no last activation, no last bias
        self.branch2_net = build_mlp(
            branch2_input_dim,
            branch2_layers,
            activation=Branch2_ActFunc,
            last_layer_has_activation=False,
            last_layer_has_bias=True
        )

        if trunk_activation == "tanh":  
            Trunk_ActFunc=nn.Tanh
        elif trunk_activation == "gelu":
            Trunk_ActFunc=nn.GELU

        # Trunk: last activation, normal bias
        self.trunk_net = build_mlp(
            trunk_input_dim,
            trunk_layers,
            activation=Trunk_ActFunc,
            last_layer_has_activation=True,
            last_layer_has_bias=True
        )

        # --- Output bias (scalar) ---
        self.output_bias = nn.Parameter(torch.tensor(0.0))

        self._initialize_weights()

    def forward(self, branch1_input, branch2_input, trunk_input):
        # --- Branch Forward ---
        branch1_output = self.branch1_net(branch1_input)  # [n, features]
        branch2_output = self.branch2_net(branch2_input)  # [n, features]

        # --- Trunk Forward ---
        trunk_output = self.trunk_net(trunk_input)  # [n, p, features]

        # Do a dot product between trunk and branch outputs.
        output = torch.einsum('nf,nf,npf->np', branch1_output, branch2_output, trunk_output) + self.output_bias  # [n, p]

        return output.unsqueeze(-1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

#-----------------------------------------------------------------------------------------
# Function for the governing PDE.
#-----------------------------------------------------------------------------------------
def advection(dy, dX, c):
    
    # Calculate the gradients (derivatives of h w.r.t x and t), create_graph ensures 
    # further derivatives can be calculated.
    dGdX = torch.autograd.grad(outputs=dy, inputs=dX, grad_outputs=torch.ones_like(dy), 
                            retain_graph=True, create_graph=True, allow_unused=False)[0]

    # Filter out gradients w.r.t initial and boundary domain points.
    dG_dx = dGdX[:,:,0] # Extracting gradients w.r.t x
    dG_dy = dGdX[:,:,1] # Extracting gradients w.r.t y
    dG_dt = dGdX[:,:,2] # Extracting gradients w.r.t t

    # Advection equation
    f = dG_dt + c[0] * dG_dx + c[1] * dG_dy
    return f

#-----------------------------------------------------------------------------------------
# Functions for calculating the loss.
#-----------------------------------------------------------------------------------------
def loss_initial(G, G0, mse_loss):
    # For initial, compare against 'exact' solution.
    MSE_0 = mse_loss(G[:,:,0], G0[:,:])
    return MSE_0

def loss_boundary(G, exact_boundary, mse_loss):
    # For boundaries, take every other element from h. This is due to the flattening done 
    # during domain creation causing the two boundaries to alternate.
    MSE_b = mse_loss(G[:,:,0], exact_boundary)
    
    return MSE_b

def loss_PDE(G, variables, c, mse_loss):
    # For internal, use the governing PDE.
    f = advection(G, variables, c)

    # Target is zero for residual (since we want the residual to be small)
    target_residual = torch.zeros_like(f, dtype=torch.float32)
    
    # Compute loss for internal points based on the residual
    MSE_I = mse_loss(f, target_residual)
    return MSE_I

#-----------------------------------------------------------------------------------------
# Functions and classes for the meshes and collocation points. First dimension of meshes 
# correspond to the x dimension, second dimension for y, third dimension for t.
#-----------------------------------------------------------------------------------------
# Function to create meshgrid and split into initial, boundary and internal collocation points.
def mesh_building(ns, limits, tensor=True, for_input_function=True):
    if for_input_function:
        x = np.linspace(limits[0,0], limits[0,1], ns[0])
        y = np.linspace(limits[1,0], limits[1,1], ns[1])
        t = np.linspace(limits[2,0], limits[2,1], ns[2])

        # Save the mesh points.
        input_function_sensor_points = {
            "xlinspace": x,
            "ylinspace": y,
            "tlinspace": t,
            "nx": ns[0],
            "ny": ns[1],
            "nt": ns[2]
        }
        return input_function_sensor_points
    
    else:
        x = np.linspace(limits[0,0], limits[0,1], ns[0])
        y = np.linspace(limits[1,0], limits[1,1], ns[1])
        t = np.linspace(limits[2,0], limits[2,1], ns[2])

        # Create the mesh grid
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')

        # Want to separate the above x, y and t meshes into the initial, boundary and internal 
        # collocation points.
        # Initial points
        X_0 = X[1:,1:,0]
        Y_0 = Y[1:,1:,0]
        T_0 = T[1:,1:,0]
        # Boundary points
        X_xlb = X[0,:,:] # lower x boundary
        Y_xlb = Y[0,:,:] # lower x boundary
        T_xlb = T[0,:,:] # lower x boundary
        X_ylb = X[1:,0,:] # lower y boundary
        Y_ylb = Y[1:,0,:] # lower y boundary
        T_ylb = T[1:,0,:] # lower y boundary
        # Internal points, removing the boundary points in the first dimension and the initial points in the last dimension.
        X = X[1:,1:,1:]
        Y = Y[1:,1:,1:]
        T = T[1:,1:,1:]

        # Flatten the meshgrid arrays to create pairs of (x, y, t)
        X_0 = X_0.flatten() # Initial points
        Y_0 = Y_0.flatten() # Initial points
        T_0 = T_0.flatten() # Initial points
        X_xlb = X_xlb.flatten() # x Boundary points
        Y_xlb = Y_xlb.flatten() # x Boundary points
        T_xlb = T_xlb.flatten() # x Boundary points
        X_ylb = X_ylb.flatten() # y Boundary points
        Y_ylb = Y_ylb.flatten() # y Boundary points
        T_ylb = T_ylb.flatten() # y Boundary points
        X = X.flatten() # Internal points
        Y = Y.flatten() # Internal points
        T = T.flatten() # Internal points

        # Getting the initial points in tensor form.
        input_data_0 = np.vstack([X_0, Y_0 ,T_0]).T
        if tensor:
            input_tensor_0 = torch.tensor(input_data_0, dtype=torch.float32, requires_grad=False)
        else:
            input_tensor_0 = input_data_0

        # Getting the boundary points in tensor form. 
        input_data_xb = np.vstack([X_xlb, Y_xlb, T_xlb]).T
        if tensor:
            input_tensor_xb = torch.tensor(input_data_xb, dtype=torch.float32, requires_grad=False)
        else:
            input_tensor_xb = input_data_xb
        input_data_yb = np.vstack([X_ylb, Y_ylb, T_ylb]).T
        if tensor:
            input_tensor_yb = torch.tensor(input_data_yb, dtype=torch.float32, requires_grad=False)
        else:
            input_tensor_yb = input_data_yb

        # Getting the internal points in tensor form.
        input_data_I = np.vstack([X, Y, T]).T
        if tensor:
            input_tensor_I = torch.tensor(input_data_I, dtype=torch.float32, requires_grad=False)
        else:
            input_tensor_I = input_data_I

        # Create collocation points as a class
        collocation_points = {
            "internal": input_tensor_I,
            "xboundary": input_tensor_xb,
            "yboundary": input_tensor_yb,
            "initial": input_tensor_0,
            "nx": ns[0],
            "ny": ns[1],
            "nt": ns[2]
        }

        return collocation_points

#-----------------------------------------------------------------------------------------
# Training loop
#-----------------------------------------------------------------------------------------
def training_loop(model, optimizer, mse_loss, train_tensor_xb, train_tensor_yb, 
                  sampled_collocation_points, sampled_exact_solutions, 
                  c, num_epochs, lambda1, lambda2, lambda3, lambda4, losses, losses_0, losses_xb, losses_yb, losses_I, epochs):
    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()

        # Forward pass.
        G_initial = model(train_tensor_xb, train_tensor_yb, sampled_collocation_points["initial"])
        G_xboundary = model(train_tensor_xb, train_tensor_yb, sampled_collocation_points["xboundary"])
        G_yboundary = model(train_tensor_xb, train_tensor_yb, sampled_collocation_points["yboundary"])
        G_internal = model(train_tensor_xb, train_tensor_yb, sampled_collocation_points["internal"])

        loss_0 = loss_initial(G_initial, sampled_exact_solutions["initial"], mse_loss)
        loss_xb = loss_boundary(G_xboundary, sampled_exact_solutions["xboundary"], mse_loss)
        loss_yb = loss_boundary(G_yboundary, sampled_exact_solutions["yboundary"], mse_loss)
        loss_I = loss_PDE(G_internal, sampled_collocation_points["internal"], c, mse_loss)
        loss = lambda1*loss_0 + lambda2*loss_xb + lambda3*loss_yb + lambda4*loss_I

        # backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Save epoch and loss
        epochs.append(epoch + 1)
        losses.append(loss.item())
        losses_0.append(loss_0.item())
        losses_xb.append(loss_xb.item())
        losses_yb.append(loss_yb.item())
        losses_I.append(loss_I.item())

        if (epoch+1) % (num_epochs/10) == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    return losses, losses_0, losses_xb, losses_yb, losses_I, epochs, model

#-----------------------------------------------------------------------------------------
# Function to save_data
#-----------------------------------------------------------------------------------------
def save_data(base_path, branch1_layers, branch2_layers, trunk_layers, Branch1_Activation, Branch2_Activation, Trunk_Activation,
              num_epochs, lambda1, lambda2, lambda3, lambda4, 
              learning_rate, c, omega_x, omega_y, omega_t, input_function_ns, testing_collocation_ns,  
              SampleNum_0, SampleNum_xb, SampleNum_yb, SampleNum_I, 
              train_num, test_num, Processor,
              model, epochs, losses, losses_0, losses_xb, losses_yb, losses_I,
              test_tensor_xb, test_tensor_yb, test_As_xb, test_As_yb,
              limits, collocation_points,
              runtime, train_time,
              Save_model=True, Save_Losses=True, Save_test_data=True, Save_mesh=True):
    
    # Convert branch and trunk layers to strings for folder naming.
    branch1_str = "-".join(map(str, branch1_layers))
    branch2_str = "-".join(map(str, branch2_layers))
    trunk_str = "-".join(map(str, trunk_layers))
    
    new_folder_name = f"CaseII_{num_epochs}_{lambda1}_{lambda2}_{lambda3}_{lambda4}_{float(learning_rate.item()):.1e}_{branch1_str}_{branch2_str}_{trunk_str}_{Branch1_Activation}_{Branch2_Activation}_{Trunk_Activation}_[{c[0].item():.2f},{c[1].item():.2f}]_{omega_x:.2f}_{omega_y:.2f}_{omega_t:.2f}_{input_function_ns[0]}x{input_function_ns[1]}x{input_function_ns[2]}_{testing_collocation_ns[0]}x{testing_collocation_ns[1]}x{testing_collocation_ns[2]}_{SampleNum_0},{SampleNum_xb},{SampleNum_yb},{SampleNum_I}_{train_num},{test_num}_{Processor}"

    # Create the full path for the new folder
    new_folder_path = os.path.join(base_path, new_folder_name)
    # Create the new folder
    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Folder created successfully at: {new_folder_path}")
    except Exception as e:
        print(f"Error creating folder: {e}")

    # Making a txt file to desribe the parameters used.
    txt_file_path = os.path.join(new_folder_path, "README.txt")

    with open(txt_file_path, 'w') as file:
        file.write("Parameters used for this trained model.\n")
        file.write(f"Epoch number: {num_epochs}\n")
        file.write(f"Lambda_1: {lambda1}\n")
        file.write(f"Lambda_2: {lambda2}\n")
        file.write(f"Lambda_3: {lambda3}\n")
        file.write(f"Lambda_4: {lambda4}\n")
        file.write(f"Learning Rate: {float(learning_rate.item()):.1e}\n")
        file.write(f"Branch1 layers: {branch1_layers}\n")
        file.write(f"Branch2 layers: {branch2_layers}\n")
        file.write(f"Trunk layers: {trunk_layers}\n")
        file.write(f"Branch1 activation: {Branch1_Activation}\n")
        file.write(f"Branch2 activation: {Branch2_Activation}\n")
        file.write(f"Trunk activation: {Trunk_Activation}\n")
        file.write(f"Advection speed: [{c[0].item():.2f},{c[1].item():.2f}]\n")
        file.write(f"Input function omega values: {omega_x:.2f}, {omega_y:.2f}, {omega_t:.2f}\n")
        file.write(f"Number of input function sensor points: {input_function_ns[0]}x{input_function_ns[1]}x{input_function_ns[2]}\n")
        file.write(f"Number of testing collocation points: {testing_collocation_ns[0]}x{testing_collocation_ns[1]}x{testing_collocation_ns[2]}\n")
        file.write(f"Number of initial, boundary and internal collocation points: {SampleNum_0}, {SampleNum_xb}, {SampleNum_yb}, {SampleNum_I}\n")
        file.write(f"Number of training and test samples: {train_num}, {test_num}\n")
        file.write(f"Runtime: {runtime:.6f} seconds\n")
        file.write(f"Training time: {train_time:.6f} seconds\n")

    # Save the trained model inside the new folder
    if Save_model:
        model_save_path = os.path.join(new_folder_path, "trained_DeepONet.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved successfully")

    # Save the training losses inside the new folder
    if Save_Losses:
        loss_save_path = os.path.join(new_folder_path, "training_loss.pkl")
        with open(loss_save_path, 'wb') as f:
            pickle.dump({'epochs': np.array(epochs), 'losses': np.array(losses), 'losses_initial': np.array(losses_0), 'losses_xboundary': np.array(losses_xb), 'losses_yboundary': np.array(losses_yb), 'losses_internal': np.array(losses_I)}, f)
        print(f"Losses saved successfully")

    # Saving the test data.
    if Save_test_data:
        testdata_save_path = os.path.join(new_folder_path, "test_data.pkl")
        with open(testdata_save_path, 'wb') as f:
            pickle.dump({'xBC': test_tensor_xb, 'yBC': test_tensor_yb, 'omega_x': omega_x, 'omega_y': omega_y, 'omega_t': omega_t, 'A_xb': test_As_xb, 'A_yb': test_As_yb, 'c': c, 'branch1_layers': branch1_layers, 'branch2_layers': branch2_layers, 'trunk_layers': trunk_layers, 'Branch1_Activation': Branch1_Activation, 'Branch2_Activation': Branch2_Activation, 'Trunk_Activation': Trunk_Activation, 'test_num': test_num, 'domain_limits': limits}, f)
        print(f"Test data saved successfully")

    # # Saving the mesh points. 
    if Save_mesh:
        zip_save_path = os.path.join(new_folder_path, "collocation_points.zip")

        # Save directly into ZIP without creating an intermediate .pkl file
        with zipfile.ZipFile(zip_save_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            with io.BytesIO() as buffer:  # Create in-memory binary stream
                pickle.dump(collocation_points, buffer)  # Save to buffer
                buffer.seek(0)  # Reset buffer position
                zipf.writestr("collocation_points.pkl", buffer.read())  # Save buffer to ZIP

        print(f"Mesh points saved successfully")
        
    # Remove the previously saved trained_model_solution file if it exists as it won't match this run.
    trained_model_solutions_path = os.path.join(new_folder_path, "trained_model_solution.pkl")
    if os.path.exists(trained_model_solutions_path):
        os.remove(trained_model_solutions_path)
        print("Previous trained_model_solution file removed.") 
    print("Data saved successfully.")
