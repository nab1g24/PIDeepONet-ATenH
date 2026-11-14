import numpy as np
import torch
import torch.nn as nn
import random
import os
import pickle
import gzip

#-----------------------------------------------------------------------------------------
# Function to generate the lower boundary condition.
#-----------------------------------------------------------------------------------------
def lower_boundary(t, A, domain_limits, omega):
    return A * np.exp(-((t - ((domain_limits[1,1] + domain_limits[1,0]) / 2)) ** 2) / (2 * omega ** 2))

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
                 branch_input_dim, branch_layers, Branch_Activation,
                 trunk_input_dim, trunk_layers, Trunk_Activation):
        super().__init__()

        if Branch_Activation == "tanh":
            Branch_ActFunc=nn.Tanh
        elif Branch_Activation == "gelu":
            Branch_ActFunc=nn.GELU

        # Branch for the x boundary condition: no last activation, no last bias
        self.branch_net = build_mlp(
            branch_input_dim,
            branch_layers,
            activation=Branch_ActFunc,
            last_layer_has_activation=False,
            last_layer_has_bias=True
        )

        if Trunk_Activation == "tanh":
            Trunk_ActFunc=nn.Tanh
        elif Trunk_Activation == "gelu":
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

    def forward(self, branch_input, trunk_input):
        # --- Branch Network Forward Pass ---
        branch_output = self.branch_net(branch_input)  # Shape: [n, features]

        # --- Trunk Network Forward Pass ---
        trunk_output = self.trunk_net(trunk_input)   # Shape: [n, p, features]

        # --- Compute Final Output ---
        output = torch.einsum('npf,nf->np', trunk_output, branch_output) + self.output_bias # [n, p]

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
    
    # Creating an identity tensor to extract the element-wise gradients. Avoids
    # contributions to gradients from alternate points.
    batchnum, pointnum, _ = dy.shape
    grad_outputs = torch.zeros_like(dy)  # Initialize zeros
    grad_outputs[:, range(pointnum), 0] = 1  # Set only diagonal to 1

    # Calculate the gradients (derivatives of h w.r.t x and t), create_graph ensures 
    # further derivatives can be calculated.
    dGdX = torch.autograd.grad(outputs=dy, inputs=dX, grad_outputs=grad_outputs, 
                               retain_graph=True, create_graph=True, allow_unused=False)[0]
    
    # Filter out gradients w.r.t initial and boundary domain points.
    dG_dx = dGdX[:,:,0] # Extracting gradients w.r.t x
    dG_dt = dGdX[:,:,1] # Extracting gradients w.r.t t

    # Advection equation
    # print(c)
    f = torch.add(dG_dt, c * dG_dx)
    return f

#-----------------------------------------------------------------------------------------
# Functions for calculating the loss.
#-----------------------------------------------------------------------------------------
def loss_initial(G, G0, mse_loss):
    # For initial, compare against 'exact' solution.
    MSE_0 = mse_loss(G[:,:,0], G0[:,:])
    return MSE_0

def loss_boundary(G, exact_boundary, mse_loss):
    # For boundary, compare against 'exact' solution.
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
# Function for creating the mesh points and the collocation points used in testing the model.
#-----------------------------------------------------------------------------------------
# Function to create meshgrid and split into initial, boundary and internal collocation points.
def mesh_building(ns, limits, tensor=True, for_input_function=True):
    if for_input_function:
        t = np.linspace(limits[1,0], limits[1,1], ns[0])

        # Save the mesh points.
        input_function_sensor_points = {
            "tlinspace": t,
            "nt": ns[0]
        }
        return input_function_sensor_points
    
    else:
        x = np.linspace(limits[0,0], limits[0,1], ns[0])
        t = np.linspace(limits[1,0], limits[1,1], ns[1])

        # Create the mesh grid
        X, T = np.meshgrid(x, t)

        # Want to separate the above x, y and t meshes into the initial, boundary and internal 
        # collocation points.
        # Initial points
        X_0 = X[0,1:] # Initial points
        T_0 = T[0,1:] # Initial points
        X_b = X[:,0] # lower bound
        T_b = T[:,0] # lower bound
        X = X[1:,1:] # Internal points
        T = T[1:,1:] # Internal points

        # Flatten the meshgrid arrays to create pairs of (x, t)
        X_flat = X.flatten() # Internal points
        T_flat = T.flatten() # Internal points

        # Getting the initial points in tensor form.
        input_data_0 = np.vstack([X_0, T_0]).T
        if tensor:
            input_tensor_0 = torch.tensor(input_data_0, dtype=torch.float32, requires_grad=False)
        else:
            input_tensor_0 = input_data_0

        # Getting the boundary points in tensor form.
        input_data_b = np.vstack([X_b, T_b]).T
        if tensor:
            input_tensor_b = torch.tensor(input_data_b, dtype=torch.float32, requires_grad=False)
        else:
            input_tensor_b = input_data_b

        # Getting the internal points in tensor form.
        input_data_I = np.vstack([X_flat, T_flat]).T
        if tensor:
            input_tensor_I = torch.tensor(input_data_I, dtype=torch.float32, requires_grad=False)
        else:
            input_tensor_I = input_data_I

        # Create collocation points as a class
        collocation_points = {
            "internal": input_tensor_I,
            "boundary": input_tensor_b,
            "initial": input_tensor_0,
            "nx": ns[0],
            "nt": ns[1]
        }
        
        return collocation_points

#-----------------------------------------------------------------------------------------
# Training loop
#-----------------------------------------------------------------------------------------
def training_loop(model, optimizer, mse_loss, train_tensor,
                  sampled_collocation_points, sampled_exact_solutions, 
                  c, num_epochs, lambda1, lambda2, lambda3, losses, losses_0, losses_b, losses_I, epochs):
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()

        G_initial = model(train_tensor, sampled_collocation_points["initial"])
        G_boundary = model(train_tensor, sampled_collocation_points["boundary"])
        G_internal = model(train_tensor, sampled_collocation_points["internal"])

        loss_0 = loss_initial(G_initial, sampled_exact_solutions["initial"], mse_loss)
        loss_b = loss_boundary(G_boundary, sampled_exact_solutions["boundary"], mse_loss)
        loss_I = loss_PDE(G_internal, sampled_collocation_points["internal"], c, mse_loss)
        loss = lambda1*loss_0 + lambda2*loss_b + lambda3*loss_I
        
        # backward pass.
        loss.backward()

        # update weights.
        optimizer.step()

        # Save epoch and loss
        epochs.append(epoch + 1)
        losses.append(loss.item())
        losses_0.append(loss_0.item())
        losses_b.append(loss_b.item())
        losses_I.append(loss_I.item())

        if (epoch+1) % (num_epochs/10) == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.6f}')

    return losses, losses_0, losses_b, losses_I, epochs, model

#-----------------------------------------------------------------------------------------
# Function to save_data
#-----------------------------------------------------------------------------------------
# Saving data for no validation loss.
def save_data(base_path, branch_layers, trunk_layers, Branch_Activation, Trunk_Activation, num_epochs, lambda1, lambda2, lambda3, 
              learning_rate, c, omega, input_function_ns, testing_collocation_ns, 
              SampleNum_0, SampleNum_b, SampleNum_I, 
              train_num, test_num, Processor,
              model, epochs, losses, losses_0, losses_b, losses_I,
              test_tensor, test_As,
              limits, collocation_points,
              runtime, train_time,
              Save_model=True, Save_Losses=True, Save_test_data=True, Save_mesh=True):
    
    # Convert branch and trunk layers to strings for folder naming.
    branch_str = "-".join(map(str, branch_layers))
    trunk_str = "-".join(map(str, trunk_layers))
    
    new_folder_name = f"CaseII_{num_epochs}_{lambda1}_{lambda2}_{lambda3}_{float(learning_rate.item()):.1e}_{branch_str}_{trunk_str}_{Branch_Activation}_{Trunk_Activation}_{c.item()}_{omega:.2f}_{input_function_ns[0]}_{testing_collocation_ns[0]}x{testing_collocation_ns[1]}_{SampleNum_0},{SampleNum_b},{SampleNum_I}_{train_num},{test_num}_{Processor}"

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
        file.write(f"Learning Rate: {float(learning_rate.item()):.1e}\n")
        file.write(f"Branch layers: {branch_layers}\n")
        file.write(f"Trunk layers: {trunk_layers}\n")
        file.write(f"Branch activation: {Branch_Activation}\n")
        file.write(f"Trunk activation: {Trunk_Activation}\n")
        file.write(f"Advection speed: {c.item()}\n")
        file.write(f"Gaussian std dev: {omega}\n")
        file.write(f"Number of input function sensor points: {input_function_ns[0]}\n")
        file.write(f"Number of testing collocation points: {testing_collocation_ns[0]}x{testing_collocation_ns[1]}\n")
        file.write(f"Number of initial, boundary and internal collocation points: {SampleNum_0}, {SampleNum_b}, {SampleNum_I}\n")
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
            pickle.dump({'epochs': np.array(epochs), 'losses': np.array(losses), 'losses_initial': np.array(losses_0), 'losses_boundary': np.array(losses_b), 'losses_internal': np.array(losses_I)}, f)
        print(f"Losses saved successfully")

    # Saving the test data.
    if Save_test_data:
        testdata_save_path = os.path.join(new_folder_path, "test_data.pkl.gz")  # .gz extension to indicate it's compressed
        with gzip.open(testdata_save_path, 'wb') as f:
            pickle.dump({'BC': test_tensor, 'A': test_As, 'c': c, 'omega': omega, 'branch_layers': branch_layers, 'trunk_layers': trunk_layers, 'Branch_Activation': Branch_Activation, 'Trunk_Activation': Trunk_Activation, 'test_num': test_num, 'domain_limits': limits}, f)
        print(f"Test data saved successfully")

    # Saving the mesh points. Saved as [n, p, d] since this can be easily compressed to a much smaller size.
    if Save_mesh:
        mesh_save_path = os.path.join(new_folder_path, "collocation_points.pkl")
        with open(mesh_save_path, 'wb') as f:
            pickle.dump(collocation_points, f)
        print(f"Mesh points saved successfully")

    # Remove the previously saved trained_model_solution file if it exists as it won't match this run.
    trained_model_solutions_path = os.path.join(new_folder_path, "trained_model_solution.pkl")
    if os.path.exists(trained_model_solutions_path):
        os.remove(trained_model_solutions_path)
        print("Previous trained_model_solution file removed.") 
    print("Data saved successfully.")
