from Train_PIDeepONet import *
from Functions.PIDeepONet import *

#-----------------------------------------------------------------------------------------
# Options for the script.
#-----------------------------------------------------------------------------------------
# Save out the model, mesh points, and losses and data for testing.
Save_model = True
Save_mesh = True
Save_Losses = True
Save_test_data = True

Processor = 'NvidiaGPU' # Choose your processor: 'CPU' or 'NvidiaGPU'.

# Seed for fixing the initial model parameters.
Set_NN_Seed = False

#-----------------------------------------------------------------------------------------
# Initialising variables
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
## Model set-up
# Learning rate for the optimizer.
learning_rate = 1e-4

num_epochs = 300000

# Subnetwork architecture.
k = 80 # Number of neurons for the last layer of each the subnetwork.
# Define hidden layers (flexible, but both must end in k), can specify number of layers and neurons in each layer.
branch1_layers = [80, 80, 80, 80, k] # x boundary branch net layers
branch2_layers = [80, 80, 80, 80, k] # y boundary branch net layers
branch3_layers = [80, 80, 80, 80, k] # t=0 branch net layers
trunk_layers = [80, 80, 80, 80, k]

# Subnetwork activation function.
Branch1_Activation = "tanh"
Branch2_Activation = "tanh"
Branch3_Activation = "tanh"
Trunk_Activation = "tanh"

# Number of training and test samples.
train_num = 500
test_num = 50

# How many collocation points to sample.
SampleNum_0 = 1000
SampleNum_xb = 1000
SampleNum_yb = 1000
SampleNum_zb = 1000
SampleNum_I = 10000

# Loss component weights.
lambda1 = 1
lambda2 = 1
lambda3 = 1
lambda4 = 1
lambda5 = 1

#-----------------------------------------------------------------------------------------
## Physics parameters.
# Advection speed [cx, cy, cz].
c = [1.0, 1.0, 1.0]

# Gaussian standard deviation.
omega_x = 0.07
omega_y = 0.07
omega_z = 0.07
omega_t = 0.07

# Max and min amplitude of BC.
max_A = 1
min_A = 0.1

#-----------------------------------------------------------------------------------------
## Mesh generation (for input function discretisation and testing query points).
# Number of mesh points in (x, y, t).
testing_collocation_ns = [10, 10, 10, 10]
input_function_ns = [50, 50, 50, 50]

# Domain range in (x, y, z, t).
limits = [[0, 1], [0, 1], [0, 1], [0, 1]]

#-----------------------------------------------------------------------------------------
## Miscellaneous
# Seed for reproducibility.
seed = 80

#-----------------------------------------------------------------------------------------
# Training call
#-----------------------------------------------------------------------------------------
Call_PIDeepONet_Training(learning_rate, num_epochs, branch1_layers, branch2_layers, branch3_layers, trunk_layers,
                        train_num, test_num, SampleNum_0, SampleNum_xb, SampleNum_yb, SampleNum_zb, SampleNum_I,
                        lambda1, lambda2, lambda3, lambda4, lambda5, c, omega_x, omega_y, omega_z, omega_t, max_A, min_A, testing_collocation_ns, input_function_ns, limits, seed,
                        Processor, Save_model=Save_model, Save_Losses=Save_Losses, Save_test_data=Save_test_data, Save_mesh=Save_mesh,
                        Set_NN_Seed=Set_NN_Seed, Branch1_Activation=Branch1_Activation, Branch2_Activation=Branch2_Activation, Branch3_Activation=Branch3_Activation, Trunk_Activation=Trunk_Activation)

