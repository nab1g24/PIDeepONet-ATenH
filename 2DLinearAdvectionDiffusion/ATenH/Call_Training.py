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

# Number of training epochs.
num_epochs = 100000

# Subnetwork architecture.
k = 60 # Number of neurons for the last layer of each the subnetwork.
# Define hidden layers (flexible, but both must end in k), can specify number of layers and neurons in each layer.
branch1_layers = [60, 60, 60, 60, k] # x boundary branch net layers
branch2_layers = [60, 60, 60, 60, k] # y boundary branch net layers
trunk_layers = [60, 60, 60, 60, k]

# Subnetwork activation function.
Branch1_Activation = "tanh"
Branch2_Activation = "tanh"
Trunk_Activation = "tanh"

# Number of training and test samples.
train_num = 500
test_num = 50

# How many collocation points to sample.
SampleNum_0 = 100
SampleNum_xb = 100
SampleNum_yb = 100
SampleNum_I = 1000

# Loss component weights.
lambda1 = 1
lambda2 = 1
lambda3 = 1
lambda4 = 1

#-----------------------------------------------------------------------------------------
## Physics parameters.
# Advection speed [cx, cy].
c = [1.0, 1.0]
nu = 0.05

# Gaussian standard deviation.
omega_x = 0.07
omega_y = 0.07
omega_t = 0.07

# Max and min amplitude of BC.
max_A = 1.0
min_A = 0.1

#-----------------------------------------------------------------------------------------
## Mesh generation (for input function discretisation and testing query points).
# Number of mesh points in (x, y, t).
input_function_ns = [50, 50, 50]
testing_collocation_ns = [30, 30, 30]
# Domain range in (x, y, t).
limits = [[0, 1], [0, 1], [0, 1]]

#-----------------------------------------------------------------------------------------
## Miscellaneous
# Seed for reproducibility.
seed = 80

#-----------------------------------------------------------------------------------------
# Training call.
#-----------------------------------------------------------------------------------------
Call_PIDeepONet_Training(
    learning_rate, num_epochs, branch1_layers, branch2_layers, trunk_layers,
    train_num, test_num, SampleNum_0, SampleNum_xb, SampleNum_yb, SampleNum_I,
    lambda1, lambda2, lambda3, lambda4, 
    c, nu, omega_x, omega_y, omega_t, max_A, min_A,
    testing_collocation_ns, input_function_ns, limits, seed, Processor,
    Save_model=Save_model, Save_Losses=Save_Losses,
    Save_test_data=Save_test_data, Save_mesh=Save_mesh, Set_NN_Seed=Set_NN_Seed,
    Branch1_Activation=Branch1_Activation,
    Branch2_Activation=Branch2_Activation,
    Trunk_Activation=Trunk_Activation
)
