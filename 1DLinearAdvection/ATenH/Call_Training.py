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

Processor = 'NvidiaPU' # Choose your processor: 'CPU' or 'NvidiaGPU'.

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
num_epochs = 50000

# Subnetwork architecture.
k = 40 # Number of neurons for the last layer of each the subnetwork.
# Define hidden layers (flexible, but both must end in k), can specify number of layers and neurons in each layer.
branch_layers = [40, 40, 40, 40, k]
trunk_layers = [40, 40, 40, 40, k]

# Subnetwork activation function.
Branch_Activation = "tanh"
Trunk_Activation = "tanh"

# Number of training and test samples.
train_num = 500
test_num = 50

# How many collocation points to sample.
SampleNum_0 = 10
SampleNum_b = 10
SampleNum_I = 100

# Loss component weights.
lambda1 = 1
lambda2 = 1
lambda3 = 1

#-----------------------------------------------------------------------------------------
## Physics parameters.
# Advection speed.
c = 1.0

# Max amplitude of BC.
max_A = 1
min_A = 0.1

# Width of Gaussian for BC.
omega = 0.07

#-----------------------------------------------------------------------------------------
## Mesh generation (for input function discretisation and testing query points).
# Number of mesh points in (x, t).
testing_collocation_ns = [50, 50]
input_function_ns = [50]

# Domain range in (x, t).
limits = [[0, 1], [0, 1]]

#-----------------------------------------------------------------------------------------
## Miscellaneous
# Seed for reproducibility.
seed = 80

#-----------------------------------------------------------------------------------------
# Training call
#-----------------------------------------------------------------------------------------
Call_PIDeepONet_Training(learning_rate, num_epochs, branch_layers, trunk_layers,
                        train_num, test_num, SampleNum_0, SampleNum_b, SampleNum_I,
                        lambda1, lambda2, lambda3, c, omega, max_A, min_A, testing_collocation_ns, input_function_ns, limits, seed,
                        Processor, Save_model=Save_model, Save_Losses=Save_Losses, Save_test_data=Save_test_data, Save_mesh=Save_mesh,
                        Set_NN_Seed=Set_NN_Seed, Branch_Activation=Branch_Activation, Trunk_Activation=Trunk_Activation)

