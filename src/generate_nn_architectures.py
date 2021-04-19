#%%
import json
import numpy as np

# Specify how many random neural network architectures shall be newly
# generated, append to existing config file
ARCHITECTURES_TO_GENERATE = 1000

with open("./optimization_configs.json", "r") as f:
    hyperparameters = json.load(f)

MIN_WIDTH = hyperparameters["MIN_WIDTH"]  # Minimum layer width.
MAX_WIDTH = hyperparameters["MAX_WIDTH"]  # Maximum layer width.
MIN_DEPTH = hyperparameters["MIN_DEPTH"]  # Minimum network depth.
MAX_DEPTH = hyperparameters["MAX_DEPTH"]  # Maximum network depth.

try:
    with open("./nn_hidden_layers.json", "r") as f:
        hidden = json.load(f)
except:
    hidden = []

for _ in range(ARCHITECTURES_TO_GENERATE):
    # Generate random layer depth and widths. Order widths descending.
    hidden_layers = np.random.randint(
        MIN_WIDTH, MAX_WIDTH, np.random.randint(MIN_DEPTH, MAX_DEPTH)
    )

    hidden_layers = hidden_layers[np.argsort(-hidden_layers)]

    hidden.append(hidden_layers.tolist())

print("Number of generated architectures: ", len(hidden))

with open("./nn_hidden_layers.json", "w") as f:
    json.dump(hidden, f, indent=2)
