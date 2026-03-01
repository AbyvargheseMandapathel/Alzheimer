import torch
import copy

def federated_average(client_weights):
    """
    Computes the federated average over a list of client state configurations.
    Since we use equal dataset sizes, it will be a simple mean.
    
    Args:
        client_weights (list): List of state_dict from each client.
        
    Returns:
        dict: The averaged global model state_dict.
    """
    global_weights = copy.deepcopy(client_weights[0])
    
    for key in global_weights.keys():
        for i in range(1, len(client_weights)):
            global_weights[key] += client_weights[i][key]
        
        # Scale down by number of clients
        # Use torch.div for integer/float agnostic division
        if 'Long' in global_weights[key].type():
            # For parameters like num_batches_tracked
            global_weights[key] = global_weights[key] // len(client_weights)
        else:
            global_weights[key] = torch.div(global_weights[key], len(client_weights))
            
    return global_weights
