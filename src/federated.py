import copy
import torch

class FederatedAveraging:
    """
    Simple federated averaging.
    """

    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_models):
        averaged_model = copy.deepcopy(self.global_model)
        global_dict = averaged_model.state_dict()

        # Sum parameters from all client models
        for k in global_dict.keys():
            clients_weights = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0)
            global_dict[k] = torch.mean(clients_weights, dim=0)
        averaged_model.load_state_dict(global_dict)
        self.global_model = averaged_model
        return averaged_model
