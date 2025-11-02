import torch

class FederatedAveraging:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_models):
        # Simple FedAvg
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_model.state_dict()[k].float() for client_model in client_models], 0).mean(0)
        self.global_model.load_state_dict(global_dict)
        return self.global_model
