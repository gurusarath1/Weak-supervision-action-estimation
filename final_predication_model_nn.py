import torch
import torch.nn as nn

class Final_pred_nn(nn.Module):

    def __init__(self, num_in, num_out, num_hid_neurons):
        super(Final_pred_nn, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=num_in, out_features=num_hid_neurons),
            nn.ReLU(),
            nn.Linear(in_features=num_hid_neurons, out_features=num_hid_neurons),
            nn.ReLU(),
            nn.Linear(in_features=num_hid_neurons, out_features=num_hid_neurons),
            nn.ReLU(),
            nn.Linear(in_features=num_hid_neurons, out_features=num_hid_neurons),
            nn.ReLU(),
            nn.Linear(in_features=num_hid_neurons, out_features=num_out),
        )

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.01, std=1.5)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.01, std=1.5)

    def forward(self, x):
        return self.model(x)
