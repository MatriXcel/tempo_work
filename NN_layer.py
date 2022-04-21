import torch
import torch.nn as nn

class AMFNet(nn.Module):

    def __init__(self, input_size, hidden):
        super(AMFNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden)  # 5*5 from image dimension
        self.output_layer = nn.Linear(hidden, 1)

    def forward(self, x):
        fc1_z = self.fc1(x) #[batch_size, seq_len, hidden_dim]
        fc1_output = torch.relu(fc1_z)

        output_layer_z = self.output_layer(fc1_output)

        return output_layer_z

