import torch
import torch.nn as nn
import torch.nn.functional as F
class GAN_cnn(nn.Module):
    def __init__(self):
        super(GAN_cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=4,
            ),
            nn.Dropout(0.1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512, 256)

    def forward(self, inputs):
        cnn_ = self.conv(inputs.permute(1, 2, 0)) 
        cnn_out = self.fc(cnn_.transpose(1, 2)) 

        return torch.mean(cnn_out, dim=1) 


