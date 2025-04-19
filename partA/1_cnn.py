import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self,
                 input_shape=(3, 224, 224),
                 conv_filters=[32, 64, 128, 256, 512],
                 kernel_sizes=[3, 3, 3, 3, 3],
                 activation_fn=nn.ReLU,
                 dense_neurons=256,
                 num_classes=10):
        super(SmallCNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.activation = activation_fn()

        in_channels = input_shape[0]
        self.feature_shapes = []
        x_dummy = torch.randn(1, *input_shape)  

        for i in range(5):
            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=conv_filters[i],
                             kernel_size=kernel_sizes[i],
                             padding=kernel_sizes[i] // 2)  
            self.conv_layers.append(conv)
            in_channels = conv_filters[i]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            for i in range(5):
                x_dummy = self.conv_layers[i](x_dummy)
                x_dummy = self.activation(x_dummy)
                x_dummy = self.pool(x_dummy)
            self.flattened_size = x_dummy.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, dense_neurons)
        self.fc2 = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = self.activation(x)
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
