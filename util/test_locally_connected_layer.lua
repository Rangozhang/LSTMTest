require 'nn'
require 'locally_connected_layer'

model = nn.LocallyConnected(4, 6, 2)
x = torch.randn(3, 4)
print(x)
print(model:forward(x))
x[{{}, {1, 2}}] = 0
print(x)
print(model:forward(x))
