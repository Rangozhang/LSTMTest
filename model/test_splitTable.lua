require 'nn'
require 'nngraph'

x = nn.Identity()()
local reshaped = nn.Reshape(4, 3)(x)
local output = nn.SplitTable(2)(reshaped)
local output_concat = nn.JoinTable(2){nn.SelectTable(1)(output), nn.SelectTable(2)(output)}
local m = nn.gModule({x}, {output_concat})

input = torch.randn(3, 12)
print(input)
print(m:forward(input))

LSTM_mc = require 'LSTM_mc'
mm = LSTM_mc.lstm(10, 5, 10, 2, 0, 5, true)
