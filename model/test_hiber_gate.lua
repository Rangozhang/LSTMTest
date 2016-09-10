require 'nn'
require 'nngraph'
require 'hiber_gate'

local model = hiber_gate(10, 5, 3)
local h = torch.randn(32, 10)
local input = torch.randn(32, 5)

local res = model:forward{h, input}
print(res)
