require 'nn'
require 'print'

m = nn.Sequential()
m:add(nn.Linear(3, 5))
m:add(PrintLayer())
m:add(nn.Linear(5, 3))

criterion = nn.BCECriterion()

target = torch.Tensor{1, 0, 0}

x = torch.randn(3)
output = m:forward(x)
loss = criterion:forward(output, target)
print("loss")
print(loss)
dw = criterion:backward(output, target)
print("dw")
print(dw)
final = m:backward(x, dw)
print(final)
