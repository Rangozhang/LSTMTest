local num_hidden_state = 3
local hidden_state_size = 2
local batch_size = 5
local num_class = 2

torch.manualSeed(123)

local x = torch.randn(batch_size, hidden_state_size*num_hidden_state)
local y = torch.Tensor{{1}, {2}, {2}, {1}, {1}}
y = torch.zeros(batch_size, 2):scatter(2, y:long(), 1)
print(x)
print(y)
io.read()

local sample_index = {}

for i = 1, num_class do
  sample_index[i] = torch.range(1, y:size(1))[y[{{}, i}]:byte()]
end

for i = 1, num_class do
  local shuffled_mat_ind = (torch.rand(batch_size)*sample_index[i]:size(1)):ceil():long()
  print(shuffled_mat_ind)
  local shuffled_mat = x:index(1, shuffled_mat_ind)
  print(shuffled_mat)
end

io.read()

for c = 1, num_class do
  print(sample_index[c])
  local shuffle_mask = torch.zeros(batch_size, num_class):scatter(2, torch.zeros(batch_size, 1):fill(c):long(), (1-y[{{}, c}]):view(-1, 1))
  print(shuffle_mask)
  io.read()
end
