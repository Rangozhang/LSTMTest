local LSTM = require 'model.LSTM'
require 'model.MapTable'

local LSTM_1vsA = {}
function LSTM_1vsA.lstm(input_size, output_size, rnn_size, n, group, dropout, withDecoder)
  local dropout = dropout or 0 
  local withDecoder = withDecoder or false

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local outputs_tbl = {}
  local inputs_tbl = {}
  local group_inputs = {}
  for i = 1, group do
    group_inputs[i] = {inputs[1]}
  end
  for L = 2, 2*n+1 do
    local splitted_tbl = nn.SplitTable(2)(
                            nn.Reshape(group, rnn_size)(
                            inputs[L]))
    for i = 1, group do
      local group_input = nn.SelectTable(i)(splitted_tbl)
      table.insert(group_inputs[i], group_input)
    end
  end
  for i = 1, group do
    inputs_tbl[i] = nn.JoinTable(2)(group_inputs[i])
  end

  local mlp = nn.Sequential()
  local narrowed_tbl = nn.ConcatTable():add(nn.Narrow(2, 1, input_size))
  for i = 1, 2*n do
    narrowed_tbl:add(nn.Narrow(2, input_size+1+(i-1)*rnn_size, rnn_size))
  end
  mlp:add(nn.MapTable():add(narrowed_tbl))
  local parallel = nn.ParallelTable()
  for i = 1, group do
    parallel:add(LSTM.lstm(input_size, 1, rnn_size, n, dropout, withDecoder))
  end
  mlp:add(parallel)
  local para_out = mlp(inputs_tbl)

  for i = 1, group do
    -- local output = LSTM.lstm(input_size, 1, rnn_size, n,
    --   dropout, withDecoder)(group_inputs)
    local output = nn.SelectTable(i)(para_out)
    for t = 1, 2*n+1 do
      if outputs_tbl[t] == nil then outputs_tbl[t] = {} end
      local selected = nn.SelectTable(t)(output)
      table.insert(outputs_tbl[t], selected)
    end
  end

  local outputs = {}
  for t = 1, 2*n+1 do
    local joined = nn.JoinTable(2)(outputs_tbl[t])
    table.insert(outputs, joined)
  end

  return nn.gModule(inputs, outputs)
end

return LSTM_1vsA
