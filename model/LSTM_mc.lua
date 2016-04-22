
local LSTM_mc = {}
function LSTM_mc.lstm(input_size, output_size, rnn_size_all, n, dropout, group,  withDecoder)
  dropout = dropout or 0 
  withDecoder = withDecoder or false
  rnn_size = rnn_size_all / group

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    
    local prev_h_tbl = nn.SplitTable(2)(nn.Reshape(group, rnn_size)(prev_h))
    local prev_c_tbl = nn.SplitTable(2)(nn.Reshape(group, rnn_size)(prev_c))
    local next_c = {}
    local next_h = {}

    for t = 1, group do
        -- the input to this layer
        if L == 1 then 
          x = inputs[1]
          input_size_L = input_size
        else 
          x = outputs[(L-1)*2] 
          if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
          input_size_L = rnn_size
        end
        -- evaluate the input sums at once for efficiency
        local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
        local h2h = nn.Linear(rnn_size, 4 * rnn_size)(nn.SelectTable(t)(prev_h_tbl)):annotate{name='h2h_'..L}
        local all_input_sums = nn.CAddTable()({i2h, h2h})

        local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
        -- decode the gates
        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local out_gate = nn.Sigmoid()(n3)
        -- decode the write inputs
        local in_transform = nn.Tanh()(n4)
        -- perform the LSTM update
        next_c[t]          = nn.CAddTable()({
            nn.CMulTable()({forget_gate, nn.SelectTable(t)(prev_c_tbl)}),
            nn.CMulTable()({in_gate,     in_transform})
          })
        -- gated cells form the output
        next_h[t] = nn.CMulTable()({out_gate, nn.Tanh()(next_c[t])})
    end

    local next_c_concat = nn.JoinTable(2)(next_c)
    local next_h_concat = nn.JoinTable(2)(next_h)

    table.insert(outputs, next_c_concat)
    table.insert(outputs, next_h_concat)
  end

  if withDecoder then
      -- set up the decoder
      local top_h = outputs[#outputs]
      if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
      local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
      --local logsoft = nn.LogSoftMax()(proj)
      local sig = nn.Sigmoid()(proj)
      table.insert(outputs, sig)
  end

  return nn.gModule(inputs, outputs)
end

return LSTM_mc

