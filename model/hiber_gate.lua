require 'util.NoBP'

function hiber_gate(concated_rnn_size, input_size, embeded_size, output_size, group)
  -- h: batch_size x rnn_size
  local h = nn.Identity()()
  -- input: batch_size x input_size
  local input = nn.Identity()()
  local h_size_each = concated_rnn_size / group

  local pre_output_tbl = {}
  local submodule = hiber_gate_each(concated_rnn_size/group, input_size, embeded_size/group, 1) -- embeded_size/group)

  for i = 1, group do
    local h_each = nn.Narrow(2, (i-1)*h_size_each+1, h_size_each)(h)
    local elementwise_product = hiber_gate_each(concated_rnn_size/group, input_size, embeded_size/group, 1){h_each, input}
    -- local elementwise_product = submodule:clone('weight', 'bias', 'gradWeight', 'gradBias')({h_each, input})
    table.insert(pre_output_tbl, elementwise_product)
  end

  local pre_output = nn.JoinTable(2)(pre_output_tbl)

  -- add one more dimension for the noise
  local noise_hidden = nn.Linear(input_size, embeded_size/group)(input)
  local noise_output = nn.Linear(embeded_size/group, 1)(noise_hidden)
  local raw_output = nn.JoinTable(2){pre_output, noise_output}
  -- local hidden = nn.Linear(output_size-1, output_size)(pre_output)
  -- local raw_output = nn.Linear(output_size, output_size)(hidden)

  local output_norm = nn.BatchNormalization(output_size)(raw_output)
  local output = nn.Sigmoid()(output_norm)
  return nn.gModule({h, input}, {output})
end

function hiber_gate_each(h_size, input_size, embeded_size, output_size)
  local input = nn.Identity()()
  local h = nn.Identity()()

  local embeded_h = nn.ReLU(true)(nn.Linear(embeded_size, embeded_size)(nn.Linear(h_size, embeded_size)(h)))
  local embeded_input = nn.ReLU(true)(nn.Linear(embeded_size, embeded_size)(nn.Linear(input_size, embeded_size)(input)))

  local elementwise_product = nn.CMulTable()({embeded_h, embeded_input})
  local hidden_layer = nn.Linear(embeded_size, embeded_size)(elementwise_product)
  local output = nn.Linear(embeded_size, output_size)(hidden_layer)
  return nn.gModule({h, input}, {output})

end

function hiber_gate2(concated_rnn_size, input_size, embeded_size, output_size, group)

  -- h: batch_size x rnn_size
  local h = nn.Identity()()
  -- input: batch_size x input_size
  local input = nn.Identity()()
  local h_size_each = concated_rnn_size / group

  local pre_output_tbl = {}
  -- local submodule = hiber_gate_each(concated_rnn_size/group, input_size, embeded_size/group, 1) -- embeded_size/group)

  for i = 1, group do
    local h_each = nn.Narrow(2, (i-1)*h_size_each+1, h_size_each)(h)
    local elementwise_product = hiber_gate_each(concated_rnn_size/group, input_size, embeded_size/group, 1){h_each, input}
    -- local elementwise_product = submodule:clone('weight', 'bias', 'gradWeight', 'gradBias')({h_each, input})
    table.insert(pre_output_tbl, elementwise_product)
  end

  local pre_output = nn.JoinTable(2)(pre_output_tbl)

  -- add one more dimension for the noise
  local noise_hidden = nn.NoBP()(nn.Linear(output_size-1, embeded_size/group)(pre_output))
  local noise_output = nn.Linear(embeded_size/group, 1)(noise_hidden)
  local raw_output = nn.JoinTable(2){pre_output, noise_output}
  -- local hidden = nn.Linear(output_size-1, output_size)(pre_output)
  -- local raw_output = nn.Linear(output_size, output_size)(hidden)

  local output_norm = nn.BatchNormalization(output_size)(raw_output)
  local output = nn.Sigmoid()(output_norm)
  return nn.gModule({h, input}, {output})
end

function hiber_gate_each(h_size, input_size, embeded_size, output_size)
  local input = nn.Identity()()
  local h = nn.Identity()()

  local embeded_h = nn.ReLU(true)(nn.Linear(embeded_size, embeded_size)(nn.Linear(h_size, embeded_size)(h)))
  local embeded_input = nn.ReLU(true)(nn.Linear(embeded_size, embeded_size)(nn.Linear(input_size, embeded_size)(input)))

  local elementwise_product = nn.CMulTable()({embeded_h, embeded_input})
  local hidden_layer = nn.Linear(embeded_size, embeded_size)(elementwise_product)
  local output = nn.Linear(embeded_size, output_size)(hidden_layer)
  return nn.gModule({h, input}, {output})
end

--[[
function linear_classifier(input_size, output_size)
  -- input: batch_size x input_size
  local input = nn.Identity()()

  local pre_output = nn.Linear(input_size, output_size)(input)
  local output = nn.LogSoftMax()(pre_output)

  return nn.gModule({input}, {output})
end
--]]
