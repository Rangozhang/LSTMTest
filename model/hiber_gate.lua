function hiber_gate(concated_rnn_size, input_size, embeded_size, output_size)

  -- h: batch_size x rnn_size*2*num_layers
  local h = nn.Identity()()
  -- input: batch_size x input_size
  local input = nn.Identity()()

  local embeded_h = nn.Tanh()(nn.Linear(concated_rnn_size, embeded_size)(h))
  local embeded_input = nn.Tanh()(nn.Linear(input_size, embeded_size)(input))

  local elementwise_product = nn.CAddTable()({embeded_h, embeded_input})
  local pre_output = nn.Linear(embeded_size,output_size)(elementwise_product)
  local pre_output2 = nn.Linear(output_size,output_size)(pre_output)
  local output = nn.LogSoftMax()(pre_output2)
  return nn.gModule({h, input}, {output})
end
