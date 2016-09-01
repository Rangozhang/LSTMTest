require 'nn'
package.path = "../?.lua;" .. package.path
local LSTM = require 'model.LSTM'
local LSTM_1vsA = require 'model.LSTM_1vsA'

local layer, parent = torch.class('nn.LSTMLayer', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  self.is1vsA = opt.is1vsA or false

  self.input_size = opt.input_size
  self.output_size = opt.output_size
  self.rnn_size = opt.rnn_size
  self.num_layers = opt.num_layers
  local dropout = opt.dropout
  self.seq_length = opt.seq_length
  self.group = self.output_size -- assign each group only one output: 1vsAll
  if self.is1vsA then 
      -- rnn_size here means the size for each model
      assert(self.rnn_size % self.group == 0, "rnn_size and group are invalid")
      self.core = LSTM_1vsA.lstm(self.input_size, self.output_size, self.rnn_size/self.group, self.num_layers, self.group, dropout, true)
  else 
      self.core = LSTM.lstm(self.input_size, self.output_size, self.rnn_size, self.num_layers, dropout, true) 
  end
  for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(self.core.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then--group_idx .. '_' .. layer_idx then
             print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
             node.data.module.bias[{{self.rnn_size+1, 2*self.rnn_size}}]:fill(1.0)
        end
    end
  end
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  -- contruct the hiden state, has to be all zeros
  assert(batch_size ~= nil, 'batch size must be provided')
  if not self.init_state then self.init_state = {} end
  for h=1,self.num_layers*2 do
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero()
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones')
  self.clones = {self.core}
  for t=2,self.seq_length do -- t == 1 is self.core itself
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end

function layer:getModulesList()
  return {self.core}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1, g1 = self.core:parameters()

  local params = {}
  for k, v in pairs(p1) do table.insert(params, v) end
  
  local grad_params = {}
  for k, v in pairs(g1) do table.insert(grad_params, v) end

  return params, grad_params
end

function layer:training()
  -- create these lazily if needed
  if self.clones == nil then self:createClones() end
  for k,v in pairs(self.clones) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end
  for k,v in pairs(self.clones) do v:evaluate() end
end

function layer:updateOutput(input)
  local seq = input -- seq_length * batch_size * input_size
  if self.clones == nil then self:createClones() end
  -- lazily create clones on first forward pass

  assert(seq:size(1) == self.seq_length)
  assert(seq:size(3) == self.input_size)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length, batch_size, self.output_size)
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  for t=1, self.seq_length do
      self.inputs[t] = {seq[t],unpack(self.state[t-1])}
      -- forward the network
      local out = self.clones[t]:forward(self.inputs[t])
      -- process the outputs
      self.output[t] = out[self.num_state+1] -- last element is the output vector
      self.state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
  end

  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local dinputs = input:clone():zero() -- grad on input images

  local dstate = {[self.seq_length] = self.init_state}
  for t=self.seq_length,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    local dinputs_t = self.clones[t]:backward(self.inputs[t], dout)
    -- split the gradient to xt and to state
    dinputs[t] = dinputs_t[1] -- first element is the input vector
    dstate[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs_t[k]) end
  end

  self.gradInput = dinputs
  return self.gradInput
end

function layer:sample(input)
  local seq = input --input_seq_length * batch_size * input_size
  local batch_size = seq:size(2)
  local input_seq_length = seq:size(1)
  self.output:resize(input_seq_length, batch_size, self.output_size)
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  for t=1, input_seq_length do
    self.inputs[t] = {seq[t],unpack(self.state[t-1])}
    -- forward the network
    local out = self.clones[1]:forward(self.inputs[t])
    -- process the outputs
    self.output[t] = out[self.num_state+1] -- last element is the output vector
    self.state[t] = {} -- the rest is state
    for i=1,self.num_state do table.insert(self.state[t], out[i]) end
  end

  return self.output
end
