require 'nn'
package.path = "../?.lua;" .. package.path
local LSTM = require 'model.LSTM'
local LSTM_mc = require 'model.LSTM_mc'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LSTMHierarchicalLayer', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  self.input_size = opt.input_size
  self.output_size = opt.output_size
  self.rnn_size = opt.rnn_size
  self.num_layers = opt.num_layers
  local dropout = opt.dropout
  self.seq_length = opt.seq_length

  ------------ 1vsAll param ------------
  self.is1vsA = opt.is1vsA or false
  -- assign each group only one output: 1vsAll
  self.group = self.output_size      
  -------------------------------------
  
  -------- Hierarchical param --------
  self.conv_size = opt.conv_size
  self.stride = opt.stride
  self.unroll_len = {}
  ------------------------------------
  
  ---------- Temporal Conv -----------
  self.subsampling = {}
  for t = 1, self.num_layers do
    local kW = self.conv_size
    local dW = self.stride
    if t == 1 then 
        kW = 1
        dW = 1
    end
    self.subsampling[t] = nn.TemporalMaxPooling(kW, dW)
    --self.subsampling[t] = nn.TemporalConvolution(self.rnn_size, self.rnn_size, self.conv_size, self.stride)
    --self.subsampling[t] = nn.TemporalAttention()
  end
  -----------------------------------
  
  self.core = {}
  for t = 1, self.num_layers do
      local withDecoder = false
      if t == self.num_layers then withDecoder = true end
      if self.is1vsA then 
          self.core[t] = LSTM_mc.lstm(self.input_size, self.output_size, self.rnn_size, 1, dropout, self.group, withDecoder)
          for layer_idx = 1, opt.num_layers do
            for group_idx = 1, self.group do
                for _,node in ipairs(self.core.forwardnodes) do
                    if node.data.annotations.name == "i2h_" .. group_idx .. '_' .. layer_idx then
                         print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                         node.data.module.bias[{{self.rnn_size/self.group+1, 2*self.rnn_size/self.group}}]:fill(1.0)
                    end
                end
            end
          end
      else 
          self.core[t] = LSTM.lstm(self.input_size, self.output_size, self.rnn_size, 1, dropout, withDecoder)
          for layer_idx = 1, opt.num_layers do
            for _,node in ipairs(self.core.forwardnodes) do
                if node.data.annotations.name == "i2h_" .. layer_idx then--group_idx .. '_' .. layer_idx then
                     print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                     node.data.module.bias[{{self.rnn_size+1, 2*self.rnn_size}}]:fill(1.0)
                end
            end
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
  self.clones = {}
  self.subsamplingClones = {}
  self.unroll_len[self.num_layers] = self.seq_length 
  for l = self.num_layers, 1, -1 do
      if l ~= self.num_layers then 
          self.unroll_len[l] = self.stride*(self.unroll_len[l]-1)+self.conv_size 
      end
      self.clones[l] = {self.core[l]}
      self.subsamplingClones[l] = {self.subsampling[l]}
      for t=2, self.unroll_len[l] do
          self.clones[l][t] = self.core[l]:clone('weight', 'bias', 'gradWeight', 'gradBias')
          self.subsamplingClones[l][t] = self.subsampling[l]:clone('weight', 'bias', 'gradWeight', 'gradBias')
      end
  end
end

function layer:getModulesList()
  return {self.core, self.subsampling}
end

function layer:parameters()
  local params = {}
  local grad_params = {}
  
  for l = 1, self.num_layers do
      local p1,g1 = self.core[l]:parameters()
      for k,v in pairs(p1) do table.insert(params, v) end
      for k,v in pairs(g1) do table.insert(grad_params, v) end

      p1, g1 = self.subsampling[l]:parameters()
      for k,v in pairs(p1) do table.insert(params, v) end
      for k,v in pairs(g1) do table.insert(grad_params, v) end
  end

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do 
      if torch.type(v) == 'table' then for kk, vv in pairs(v) do vv:training() end
      else v:training() end
  end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do 
      if torch.type(v) == 'table' then for kk, vv in pairs(v) do vv:evaluate() end
      else v:evaluate() end
  end
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
  self.interm_pm = {}  -- before merge
  self.interm_am = {}  -- after merge
  for l = 1, self.num_layers do
      self.inputs[l] = {}
      if l ~= self.num_layers then
        self.interm_pm[l] = torch.zeros(self.unroll_len[l], batch_size, self.rnn_size)
        self.interm_am[l] = torch.zeros(self.unroll_len[l+1], batch_size, self.rnn_size)
      end

      -- output_gen
      for t=1, self.unroll_len[l] do
          -- inputs are input, c, h
          self.inputs[l][t] = {seq[t], self.state[t-1][(l-1)*2+1], self.state[t-1][l*2]}
          -- forward the network
          local out = self.clones[t]:forward(self.inputs[t])
          -- process the outputs
          if l ~= self.num_layers then self.interm_pm[l][t] = out[self.num_state]  -- which is h
          else self.output[t] = out[self.num_state+1] end
          if self.state[t] == nil then self.state[t] = {} end
          -- each time only insert one c and one h
          for i=1,2 do table.insert(self.state[t], out[i]) end
      end

      -- temporal conv
      if l ~= self.num_layers then
          for t=1, self.unroll_len[l+1] do
              self.interm_am[l] = self.subsampling[l]:forward(self.interm_pm[l]:transpose(1, 2)):transpose(1, 2)
          end
          --seq = self.interm_
      end
  end 
  return self.output
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
