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
    local kW = self.conv_size[t]
    local dW = self.stride[t]
    self.subsampling[t] = nn.TemporalMaxPooling(kW, dW)
    --self.subsampling[t] = nn.TemporalConvolution(self.rnn_size, self.rnn_size, self.conv_size, self.stride)
    --self.subsampling[t] = nn.TemporalAttention()
  end
  -----------------------------------
  
  self.core = {}
  for t = 1, self.num_layers do
      local withDecoder = false
      local input_size = self.rnn_size
      if t == 1 then input_size = self.input_size end
      if t == self.num_layers then withDecoder = true end
      if self.is1vsA then 
          self.core[t] = LSTM_mc.lstm(input_size, self.output_size, self.rnn_size, 1, dropout, self.group, withDecoder)
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
          self.core[t] = LSTM.lstm(input_size, self.output_size, self.rnn_size, 1, dropout, withDecoder)
          for layer_idx = 1, opt.num_layers do
            for _,node in ipairs(self.core[t].forwardnodes) do
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
  for l = self.num_layers, 1, -1 do
      if l ~= self.num_layers then self.unroll_len[l] = self.stride[l+1]*(self.unroll_len[l+1]-1)+self.conv_size[l+1]
      else self.unroll_len[self.num_layers] = self.seq_length end
      self.clones[l] = {self.core[l]}
      self.subsamplingClones[l] = {self.subsampling[l]}
      for t=2, self.unroll_len[l] do
          self.clones[l][t] = self.core[l]:clone('weight', 'bias', 'gradWeight', 'gradBias')
          self.subsamplingClones[l][t] = self.subsampling[l]:clone('weight', 'bias', 'gradWeight', 'gradBias')
      end
  end
  self.unroll_len[0] = self.stride[1]*(self.unroll_len[1]-1)+self.conv_size[1]
  print("Unroll len for each layer")
  for t = 0, self.num_layers do
      print(self.unroll_len[t])
  end
  print("End")
end

function layer:getModulesList()
  return {self.core, self.subsampling}
end

function layer:getInputSeqLength()
  return self.unroll_len[0]
end

function layer:parameters()
  local params = {}
  local grad_params = {}
  
  for l = 1, self.num_layers do
      local p1,g1 = self.core[l]:parameters()
      for k,v in pairs(p1) do table.insert(params, v) end
      for k,v in pairs(g1) do table.insert(grad_params, v) end

      p1, g1 = self.subsampling[l]:parameters()
      if p1 then
          for k,v in pairs(p1) do table.insert(params, v) end
          for k,v in pairs(g1) do table.insert(grad_params, v) end
      end
  end

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do 
      if torch.type(v) == 'table' then for kk, vv in pairs(v) do vv:training() end
      else v:training() end
  end
  for k,v in pairs(self.subsamplingClones) do 
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
  for k,v in pairs(self.subsamplingClones) do 
      if torch.type(v) == 'table' then for kk, vv in pairs(v) do vv:evaluate() end
      else v:evaluate() end
  end

end

function layer:updateOutput(input)
  local seq = input -- layer:getInputSeqLength() * batch_size * input_size
  if self.clones == nil then self:createClones() end
    -- lazily create clones on first forward pass

  --assert(seq:size(1) == self.seq_length)
  assert(seq:size(3) == self.input_size)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length, batch_size, self.output_size)
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.LSTM_input = {}  -- after subsampling before LSTM
  self.interm_val= {[0] = seq}  -- after LSTM
  for l = 1, self.num_layers do
      self.inputs[l] = {}
      self.LSTM_input[l] = torch.zeros(self.unroll_len[l], batch_size, self.rnn_size):cuda()

      if l ~= self.num_layers then self.interm_val[l] = torch.zeros(self.unroll_len[l], batch_size, self.rnn_size):cuda() end
      
      -- Subsampling
      -- WARNING: this is using subsampling instead of subsamplingClones[]
      -- since for both maxpooling and temporalConvolution, the input must be batch_size x nInputframe x frame_size
      self.LSTM_input[l] = self.subsampling[l]:forward(self.interm_val[l-1]:transpose(1, 2)):transpose(1, 2)

      assert(self.LSTM_input[l]:size(1) == self.unroll_len[l], "The subsampling res is not consistent with pre-computed result")

      -- LSTM
      for t=1, self.unroll_len[l] do
          -- inputs are input, c, h; Only input one h and c since each layer only has one layer
          self.inputs[l][t] = {self.LSTM_input[l][t], self.state[t-1][(l-1)*2+1], self.state[t-1][l*2]}
          -- forward the network
          print(l)
          print(t)
          print(self.inputs[l][t])
          io.read()
          local out = self.clones[l][t]:forward(self.inputs[l][t])
          -- process the outputs
          -- print(self.interm_val[l][t]:size())
          print(out)
          self.interm_val[l][t]:copy(out[#out])
          -- io.read()
          --if l ~= self.num_layers then self.interm_val[l][t] = out[#out]  -- which is h
          --else self.output[t] = out[self.num_state+1] end
          if self.state[t] == nil then self.state[t] = {} end
          -- each time only insert one c and one h
          for i=1,2 do table.insert(self.state[t], out[i]) end
      end

  end 
  return self.output
end

function layer:sample(input)
  local seq = input --input_seq_length * batch_size * input_size
  if self.clones == nil then self:createClones() end
    -- lazily create clones on first forward pass

  --assert(seq:size(1) == self.seq_length)
  assert(seq:size(3) == self.input_size)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length, batch_size, self.output_size)
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.LSTM_input = {}  -- after subsampling before LSTM
  self.interm_val= {[0] = seq}  -- after LSTM
  for l = 1, self.num_layers do
      self.inputs[l] = {}

      -- Subsampling
      -- since for both maxpooling and temporalConvolution, the input must be batch_size x nInputframe x frame_size
      self.LSTM_input[l] = self.subsampling[l]:forward(self.interm_val[l-1]:transpose(1, 2)):transpose(1, 2)

      -- LSTM
      local unroll_len = self.LSTM_input[l]:size(1)
      for t=1, unroll_len do
          -- inputs are input, c, h; Only input one h and c since each layer only has one layer
          self.inputs[l][t] = {self.LSTM_input[t], self.state[t-1][(l-1)*2+1], self.state[t-1][l*2]}
          -- forward the network
          local out = self.clones[l][t]:forward(self.inputs[l][t])
          -- process the outputs
          if l ~= self.num_layers then self.interm_val[l][t] = out[self.num_state]  -- which is h
          else self.output[t] = out[self.num_state+1] end
          if self.state[t] == nil then self.state[t] = {} end
          -- each time only insert one c and one h
          for i=1,2 do table.insert(self.state[t], out[i]) end
      end

  end 
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local dinputs = input:clone():zero() -- grad on input images
  local dstate = {[self.seq_length] = self.init_state}
  assert(self.unroll_len[self.num_layers] == self.seq_length, "unroll length for last layer should be equal to seq_length")

  local dInterm_val = {[self.num_layers+1] = gradOutput}
  for l = self.num_layers, 1, -1 do

    -- LSTM
    local dLSTM_input_l = torch.zeros(self.LSTM_input[l]:size())
    for t = self.unroll_len[l], 1, -1 do
      local dout = {} 
      for k = 2*l-1, 2*l do table.insert(dout, dstate[t][k]) end
      table.insert(dout, dInterm_val[l+1][t])
      local dinputs_t = self.clones[l][t]:backward(self.inputs[l][t], dout)
      dLSTM_input_l[t] = dinputs_t[1]
      dstate[t-1] = {}
      for k = 2, 3 do table.insert(dstate[t-1], dinputs_t[k]) end
    end

    -- Subsampling
    dInterm_val[l] = self.subsamplingClones[l]:backward(self.interm_val[l]:transpose(1,2), dLSTM_input_l:transpose(1,2)):transpose(1,2)
    assert(dInterm_val[l]:size(1) == self.unroll_len[l-1], "First interm_val dim should be the same as unroll_len[l-1]")
  end

  self.gradInput = dInterm_val[1]
  return self.gradInput
end
