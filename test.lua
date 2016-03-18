require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'

-- there is a better one called llap
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate()
loader = checkpoint.loader
print(loader)

local current_state
current_state = {}
  for L = 1,checkpoint.opt.num_layers do
       -- c and h for all layers
       local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
       table.insert(current_state, h_init:clone())
       if checkpoint.opt.model == 'lstm' then
            table.insert(current_state, h_init:clone())
       end
   end
state_size = #current_state

function next_batch(split_index)
     loader.batch_ix[split_index] = loader.batch_ix[split_index] + 1
     if loader.batch_ix[split_index] > loader.split_sizes[split_index] then
          loader.batch_ix[split_index] = 1 -- cycle around to beginning
     end
     local ix = loader.batch_ix[split_index]
     if split_index == 2 then ix = ix + loader.ntrain end -- offset by train set size
     if split_index == 3 then ix = ix + loader.ntrain + loader.nval end -- offset by train + val
     return loader.x_batches[ix], loader.y_batches[ix]
end

--x, y = next_batch(3)
x = torch.Tensor{16, 13, 17, 8, 3}

local rnn_state = {[0] = current_state}

for t = 1, x:size(1) do
    local lst = protos.rnn:forward{torch.Tensor{x[t]}, unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i = 1, #current_state do table.insert(rnn_state[t], lst[i]) end
    prediction = lst[#lst]
    -- print(x[t])
    print(prediction)
end
