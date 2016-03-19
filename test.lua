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
cmd:option('-data_dir','data/test')
cmd:option('-batch_size',128)
cmd:option('-seq_length', 3)
cmd:option('-n_class', 10)
cmd:option('-nbatches', 500)
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

local split_sizes = {0.90,0.05,0.05}
loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.n_class, opt.nbatches)
n_data = loader.test_n_data

correct = 0.0
total = 0.0

for i = 1, n_data do
    x, y = loader:next_test_data()
    print(x)

    local rnn_state = {[0] = current_state}
    local final_pred = torch.Tensor(opt.n_class):fill(0)
    for t = 1, x:size(1) do
        local lst = protos.rnn:forward{torch.Tensor{x[t]}, unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #current_state do table.insert(rnn_state[t], lst[i]) end
        prediction = lst[#lst]
        print(prediction)
        final_pred = final_pred + prediction
    end
    final_pred = final_pred/x:size(1)
    _, res_rank = torch.sort(final_pred)
    res_y = res_rank[#res_rank]
    print(final_pred)
    print(y)
    print(res_y)
    total = total + 1
    if y == res_y then
        correct = correct + 1
    end
end

print("Accuracy:")
print(correct/total)
