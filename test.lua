require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'cudnn'
require 'cunn'
require 'xlua'

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
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-data_dir','data/test')
cmd:option('-batch_size',128)
cmd:option('-seq_length', 3)
cmd:option('-n_class', 10)
cmd:option('-nbatches', 500)
cmd:option('-OverlappingData', false)
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

checkpoint = torch.load(opt.model)
protos = checkpoint.protos

current_state = {}
for L = 1,checkpoint.opt.num_layers do
   -- c and h for all layers
   local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
   if opt.gpuid >= 0 then h_init = h_init:cuda() end
   table.insert(current_state, h_init:clone())
   if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
   end
end
 
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end


local split_sizes = {0.90,0.05,0.05}
loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.n_class, opt.nbatches)
n_data = loader.test_n_data

correct = 0.0
total = 0.0
local accuracy_for_each_class = torch.Tensor(opt.n_class):fill(0)
local n_data_for_each_class = accuracy_for_each_class:clone()

protos.rnn:evaluate()

for i = 1, n_data do
    xlua.progress(i, n_data)
    local x, y = loader:next_test_data()
    print("-----------Data----------")
    print(x)
    print("-----------Groundtruth----------")
    print(y)

    if opt.gpuid >= 0 then
        x = x:float():cuda()
    end

    local rnn_state = {[0] = current_state}
    local final_pred = torch.Tensor(opt.n_class):fill(0):cuda()
    for t = 1, x:size(1) do
        local lst = protos.rnn:forward{torch.Tensor{x[t]}:cuda(), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #current_state do table.insert(rnn_state[t], lst[i]) end
        prediction = lst[#lst]
        --print(prediction)
        final_pred = final_pred + prediction
    end
    final_pred = final_pred/x:size(1)
    res_val_rank, res_rank = torch.sort(final_pred)
    res_y = res_rank[#res_rank]
    --print(final_pred)
    --print(res_y)
    total = total + 1
    n_data_for_each_class[y] = n_data_for_each_class[y] + 1
    if y == res_y then
        correct = correct + 1
        accuracy_for_each_class[y] = accuracy_for_each_class[y] + 1
    end
    
end

accuracy_for_each_class = torch.cdiv(accuracy_for_each_class, n_data_for_each_class)

print("Accuracy for each class:")
print(accuracy_for_each_class)

print("Accuracy:")
print(correct/total)
