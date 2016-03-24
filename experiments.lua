require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'cudnn'
require 'cunn'
require 'xlua'
require 'gnuplot'

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
cmd:option('-OverlappingData', true)
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
loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.n_class, opt.nbatches, opt.OverlappingData)
n_data = loader.test_n_data
vocab_mapping = loader.vocab_mapping
vocab = {}
for k, v in pairs(vocab_mapping) do
    vocab[v] = k
end

correct = 0.0
total = 0.0
local accuracy_for_each_class = torch.Tensor(opt.n_class):fill(0)
local n_data_for_each_class = accuracy_for_each_class:clone()
local accuracy_2 = 0.0 --accuracy_for_each_class:clone()
local accuracy_1 = 0.0 --accuracy_for_each_class:clone()

protos.rnn:evaluate()

for i = 1, n_data do
    xlua.progress(i, n_data)
    local x, y = loader:next_test_data()
    
    print("-----------Data----------")
    --print(x)
    --[[
    ina = {'c', 'y', 'w', 'd', 'r', 'r', 'x', 'n', 'f', 'i', 'j'}
    x = torch.Tensor(#ina)
    for h = 1, #ina do
        x[h] = vocab_mapping[ina[h]
    end
    --]]
    tmp_str = ""
    for z = 1, x:size(1) do
        tmp_str = tmp_str .. " " .. vocab[x[z]]
    end
    print('------data------')
    print(tmp_str)
    print(y)

    if opt.gpuid >= 0 then
        x = x:float():cuda()
    end

    draw1 = torch.Tensor(x:size(1)):fill(0)
    draw2 = torch.Tensor(x:size(1)):fill(0)

    local rnn_state = {[0] = current_state}
    local final_pred = torch.Tensor(opt.n_class):fill(0):cuda()
    for t = 1, x:size(1) do
        local lst = protos.rnn:forward{torch.Tensor{x[t]}:cuda(), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #current_state do table.insert(rnn_state[t], lst[i]) end
        prediction = lst[#lst]
        draw1[t] = prediction[{1, y[1]}]
        draw2[t] = prediction[{1, y[2]}]
        tmp_str = vocab[x[t]] .. "\t"
        for m = 1, prediction:size(2) do
            tmp_str = tmp_str .. '  ' .. string.format("%.3f", prediction[{1, m}])
        end
        print(tmp_str)
        --print(tmp_str)
        final_pred = final_pred + prediction
    end
    x_axis = torch.range(1, x:size(1))
    gnuplot.pngfigure('./image/instance' .. tostring(i) .. '.png')
    gnuplot.plot({'class '..tostring(y[1]), x_axis, draw1, '~'}, {'class '..tostring(y[2]), x_axis, draw2, '~'})
    x_str = 'set xtics ("'
    for mm = 1, x:size(1)-1 do
        x_str = x_str .. tostring(vocab[x[mm]]) .. '" ' .. tostring(mm) .. ', "'
    end
    x_str = x_str .. tostring(vocab[x[x:size(1)]]) .. '" ' .. tostring(x:size(1)) .. ') '
    gnuplot.raw(x_str)
    gnuplot.title('London average temperature')
    gnuplot.plotflush()
    final_pred = final_pred/x:size(1)
    tmp_str = "Total:\t"
    for m = 1, final_pred:size(1) do
        tmp_str = tmp_str .. "  " .. string.format("%.3f", final_pred[{m}])
    end
    print(tmp_str)
    --io.read()
    --print(final_pred:sum())
    --io.read()
    --print(res_y)
    total = total + 1
    if not opt.OverlappingData then
        fail_list = {}
        fail_list_ind = 1
        y = y[1]
        _, res_rank = torch.sort(final_pred)
        res_y = res_rank[#res_rank]
        --[[
        print(x)
        print(y)
        print(final_pred)
        print(res_rank)
        --]]
        n_data_for_each_class[y] = n_data_for_each_class[y] + 1
        if y == res_y then
            correct = correct + 1
            accuracy_for_each_class[y] = accuracy_for_each_class[y] + 1
        else
            print(y .. ':' .. res_y)
        end
    else
        res_y = final_pred:gt(0.1)
        --print(res_y)
        res1 = (res_y:eq(y[1]):sum() >= 1)
        res2 = (res_y:eq(y[2]):sum() >= 1)
        --print(res1)
        --print(res2)
        if res1 and res2 and #res_y == 2 then
            accuracy_2 = accuracy_2 + 1
            accuracy_1 = accuracy_1 + 1
        else if res1 or res2 then
            accuracy_1 = accuracy_1 + 1
        end
        end
    end
end

if not opt.OverlappingData then
    accuracy_for_each_class = torch.cdiv(accuracy_for_each_class, n_data_for_each_class)

    print("Accuracy for each class:")
    print(accuracy_for_each_class)

    print("Accuracy:")
    print(correct/total)
else 
    print("Accuracy of exact correct:")
    print(accuracy_2 / total)
    print("Accuracy of only one is correct or two are correct but there are other false positive")
    print(accuracy_1 / total)
end