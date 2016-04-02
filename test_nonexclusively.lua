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
local DataLoader = require 'util.DataLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'

-- there is a better one called llap
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:argument('-dir', 'directory of the models')
cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-data_dir','data/test')
cmd:option('-batch_size',128)
cmd:option('-seq_length', 3)
cmd:option('-n_class', 10)
cmd:option('-nbatches', 500)
cmd:option('-OverlappingData', false)
cmd:option('-draw', true)
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

protos_list = {}
for i = 1, opt.n_class do
    checkpoint = torch.load(opt.dir..'/'..tostring(i)..'_'..opt.model)
    table.insert(protos_list, checkpoint.protos)
end

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
    for i = 1, #protos_list do
        local protos = protos_list[i]
        for k,v in pairs(protos) do v:cuda() end
    end
end


local split_sizes = {0.90,0.05,0.05}
loader = DataLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.n_class, opt.nbatches, opt.OverlappingData)
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
local accuracy_1_ = 0.0
local accuracy_tmp = 0.0
local accuracy_tmp2 = 0.0

for i = 1, #protos_list do
    protos_list[i].rnn:evaluate()
end

for i = 1, n_data do
    xlua.progress(i, n_data)
    local x, y = loader:next_test_data()
    
    --print("-----------Data----------")
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
    --print('------data------')
    --print(tmp_str)
    --print(y)

    if opt.gpuid >= 0 then
        x = x:float():cuda()
    end

    draw1 = torch.Tensor(x:size(1)):fill(0)
    draw2 = torch.Tensor(x:size(1)):fill(0)
    
    local pred = torch.Tensor(x:size(1), opt.n_class):fill(0):cuda()
    local final_pred = torch.Tensor(opt.n_class):fill(0):cuda()

    for protos_ind = 1, #protos_list do
        local protos = protos_list[protos_ind]
        local rnn_state = {[0] = clone_list(current_state)}
        for t = 1, x:size(1) do
            local lst = protos.rnn:forward{torch.Tensor{x[t]}:cuda(), unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i = 1, #current_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst]
            if protos_ind == y[1] then
                draw1[t] = prediction[1][1]
            else
                if opt.OverlappingData and protos_ind == y[2] then
                    draw2[t] = prediction[1][1]
                end
            end
            tmp_str = vocab[x[t]] .. "\t"
            for m = 1, prediction:size(2) do
                tmp_str = tmp_str .. '  ' .. string.format("%.3f", prediction[{1, m}])
            end
            --print(tmp_str)
            pred[t][protos_ind] = pred[t][protos_ind] + prediction[1][1]
            final_pred[protos_ind] = final_pred[protos_ind] + prediction[1][1]
        end
    end
    if opt.draw then
        x_axis = torch.range(1, x:size(1))
        if not opt.OverlappingData then
            gnuplot.pngfigure('./image_pureData_nonexclusively/instance' .. tostring(i) .. '.png')
            gnuplot.plot({'class '..tostring(y[1]), x_axis, draw1, '~'})
        else
            gnuplot.pngfigure('./image_nonexclusively/instance' .. tostring(i) .. '.png')
            gnuplot.plot({'class '..tostring(y[1]), x_axis, draw1, '~'}, {'class '..tostring(y[2]), x_axis, draw2, '~'})
        end
        x_str = 'set xtics ("'
        for mm = 1, x:size(1)-1 do
            x_str = x_str .. tostring(vocab[x[mm]]) .. '" ' .. tostring(mm) .. ', "'
        end
        x_str = x_str .. tostring(vocab[x[x:size(1)]]) .. '" ' .. tostring(x:size(1)) .. ') '
        gnuplot.raw(x_str)
        gnuplot.plotflush()    
    end
    final_pred = final_pred/x:size(1)
    for r = 1, x:size(1) do
        tmp_str = vocab[x[r]] .. '\t'
        for c = 1, final_pred:size(1) do
            tmp_str = tmp_str .. " " .. string.format("%.3f", pred[r][c])
        end
        print(tmp_str)
    end
    tmp_str = "Total\t"
    for m = 1, final_pred:size(1) do
        tmp_str = tmp_str .. " " .. string.format("%.3f", final_pred[{m}])
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
        _, res_rank = torch.sort(final_pred)
        res_y_tmp = res_rank[-1]
        if res_y_tmp == y[1] or res_y_tmp == y[2] then
            accuracy_tmp = accuracy_tmp + 1
            res_y_tmp2 = res_rank[-2]
            if res_y_tmp2 == y[1] or res_y_tmp2 == y[2] then
                accuracy_tmp2 = accuracy_tmp2 + 1
            end
        end
        
        res_y = torch.range(1, opt.n_class):maskedSelect(final_pred:gt(0.5):byte())
        res1 = (res_y:eq(y[1]):sum() >= 1)
        res2 = (res_y:eq(y[2]):sum() >= 1)
        if res1 and res2 then
            accuracy_1_ = accuracy_1_ + 1
            if #res_y == 2 then
                accuracy_2 = accuracy_2 + 1
                accuracy_1 = accuracy_1 + 1
            end
            else if res1 or res2 and #res_y == 2 then
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
    print("Accuracy of only one is correct or two are correct")
    print(accuracy_1 / total)
    print("Accracy as long as the result consists of the two classes")
    print(accuracy_1_ / total)
    print("Accuracy that one of the groundtruth has the highest prediction value")
    print(accuracy_tmp / total)
    print("Accuracy that two groundtruth have the highest prediction value")
    print(accuracy_tmp2 / total)
end
