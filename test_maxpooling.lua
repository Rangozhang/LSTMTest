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
cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-gpuid',1,'which gpu to use. -1 = use CPU')
cmd:option('-data_dir','data/test_')
cmd:option('-batch_size',128)
cmd:option('-seq_length', 3)
cmd:option('-n_class', 10)
cmd:option('-nbatches', 500)
cmd:option('-OverlappingData', true)
cmd:option('-draw', false)
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
loader = DataLoader.create(opt.data_dir, opt.batch_size, opt.seq_length^2, split_sizes, opt.n_class, opt.nbatches, opt.OverlappingData)
n_data = loader.test_n_data
vocab_mapping = loader.vocab_mapping
vocab_size = loader.vocab_size
vocab = {}
for k, v in pairs(vocab_mapping) do
    vocab[v] = k
end

num_level = 2

correct = 0.0
total = 0.0
local accuracy_for_each_class = torch.Tensor(opt.n_class):fill(0)
local n_data_for_each_class = accuracy_for_each_class:clone()
local accuracy_2 = 0.0 --accuracy_for_each_class:clone()
local accuracy_1 = 0.0 --accuracy_for_each_class:clone()
local accuracy_1_ = 0.0
local first_two = 0.0

protos.rnn1:evaluate()
protos.rnn2:evaluate()

for i = 1, n_data do
    xlua.progress(i, n_data)
    local x, y = loader:next_test_data()
    
    if opt.gpuid >= 0 then
        x = x:float():cuda()
    end
    
    draw1 = torch.Tensor(x:size(1)):fill(0)
    draw2 = torch.Tensor(x:size(1)):fill(0)

    local rnn_state = {}
    local level_output = {}
    for l = 1, num_level do
        rnn_state[l] = {[0] = clone_list(current_state)}
        level_output[l] = {}
    end
    
    local temporalPooling = nn.TemporalMaxPooling(3):cuda()
    local interm_size = 32
    local final_pred = torch.zeros(opt.n_class):cuda()
    local interm_val = torch.zeros(3, interm_size):cuda()
    for t = 1, x:size(1) do
        local x_OneHot = OneHot(vocab_size):forward(torch.Tensor{x[t]}):cuda()
        local lst = protos.rnn1:forward{x_OneHot, unpack(rnn_state[1][t-1])}
        rnn_state[1][t] = {}
        for i = 1, #current_state do table.insert(rnn_state[1][t], lst[i]) end
        level_output[1][t] = lst[#lst]
        interm_val[(t-1)%opt.seq_length+1]:copy(level_output[1][t])
        --interm_val[{{},{((t-1)%opt.seq_length)*interm_size+1, ((t-1)%opt.seq_length+1)*interm_size}}]:add(level_output[1][t])
        if t%opt.seq_length == 0 or t == x:size(1) then
            local out_pooling = temporalPooling:forward(interm_val)
            local denominator = (t-1)%opt.seq_length+1
            --interm_val:div(denominator)
            local t2_ind = math.floor((t-1)/opt.seq_length)+1
            local lst = protos.rnn2:forward{out_pooling, unpack(rnn_state[2][t2_ind-1])}
            rnn_state[2][t2_ind] = {}
            for i = 1, #current_state do table.insert(rnn_state[2][t2_ind], lst[i]) end
            local prediction = lst[#lst]
            for tt = 0, denominator-1 do
                draw1[t-tt] = prediction[{1, y[1]}]
            end
            if opt.OverlappingData then
                for tt = 0, denominator-1 do
                    draw2[t-tt] = prediction[{1, y[2]}]
                end
            end
            for tt = 0, denominator-1 do
                draw1[t-tt] = prediction[{1, y[1]}]
            end
            for tt = denominator-1,0,-1 do
                tmp_str = vocab[x[t-tt]] .. "\t"
                for m = 1, prediction:size(2) do
                    tmp_str = tmp_str .. '  ' .. string.format("%.3f", prediction[{1, m}])
                end
                --print(tmp_str)
            end
            
            -- Take average
            final_pred = final_pred + prediction
            interm_val:zero()
            --[[
            -- Take Maximum
            for w = 1, opt.n_class do
                final_pred[w] = math.max(final_pred[w], prediction[{1, w}])
            end
            --]]
        end
    end
    if opt.draw then
        x_axis = torch.range(1, x:size(1))
        if not opt.OverlappingData then
            gnuplot.pngfigure('./image_pureData_tc/instance' .. tostring(i) .. '.png')
            gnuplot.plot({'class '..tostring(y[1]), x_axis, draw1, '-'})
        else
            gnuplot.pngfigure('./image_tc/instance' .. tostring(i) .. '.png')
            gnuplot.plot({'class '..tostring(y[1]), x_axis, draw1, '-'}, {'class '..tostring(y[2]), x_axis, draw2, '-'})
        end
        x_str = 'set xtics ("'
        for mm = 1, x:size(1)-1 do
            x_str = x_str .. tostring(vocab[x[mm]]) .. '" ' .. tostring(mm) .. ', "'
        end
        x_str = x_str .. tostring(vocab[x[x:size(1)]]) .. '" ' .. tostring(x:size(1)) .. ') '
        gnuplot.raw(x_str)
        gnuplot.axis{'','',0,1}
        gnuplot.plotflush()
    end
    final_pred = final_pred/math.ceil(x:size(1)/opt.seq_length)
    --print(final_pred)
    --io.read()
    tmp_str = "Total:\t"
    for m = 1, final_pred:size(1) do
        tmp_str = tmp_str .. "  " .. string.format("%.3f", final_pred[{m}])
    end
    --print(tmp_str)
    --io.read()
    --print(final_pred:sum())
    --io.read()
    --print(res_y)
    total = total + 1
    k_ = 0
    increasing_ind = torch.Tensor(opt.n_class):apply(function(increasing_ind)
        k_ = k_ + 1
        return k_
    end)
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
        res_y1 = res_rank[-1]
        res_y2 = res_rank[-2]
        if res_y1 == y[1] or res_y1 == y[2] and res_y2 == y[1] or res_y2 == y[2] then
            first_two = first_two + 1
        end
        res_y = increasing_ind:maskedSelect(final_pred:gt(0.5):byte())
        res1 = (res_y:eq(y[1]):sum() >= 1)
        res2 = (res_y:eq(y[2]):sum() >= 1)
        --print(res1)
        --print(res2)
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
    print("Accuracy as first highest two are correct")
    print(first_two / total)
end
