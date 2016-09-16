require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'cudnn'
require 'cunn'
require 'xlua'
require 'gnuplot'
require 'model.LSTM_layer'
require 'model.HLSTM_layer'

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
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-data_dir','data/test_')
cmd:option('-batch_size',128) -- not useful for now
cmd:option('-seq_length', 9)
cmd:option('-n_class', 10)
cmd:option('-nbatches', 500)
cmd:option('-overlap', false)
cmd:option('-draw', false)
cmd:option('-printing', false)
cmd:option('-hiber_gate', false)
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

checkpoint = torch.load(opt.model)
protos = checkpoint.protos

--local debugger = require('fb.debugger')
--debugger.enter()

if opt.gpuid >= 0 then
    for k,v in pairs(protos) do 
        -- print(torch.type(v))
        v:cuda() 
    end
end

local split_sizes = {0.90,0.05,0.05}
loader = DataLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.n_class, opt.nbatches, opt.overlap)
n_data = loader.test_n_data
vocab_mapping = loader.vocab_mapping
vocab_size = loader.vocab_size
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
local first_two = 0.0

local hiber_accuracy = 0.0
local hiber_total = 0

protos.rnn:evaluate()

for i = 1, n_data do
    xlua.progress(i, n_data)
    local x, y = loader:next_test_data()
    x = x:reshape(1, x:size(1))
    -- x: batch_size x seq_length

    local seq_length = x:size(2)
    local y_onehot = OneHot(opt.n_class):forward(y)
    local hiber_y
    -- hiber_y: seq_length x batch_size x output_size+1
    if opt.hiber_gate then
        hiber_y = torch.zeros(seq_length, y_onehot:size(1), y_onehot:size(2)+1)
        local invalid_x = x:le(26)
        for j = 1, seq_length do
            hiber_y[{{j},{},{1,y_onehot:size(2)}}] = y_onehot:clone()
            local indices = torch.range(1, y_onehot:size(1))[invalid_x[{{},{j}}]]
            if indices:nDimension() > 0 then
                for i = 1, indices:size()[1] do
                    hiber_y[{{j},{indices[i]},{}}]:fill(0)
                    hiber_y[{{j},{indices[i]},{-1}}] = 1
                end
            end
        end 
    end
    if opt.gpuid >= 0 then
        hiber_y = hiber_y:cuda()
    end

    local x_input = torch.zeros(seq_length, 1 , vocab_size):cuda()
    for t = 1, seq_length do
        x_input[t] = OneHot(vocab_size):forward(x[{{}, t}])
    end

    --print("-----------Data----------")
    --print(x)
    --print(y)
    --[[
    ina = {'c', 'y', 'w', 'd', 'r', 'r', 'x', 'n', 'f', 'i', 'j'}
    x = torch.Tensor(#ina)
    for h = 1, #ina do
        x[h] = vocab_mapping[ina[h]
    end
    
    tmp_str = ""
    for z = 1, x:size(1) do
        tmp_str = tmp_str .. " " .. vocab[x[z]
    end
    print('------data------')
    print(tmp_str)
    print(y)
    --]]
    if opt.gpuid >= 0 then
        x_input = x_input:float():cuda()
    end
    
    draw1 = torch.Tensor(seq_length):fill(0)
    draw2 = torch.Tensor(seq_length):fill(0)

    local final_pred = torch.Tensor(opt.n_class):fill(0):cuda()
    local predictions, hiber_predictions
    if opt.hiber_gate then 
        local rnn_res = protos.rnn:sample(x_input)
        predictions = rnn_res[1]
        hiber_predictions = rnn_res[2]
    else
        predictions = protos.rnn:sample(x_input)
    end
    print(predictions:size())
    print(hiber_predictions:size())
    io.read()
    for t = 1, seq_length do
        prediction = predictions[t]
        draw1[t] = prediction[{1, y[1]}]
        if opt.overlap then
            draw2[t] = prediction[{1, y[2]}]
        end
        tmp_str = vocab[x[t]] .. "\t"
        for m = 1, prediction:size(2) do
            tmp_str = tmp_str .. '  ' .. string.format("%.3f", prediction[{1, m}])
        end
        if opt.printing then print(tmp_str) end
        -- Take average
        final_pred = final_pred + prediction
        --[[
        -- Take Maximum
        for w = 1, opt.n_class do
            final_pred[w] = math.max(final_pred[w], prediction[{1, w}])
        end
        --]]
    end
    if opt.draw then
        x_axis = torch.range(1, x:size(1))
        if not opt.overlap then
            gnuplot.pngfigure('./image_pureData/instance' .. tostring(i) .. '.png')
            gnuplot.plot({'class '..tostring(y[1]), x_axis, draw1, '-'})
        else
            gnuplot.pngfigure('./image/instance' .. tostring(i) .. '.png')
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
    final_pred = final_pred/x:size(1)
    tmp_str = "Total:\t"
    for m = 1, final_pred:size(1) do
        tmp_str = tmp_str .. "  " .. string.format("%.3f", final_pred[{m}])
    end
    -- --[[
    --print(tmp_str)
    --io.read()
    if opt.printing then
        print(final_pred:sum())
        print(final_pred)
        print(y)
        io.read()
    end
    --]]
    total = total + 1
    k_ = 0
    increasing_ind = torch.Tensor(opt.n_class):apply(function(increasing_ind)
        k_ = k_ + 1
        return k_
    end)
    if not opt.overlap then
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
        io.read()
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

if not opt.overlap then
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
