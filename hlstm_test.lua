require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
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
cmd:option('-batch_size',128)
cmd:option('-seq_length', 9)
cmd:option('-n_class', 2)
cmd:option('-nbatches', 500)
cmd:option('-overlap', false)
cmd:option('-printing', false)
cmd:option('-together', false)
cmd:text()

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpuid+1)

local checkpoint = torch.load(opt.model)
local protos = checkpoint.protos

local confusion = optim.ConfusionMatrix(opt.n_class)
local hiber_confusion = optim.ConfusionMatrix(opt.n_class+1)

if opt.gpuid >= 0 then
    for k,v in pairs(protos) do 
        v:cuda() 
    end
end

local split_sizes = {0.90,0.05,0.05}
local loader = DataLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.n_class, opt.nbatches, opt.overlap)
local n_data = loader.test_n_data
-- local n_data = 100
local vocab_mapping = loader.vocab_mapping
local vocab_size = loader.vocab_size
local vocab = {}
for k, v in pairs(vocab_mapping) do
    vocab[v] = k
end

local accuracy = 0.0
local total = 0.0
local accuracy_for_each_class = torch.Tensor(opt.n_class):fill(0)
local n_data_for_each_class = accuracy_for_each_class:clone()
local first_two = 0.0
local hiber_accuracy = 0.0
local hiber_total = 0.0

protos.rnn:evaluate()

for i = 1, n_data do
    xlua.progress(i, n_data)
    local x, y = loader:next_test_data()
    -- x: batch_size x seq_length
    x = x:reshape(1, x:size(1))

    local seq_length = x:size(2)
    local y_onehot = OneHot(opt.n_class):forward(y)

    -- hiber_y: seq_length x batch_size x output_size+1
    local hiber_y
    if opt.overlap then
        y_onehot = y_onehot[{{1},{}}] + y_onehot[{{2},{}}]
    end
    hiber_y = torch.zeros(seq_length, y_onehot:size(1), y_onehot:size(2)+1)
    local invalid_x = x:le(26) -- batch_size x input_size
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
    if opt.gpuid >= 0 then
        hiber_y = hiber_y:cuda()
    end

    -- x_input: seq_length x batch_size x input_size
    local x_input = torch.zeros(seq_length, 1 , vocab_size):cuda()
    for t = 1, seq_length do
        x_input[t] = OneHot(vocab_size):forward(x[{{}, t}])
    end
    if opt.gpuid >= 0 then
        x_input = x_input:float():cuda()
    end
    
    local final_pred = torch.CudaTensor(opt.n_class):fill(0)
    local predictions, hiber_predictions, rnn_res
    if not opt.together then rnn_res = protos.rnn:sample({x_input, hiber_y})
    else rnn_res = protos.rnn:sample({x_input}) end
    predictions = rnn_res[1]
    hiber_predictions = rnn_res[2]
    hidden_state = rnn_res[3]

    for t = 1, seq_length do
        prediction = predictions[t]
        tmp_str = vocab[x:squeeze()[t]] .. "\t"
        for m = 1, prediction:size(2) do
            tmp_str = tmp_str .. '  ' .. string.format("%.3f", prediction[{1, m}])
        end
        -- if opt.printing then print(tmp_str) end
        final_pred = final_pred + prediction
        
        local _, pred_ind = hiber_predictions[t]:max(2)
        local gt_ind = torch.range(1, hiber_y:size(3))[hiber_y[t]:eq(1):byte()]
        if opt.overlap and gt_ind:size(1) == 1 then gt_ind = torch.Tensor{gt_ind[1], gt_ind[1]} end

        if opt.printing then
          print(tmp_str)
          print(hiber_predictions[t])
          io.read()
          -- print(hidden_state[t][4])
          -- io.read()
        end

        -- if gt_ind[1] == 11 then print(hiber_predictions[t]) io.read() end
        if opt.overlap then
            if pred_ind[1] == gt_ind[1] then
                hiber_confusion:batchAdd(torch.Tensor{pred_ind:squeeze()}, torch.Tensor{gt_ind[1]})
            else
                hiber_confusion:batchAdd(torch.Tensor{pred_ind:squeeze()}, torch.Tensor{gt_ind[2]})
            end
        else
            hiber_confusion:batchAdd(torch.Tensor{pred_ind:squeeze()}, torch.Tensor{gt_ind:squeeze()})
        end

    end
    final_pred = final_pred/x:size(2)

    local tmp_str = "Total:\t"
    for m = 1, final_pred:size(1) do
        tmp_str = tmp_str .. "  " .. string.format("%.3f", final_pred[{m}])
    end
    -- if opt.printing then
    --     print(tmp_str)
    --     print(final_pred:sum())
    --     print(final_pred)
    --     print(y)
    --     io.read()
    -- end
    total = total + 1

    local _, res_rank = torch.sort(final_pred)
    if not opt.overlap then
        y = y[1]
        res_y = res_rank[-1]
        confusion:batchAdd(torch.Tensor{res_y}, torch.Tensor{y})
    else
        res_y1 = res_rank[-1]
        res_y2 = res_rank[-2]
        if (res_y1 == y[1] and res_y2 == y[2]) or (res_y2 == y[1] and res_y1 == y[2]) then
            first_two = first_two + 1
        end
    end

    print("lstm confusion")
    print(confusion)
    print("hiber confusion")
    print(hiber_confusion)
    print("overlap accuracy")
    print(first_two / total)
end

if opt.overlap then
    print("Accuracy as first highest two are correct")
    print(first_two / total)
end
