require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
require 'model.LSTM_layer'
require 'model.HLSTM_layer'
require 'model.LSTM_h_layer'
local DataLoader = require 'util.DataLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
--local GRU = require 'model.GRU'
--local RNN = require 'model.RNN'

-- there is a better one called llap
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data tinyshakespeare
cmd:option('-data_dir','data/test_','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 32, 'size of LSTM internal state') -- to train 1vsA model
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, 1vsA_lstm or Heirarchical_lstm')
cmd:option('-hiber_gate', false, 'weather use hiber gate or not')
cmd:option('-is_balanced', false, 'if balance the training set for 1vsA model')
cmd:option('-class_weight',12)
cmd:option('-n_class', 2, 'number of categories')
cmd:option('-nbatches', 1000, 'number of training batches loader prepare')
-- optimization
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-learning_rate_decay_every', 1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-weight_decay',0.0005,'weight decay')
cmd:option('-momentum',0.90, 'momentum')
cmd:option('-dropout',0.5,'drop out, 0 = no dropout')
cmd:option('-seq_length', 9,'number of timesteps to unroll for')
cmd:option('-batch_size', 512,'number of sequences to train on in parallel')
cmd:option('-max_epochs', 9,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-finetune', false, 'finetune a 1vsA model based on AvsA model')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',5,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every', 1 ,'every how many epochs should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

opt.rnn_size = opt.rnn_size * opt.n_class

-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 
local no_update_value = -1


-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed!
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- local sigma = {0.0, 1.0}
-- local sigma = {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}
-- local sigma = {0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}
local sigma = {0.0}
local epoch = 1

-- create the data loader class
-- TODO: set input_seq_length more wisely by checking mode type
local input_seq_length = opt.seq_length
local loader = DataLoader.create(opt.data_dir, opt.batch_size, input_seq_length, split_sizes, opt.n_class, opt.nbatches)
local vocab_size = loader.vocab_size  -- the number of distinct characters
-- local vocab = loader.vocab_mapping -- what's vocab used for?
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 and not opt.finetune then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    -- for c,i in pairs(checkpoint.vocab) do 
    --     if not vocab[c] == i then 
    --         vocab_compatible = false
    --     end
    -- end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset' ..
        'and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers='
            .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    rnn_opt = {}
    rnn_opt.input_size = vocab_size
    rnn_opt.output_size = opt.n_class
    rnn_opt.rnn_size = opt.rnn_size
    rnn_opt.num_layers = opt.num_layers
    rnn_opt.dropout = opt.dropout
    rnn_opt.seq_length = opt.seq_length
    rnn_opt.no_update_value = no_update_value
    if opt.model == '1vsA_lstm' then rnn_opt.is1vsA = true
      else rnn_opt.is1vsA = false end
    -- now hiber gate is only available for 1vsA_lstm
    if opt.hiber_gate then
        opt.model = '1vsA_lstm'
        rnn_opt.is1vsA = true
        protos.rnn = nn.HLSTMLayer(rnn_opt)
    elseif opt.model == 'lstm' or opt.model == '1vsA_lstm' then
        protos.rnn = nn.LSTMLayer(rnn_opt)
    elseif opt.model == 'Hierarchical_lstm' then
        rnn_opt.conv_size = {1, 3, 3}
        rnn_opt.stride = {1, 3, 3}
        protos.rnn = nn.LSTMHierarchicalLayer(rnn_opt)
    end
    protos.criterion = nn.BCECriterion()
    if opt.hiber_gate then 
        local weights = torch.zeros(opt.n_class+1):fill(opt.class_weight)
        weights[opt.n_class+1] = 1
        -- protos.hiber_gate_criterion = nn.ClassNLLCriterion(weights) 
        protos.hiber_gate_criterion = nn.BCECriterion(weights) 
    end
end

trainLogger = optim.Logger('./log/train-'..opt.model..'-'..opt.gpuid..'.log')

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

--params, grad_params = model_utils.combine_all_parameters(protos.rnn)
params, grad_params = protos.rnn:getParameters()

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers, just uniform sampling
    protos.rnn.core = require('util.weight-init')(protos.rnn.core, 'xavier')
    if opt.hiber_gate then
        protos.rnn.hiber_gate = require('util.weight-init')(protos.rnn.hiber_gate, 'xavier')
    end
end

-- init a 1vsA model based on pre-trained AvsA model
if opt.model == '1vsA_lstm'
    and opt.finetune and string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    local weights = {}
    local bias = {}
    for layer_idx = 1, opt.num_layers do
        for _,node in ipairs(checkpoint.protos.rnn.core.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx or
                node.data.annotations.name == "h2h_" .. layer_idx then
                print('Dump params from ' .. node.data.annotations.name)
                weights[node.data.annotations.name] = node.data.module.weight
                bias[node.data.annotations.name] = node.data.module.bias
            end
        end
    end
    for _,node in ipairs(protos.rnn.core.forwardnodes) do
        if node.data.annotations.name == 'lstm_layer' then
            for _,lstm_node in ipairs(node.data.module.forwardnodes) do
                for layer_idx = 1, opt.num_layers do
                    if lstm_node.data.annotations.name == "i2h_"..layer_idx or
                        lstm_node.data.annotations.name == "h2h_"..layer_idx then
                        print('Copy params from ' .. lstm_node.data.annotations.name)
                        lstm_node.data.module.weight = weights[lstm_node.data.annotations.name]:clone();
                        lstm_node.data.module.bias = bias[lstm_node.data.annotations.name]:clone();
                    end
                end
            end
        end
    end
    do_random_init = false
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local hiber_loss = 0
    -- local rnn_state = {[0] = init_state}
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        local hiber_y
        -- hiber_y: seq_length x batch_size x output_size+1
        if opt.hiber_gate then
            hiber_y = torch.zeros(input_seq_length, y:size(1), y:size(2)+1)
            local invalid_x = x:le(26)
            for j = 1, input_seq_length do
                hiber_y[{{j},{},{1,y:size(2)}}] = y:clone()
                local indices = torch.range(1,y:size(1))[invalid_x[{{},{j}}]]
                if indices:nDimension() > 0 then
                    for i = 1, indices:size()[1] do
                        hiber_y[{{j},{indices[i]},{}}]:fill(0)
                        hiber_y[{{j},{indices[i]},{-1}}] = 1
                    end
                end
            end 
            if opt.gpuid >= 0 then
                hiber_y = hiber_y:float():cuda()
            end
        end

        -- convert to one hot vector
        -- TODO: add noise
        local x_input = torch.zeros(opt.seq_length, opt.batch_size, vocab_size)
        for t = 1, opt.seq_length do
            x_input[t] = OneHot(vocab_size):forward(x[{{}, t}])
        end
        x = x_input

        if opt.gpuid >= 0 then
            x = x:float():cuda()
            y = y:float():cuda()
        end

        protos.rnn:training()
        local proto_outputs
        local predictions, hiber_predictions
        if opt.hiber_gate then
            local cur_sigma = sigma[epoch] or sigma[#sigma]
            proto_outputs = protos.rnn:forward{x, cur_sigma, hiber_y}
            predictions = proto_outputs[1]
            hiber_predictions = proto_outputs[2]
        else
            proto_outputs = protos.rnn:forward(x)
            predictions = proto_outputs
        end
        for t = 1, opt.seq_length do
            -- if opt.is_balanced then protos.criterion = nn.BCECriterion(y*(opt.n_class-2)+1); end
            local prediction = predictions[t]:clone():cmul(predictions[t]:ne(no_update_value):typeAs(predictions))
                           + y:clone():cmul(predictions[t]:eq(no_update_value):typeAs(predictions))
            loss = loss + protos.criterion:forward(prediction, y)
                        / prediction:min(2):ne(no_update_value):sum() * prediction:size(1)
            if opt.hiber_gate then
                local _, hiber_y_indices = hiber_y[t]:max(2)
                hiber_y_indices = hiber_y_indices:squeeze():cuda()
                -- hiber_loss = hiber_loss + protos.hiber_gate_criterion:forward(torch.log(hiber_predictions[t]), hiber_y_indices)
                hiber_loss = hiber_loss + protos.hiber_gate_criterion:forward(hiber_predictions[t], hiber_y[t])
            end
        end

        --rnn_state[0] = rnn_state[#rnn_state] did we need this?
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    hiber_loss = hiber_loss / opt.seq_length / n
    return {loss, hiber_loss}
end

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1) -- 1 -> trianing
    local hiber_y
    -- hiber_y: seq_length x batch_size x output_size+1
    if opt.hiber_gate then
        hiber_y = torch.zeros(input_seq_length, y:size(1), y:size(2)+1)
        local invalid_x = x:le(26)
        for j = 1, input_seq_length do
            hiber_y[{{j},{},{1,y:size(2)}}] = y:clone()
            local indices = torch.range(1,y:size(1))[invalid_x[{{},{j}}]]
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
    end
        -- convert to one hot vector
    local x_input = torch.zeros(input_seq_length, opt.batch_size, vocab_size)
    for t = 1, input_seq_length do
        x_input[t] = OneHot(vocab_size):forward(x[{{},t}])
    end
    x = x_input

    if opt.gpuid >= 0 then
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end

    -- this is for random dropping a few entries' gradients
    --[[
    d_rate = 0.5
    randdroping_mask = y:clone()
    chosen_mask = torch.randperm(10)[{{1,math.floor(opt.n_class*d_rate)}}]:cuda()
    chosen_mask = chosen_mask:repeatTensor(y:size(1), 1)
    randdroping_mask:scatter(2, chosen_mask, 1)
    --]]

    ------------------- forward pass -------------------
    local loss = 0
    local hiber_loss = 0
    protos.rnn:training()
    local proto_outputs, 
          predictions,
          hiber_predictions,
          dhiber_predictions
    if opt.hiber_gate then
        local cur_sigma = sigma[epoch] or sigma[#sigma]
        proto_outputs = protos.rnn:forward{x, cur_sigma, hiber_y}
        predictions = proto_outputs[1]
        hiber_predictions = proto_outputs[2]
        dhiber_predictions = hiber_predictions:clone():fill(0)
    else 
        proto_outputs = protos.rnn:forward(x)
        predictions = proto_outputs
    end
    -- derivative computation
    local dpredictions = predictions:clone():fill(0)
    if opt.hiber_gate then assert(dpredictions:size(3) == dhiber_predictions:size(3)-1) end
    -- print(predictions:max(), predictions[predictions:ge(0)]:mean())
    for t = 1, opt.seq_length do
        local prediction = predictions[t]:clone():cmul(predictions[t]:ne(no_update_value):typeAs(predictions))
                           + y:clone():cmul(predictions[t]:eq(no_update_value):typeAs(predictions))
        loss = loss + protos.criterion:forward(prediction, y)
                        / prediction:min(2):ne(no_update_value):sum() * prediction:size(1)
        dpredictions[t]:copy(protos.criterion:backward(prediction, y))
        if opt.hiber_gate then
            local _, hiber_y_indices = hiber_y[t]:max(2)
            hiber_y_indices = hiber_y_indices:squeeze():cuda()
            -- hiber_loss = hiber_loss + protos.hiber_gate_criterion:forward(
            --                                 torch.log(hiber_predictions[t]),
            --                                 hiber_y_indices)
            -- dhiber_predictions[t]:copy(protos.hiber_gate_criterion:backward(
            --                                 torch.log(hiber_predictions[t]),
            --                                 hiber_y_indices))

            hiber_loss = hiber_loss + protos.hiber_gate_criterion:forward(
                                            hiber_predictions[t],
                                            hiber_y[t])
            dhiber_predictions[t]:copy(protos.hiber_gate_criterion:backward(
                                            hiber_predictions[t],
                                            hiber_y[t]))
        end
        if opt.model == '1vsA_lstm' and opt.is_balanced then
            --TODO: find out a more delegate way
            dpredictions[t]:cmul(y*12+1)
        end
        --cmul(randdroping_mask), y) -- to randomly drop with a rate of d_rate
    end
    -- the loss is the average loss across time steps
    loss = loss / opt.seq_length
    hiber_loss = hiber_loss / opt.seq_length
    ------------------ backward pass -------------------
    local dimg
    if opt.hiber_gate then
        dimg = protos.rnn:backward(x, {dpredictions, dhiber_predictions})
    else
        dimg = protos.rnn:backward(x, dpredictions)
    end
    grad_params:div(opt.seq_length)
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return {loss, hiber_loss}, grad_params
end

-- start optimization here
print("start training:")
-- rmsprop
local optim_state = { learningRate = opt.learning_rate,
                      alpha = opt.momentum}
-- sgd
-- local optim_state = { learningRate = opt.learning_rate,
--                       weightDecay = opt.weight_decay,
--                       momentum = opt.momentum}


local iterations = opt.max_epochs * loader.ntrain
local loss0 = nil
local hiber_loss0 = nil
for i = 1, iterations do
    local new_epoch = math.ceil(i / loader.ntrain)
    local is_new_epoch = false
    if new_epoch > epoch then 
        epoch = new_epoch
        is_new_epoch = true
    end

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    -- local _, loss = optim.sgd(feval, params, optim_state)
    local lstm_loss = loss[1][1]
    local hiber_loss = loss[1][2]
    local time = timer:time().real

    trainLogger:add{
        ['LSTM-Loss'] = lstm_loss,
        ['Hiber-loss'] = hiber_loss,
    }
    trainLogger:style{['LSTM-Loss']= '-', ['Hiber-loss'] = '-'}
    trainLogger.showPlot = false
    trainLogger:plot()
    os.execute('convert -density 200 '..'./log/train-'..opt.model..'-'..opt.gpuid..'.log.eps ./log/train-'..opt.model..'-'..opt.gpuid..'.png')
    os.execute('rm ./log/train-'..opt.model..'-'..opt.gpuid..'.log.eps')

    -- exponential learning rate decay
    if is_new_epoch and opt.learning_rate_decay < 1 and epoch % opt.learning_rate_decay_every == 0 then
        local decay_factor = opt.learning_rate_decay
        optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
        print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
    end

    -- every now and then or on last iteration
    if is_new_epoch and epoch % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_losses = eval_split(2) -- 2 = validation
        local val_loss = val_losses[1]
        local val_hiber_loss = val_losses[2]

        local savefile = string.format('%s/%s_epoch%d_%.2f_%.2f_gpuid%d.t7', opt.checkpoint_dir, opt.model, epoch-1, val_loss, val_hiber_loss, opt.gpuid)
        print('Validating: '..epoch.."'s epoch loss = "..val_loss.." hiber_loss = "..val_hiber_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d epoch %d loss = %6.8f hiber_loss = %6.8f grad/param norm = %6.4e max grad = %6.4e min grad = %6.4e mean grad = %6.4e time/batch = %.2fs", i, iterations, epoch, lstm_loss, hiber_loss, grad_params:norm() / params:norm(), grad_params:max(), grad_params:min(), grad_params:mean(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    -- loss = NaN
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    -- loss exploding
    if loss0 == nil then loss0 = lstm_loss end
    -- if lstm_loss > loss0 * 3 then
    --     print('lstm loss is exploding, aborting.')
    --     break
    -- end
    if hiber_loss0 == nil then hiber_loss0 = hiber_loss end
    if hiber_loss > hiber_loss0 * 3 then
        print('hiber loss is exploding, aborting.')
        break
    end
end
