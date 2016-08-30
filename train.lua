require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
require 'model.LSTM_layer'
require 'model.LSTM_h_layer'
local DataLoader = require 'util.DataLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

-- there is a better one called llap
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data tinyshakespeare
cmd:option('-data_dir','data/test_','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 60, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, 1vsA_lstm or Heirarchical_lstm')
cmd:option('-n_class', 10, 'number of categories')
cmd:option('-nbatches', 1000, 'number of training batches loader prepare')
-- optimization
cmd:option('-learning_rate',1e-2,'learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-learning_rate_decay_every', 1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-seq_length', 25,'number of timesteps to unroll for')
cmd:option('-batch_size', 512,'number of sequences to train on in parallel')
cmd:option('-max_epochs', 5,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
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

-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

trainLogger = optim.Logger('./log/train_'..opt.model..'.log')

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
if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    for c,i in pairs(checkpoint.vocab) do 
        if not vocab[c] == i then 
            vocab_compatible = false
        end
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
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
    if opt.model == 'lstm' or opt.model == '1vsA_lstm' then
        if opt.model == '1vsA_lstm' then rnn_opt.is1vsA = true end
        protos.rnn = nn.LSTMLayer(rnn_opt)
    elseif opt.model == 'Hierarchical_lstm' then
        rnn_opt.conv_size = {1, 3, 3}
        rnn_opt.stride = {1, 3, 3}
        protos.rnn = nn.LSTMHierarchicalLayer(rnn_opt)
    end
    protos.criterion = nn.BCECriterion()
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

--params, grad_params = model_utils.combine_all_parameters(protos.rnn)
params, grad_params = protos.rnn:getParameters()

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers, just uniform sampling
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state} -- TODO: Check where init_state comes from
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
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
        local predictions = protos.rnn:forward(x)
        for t = 1, opt.seq_length do
            loss = loss + protos.criterion:forward(predictions[t], y)
        end

        --rnn_state[0] = rnn_state[#rnn_state] did we need this?
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1) -- 1 -> trianing
    -- print(x)
    -- print(y)
    -- io.read()
    
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
    local predictions = {}           -- softmax outputs
    local loss = 0
    protos.rnn:training()
    local predictions = protos.rnn:forward(x)
    --print(predictions)
    --print(y)
    --io.read()
    local dpredictions = predictions:clone():fill(0)
    for t = 1, opt.seq_length do
        loss = loss + protos.criterion:forward(predictions[t], y)
        dpredictions[t]:copy(protos.criterion:backward(predictions[t], y))
        --cmul(randdroping_mask), y) -- to randomly drop with a rate of d_rate
    end
    -- the loss is the average loss across time steps
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    --print(dpredictions)
    --io.read()
    local dimg = protos.rnn:backward(x, dpredictions)
    -- grad_params:div(opt.seq_length)
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here

print("start training:")
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}

local iterations = opt.max_epochs * loader.ntrain
local loss0 = nil
local epoch = 1
for i = 1, iterations do
    local new_epoch = math.ceil(i / loader.ntrain)
    local is_new_epoch = false
    if new_epoch > epoch then 
        epoch = new_epoch
        is_new_epoch = true
    end

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    trainLogger:add{
        ['Loss'] = loss[1]
    }
    trainLogger:style{'-'}
    trainLogger.showPlot = false
    trainLogger:plot()
    os.execute('convert -density 200 '..'./log/train_'..opt.model..'.log.eps ./log/train_'..opt.model..'.png')
    os.execute('rm ./log/train_'..opt.model..'.log.eps')

    -- exponential learning rate decay
    if is_new_epoch and opt.learning_rate_decay < 1 and epoch % opt.learning_rate_decay_every == 0 then
        local decay_factor = opt.learning_rate_decay
        optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
        print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
    end

    -- every now and then or on last iteration
    if is_new_epoch and epoch % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation

        local savefile = string.format('%s/%s_epoch%d_%.2f.t7', opt.checkpoint_dir, opt.model, epoch, val_loss)
        print('Validating: '..epoch.."'s epoch loss = "..val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %d) loss = %6.8f grad/param norm = %6.4e max grad = %6.4e min grad = %6.4e mean grad = %6.4e time/batch = %.2fs", i, iterations, epoch, loss[1], grad_params:norm() / params:norm(), grad_params:max(), grad_params:min(), grad_params:mean(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    -- loss = NaN
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    -- loss exploding
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end
