require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local DataLoader = require 'util.DataLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

--[[

        rnn_size    num_layers  dropout      lr     result
          32            1          0        2e-3     
--]]

-- there is a better one called llap
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data tinyshakespeare
cmd:option('-data_dir','data/test','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 32, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, gru or rnn')
cmd:option('-n_class', 10, 'number of categories')
cmd:option('-nbatches', 1000, 'number of training batches loader prepare')
-- optimization
cmd:option('-learning_rate',5e-3,'learning rate')
cmd:option('-learning_rate_decay',0.1,'learning rate decay')
cmd:option('-learning_rate_decay_after', 1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length', 4,'number of timesteps to unroll for')
cmd:option('-batch_size', 480,'number of sequences to train on in parallel')
cmd:option('-max_epochs',2,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every', 2 ,'every how many epochs should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully

if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end
--]]

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
-- 96 * opt.n_class / 2
local loader = DataLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.n_class, opt.nbatches) --, false, true)
local vocab_size = loader.vocab_size  -- the number of distinct characters
local vocab = loader.vocab_mapping
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
    protos_list = {}    -- a list of protos for each class
    for i = 1, opt.n_class do
        local protos = {}
        if opt.model == 'lstm' then
            protos.rnn = LSTM.lstm(vocab_size, 1, opt.rnn_size, opt.num_layers, opt.dropout) --binary output for each LSTM
        --[[
        -- discard gru and rnn temporarily
        elseif opt.model == 'gru' then
            protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
        elseif opt.model == 'rnn' then
            protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
        --]]
        end
        protos.criterion = nn.BCECriterion()
        table.insert(protos_list, protos)
    end
end



for protos_ind = 1, opt.n_class do
--protos_ind = 10
--while true do
     -- the initial state of the cell/hidden states
    init_state = {}
    for L=1,opt.num_layers do
        local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
        if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
        if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
        table.insert(init_state, h_init:clone())
        if opt.model == 'lstm' then
            table.insert(init_state, h_init:clone())
        end
    end   
    protos = protos_list[protos_ind]

    -- ship the model to the GPU if desired
    if opt.gpuid >= 0 and opt.opencl == 0 then
        for k,v in pairs(protos) do v:cuda() end
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then
        for k,v in pairs(protos) do v:cl() end
    end

    -- put the above things into one flattened parameters tensor
    -- why use model_utils???? since it is able to flatten two networks at the same time
    params, grad_params = model_utils.combine_all_parameters(protos.rnn)
    -- params, grad_params = protos.rnn:getParameters()
    --
    -- initialization
    if do_random_init then
        params:uniform(-0.08, 0.08) -- small uniform numbers  -- just uniform sampling
    end
    -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
    if opt.model == 'lstm' then
        for layer_idx = 1, opt.num_layers do
            --print(protos.rnn.forwardnodes)
            for _,node in ipairs(protos.rnn.forwardnodes) do -- where to get forwardnodes? in nngraph
                if node.data.annotations.name == "i2h_" .. layer_idx then
                    print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                    -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                    -- which means f is from 128+1 to 256
                    node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
                end
            end
        end
    end

    print('number of parameters in the model: ' .. params:nElement())
    -- make a bunch of clones after flattening, as that reallocates memory
    -- unroll time steps of rnn and criterion
    -- This is for Unrolling
    clones = {}
    for name,proto in pairs(protos) do
        print('cloning ' .. name)
        clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
    end

    -- evaluate the loss over an entire split
    function eval_split(split_index, max_batches)
        print('evaluating loss over split index ' .. split_index)
        local n = loader.split_sizes[split_index]
        if max_batches ~= nil then n = math.min(max_batches, n) end

        loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
        local loss = 0
        local rnn_state = {[0] = init_state}
        
        for i = 1,n do -- iterate over batches in the split
            -- fetch a batch
            local x, y = loader:next_batch(split_index)
            tmp_y = torch.Tensor(y:size(1)):fill(0)
            for y_ind = 1, y:size(1) do
                if y[y_ind][protos_ind] == 1 then
                    tmp_y[y_ind] = 1
                else
                    tmp_y[y_ind] = 0
                end
            end
            y = tmp_y
            if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
                -- have to convert to float because integers can't be cuda()'d
                x = x:float():cuda()
                y = y:float():cuda()
            end
            if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
                x = x:cl()
                y = y:cl()
            end
            -- forward pass
            for t=1,opt.seq_length do
                clones.rnn[t]:evaluate() -- for dropout proper functioning
                local x_OneHot = OneHot(vocab_size)(x[{{}, t}]):cuda()
                local lst = clones.rnn[t]:forward{x_OneHot, unpack(rnn_state[t-1])}
                rnn_state[t] = {}
                for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
                prediction = lst[#lst] 
                loss = loss + clones.criterion[t]:forward(prediction, y)
            end
            -- carry over lstm state
            rnn_state[0] = rnn_state[#rnn_state]
            print(i .. '/' .. n .. '...')
        end

        loss = loss / opt.seq_length / n
        return loss
    end

    -- do fwd/bwd and return loss, grad_params
    local init_state_global = clone_list(init_state)
    -- still don't know how to change grad_params, 
    -- How copy_many_times and combine_all_parameters work
    -- How can grad_params change when clones change
    -- grad_params is being accumulated through time steps, which means the gradient for each time step is accumulated for the whole sequence length
    -- And the clones is like a pointer, which just change the original protos.rnn automatically
    function feval(x)
        if x ~= params then
            params:copy(x)
        end
        grad_params:zero()

        ------------------ get minibatch -------------------
        --[[
        total_x = torch.Tensor(1, opt.seq_length):fill(0)
        total_y = torch.Tensor(1):fill(0)
        while true do
        --]]
        --local x, y = loader:next_batch_wrt_label(1, protos_ind)
        
        ----[[
        local x, y = loader:next_batch(1)
        local tmp_y = torch.Tensor(y:size(1)):fill(0)
        for y_ind = 1, y:size(1) do
            if y[y_ind][protos_ind] == 1 then
                tmp_y[y_ind] = 1
            else
                tmp_y[y_ind] = 0
            end
        end
        y = tmp_y
        --[[
            pos_ind = torch.range(1, y:size(1)):maskedSelect(y:byte()):long()
            if pos_ind:nDimension() ~= 0 then
                neg_ind = torch.range(1, y:size(1)):maskedSelect(y:eq(0):byte()):long()[{{1, pos_ind:size(1)}}]
                total_x = torch.cat(total_x, torch.cat(x:index(1, pos_ind), x:index(1, neg_ind), 1), 1)
                total_y = torch.cat(total_y, torch.cat(y:index(1, pos_ind), y:index(1, neg_ind), 1), 1)
                if (total_y:size(1) >= opt.batch_size) then
                    break
                end
                --print(total_x)
                --print(total_y)
                --io.read()
            end
        end
        really_range = torch.range(2, opt.batch_size+1):long()
        y = total_y:index(1, really_range)
        x = total_x:index(1, really_range)
        --]]
            if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            x = x:cl()
            y = y:cl()
        end
        ------------------- forward pass -------------------
        local rnn_state = {[0] = init_state_global}
        local predictions = {}           -- softmax outputs
        local loss = 0
        for t=1,opt.seq_length do -- 1 to 50
            clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
            local x_OneHot = OneHot(vocab_size)(x[{{}, t}]):cuda()
            local lst = clones.rnn[t]:forward{x_OneHot, unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
            predictions[t] = lst[#lst] -- last element is the prediction
            loss = loss + clones.criterion[t]:forward(predictions[t], y)
        end
        -- the loss is the average loss across time steps
        loss = loss / opt.seq_length
        ------------------ backward pass -------------------
        -- initialize gradient at time t to be zeros (there's no influence from future)
        local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones, i.e. just clone the size and assign all entries to zeros
        for t=opt.seq_length,1,-1 do
            -- backprop through loss, and softmax/linear
            local doutput_t = clones.criterion[t]:backward(predictions[t], y)
            table.insert(drnn_state[t], doutput_t)
            local dlst = clones.rnn[t]:backward({x[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t])
            -- dlst is dlst_dI, need to feed to the previous time step
            -- The following two results are the same
            --[[
            _, grad = clones.rnn[t]:getParameters()
            _, grad2 = clones.rnn[49]:getParameters()
            print(grad[1])
            print(grad2[1])
            print(grad_params[1])
            io.read()
            --]]
            drnn_state[t-1] = {}
            for k,v in pairs(dlst) do
                if k > 1 then -- k == 1 is gradient on x, which we dont need
                    -- note we do k-1 because first item is dembeddings, and then follow the 
                    -- derivatives of the state, starting at index 2. I know...
                    -- Since the input is x, pre_h, pre_c for two layers
                    -- And output is cur_h, cur_c for two layers and output softlog
                    drnn_state[t-1][k-1] = v
                    -- reverse as the forward one
                end
            end
        end
        -- print 'Out of sequence'
        ------------------------ misc ----------------------
        -- transfer final state to initial state (BPTT)
        init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
        -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
        -- clip gradient element-wise
        grad_params:clamp(-opt.grad_clip, opt.grad_clip)
        -- print(grad_params)
        -- io.read()
        return loss, grad_params
    end

    -- start optimization here

    print("start training:")
    train_losses = {}
    val_losses = {}
    local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
    local iterations = opt.max_epochs * loader.ntrain
    local iterations_per_epoch = loader.ntrain
    local loss0 = nil
    local epoch = 0
    for i = 1, iterations do
        local new_epoch = math.floor(i / loader.ntrain)
        local is_new_epoch = false
        if new_epoch > epoch then 
            epoch = new_epoch
            is_new_epoch = true
        end

        local timer = torch.Timer()
        local _, loss = optim.rmsprop(feval, params, optim_state)
        local time = timer:time().real

        local train_loss = loss[1] -- the loss is inside a list, pop it
        train_losses[i] = train_loss

        -- exponential learning rate decay
        if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
            if epoch >= opt.learning_rate_decay_after then
                local decay_factor = opt.learning_rate_decay
                optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
                print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
            end
        end

        -- every now and then or on last iteration
        if is_new_epoch and epoch % opt.eval_val_every == 0 or i == iterations then
            -- evaluate loss on validation data
            local val_loss = 0 --eval_split(2) -- 2 = validation
            val_losses[i] = val_loss

            local savefile = string.format('%s/%d_lm_%s_epoch%d.t7', opt.checkpoint_dir, protos_ind, opt.savefile, epoch)
            print('saving checkpoint to ' .. savefile)
            local checkpoint = {}
            checkpoint.protos = protos
            checkpoint.opt = opt
            checkpoint.train_losses = train_losses
            checkpoint.val_loss = val_loss
            checkpoint.val_losses = val_losses
            checkpoint.i = i
            checkpoint.epoch = epoch
            checkpoint.vocab = loader.vocab_mapping
            checkpoint.loader = loader
            torch.save(savefile, checkpoint)
        end

        if i % opt.print_every == 0 then
            print(string.format("%d/%d (epoch %d), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
        end
       
        if i % 10 == 0 then collectgarbage() end

        -- handle early stopping if things are going really bad
        if loss[1] ~= loss[1] then
            print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
            break -- halt
        end
        if loss0 == nil then loss0 = loss[1] end
        if loss[1] > loss0 * 3 then
            print('loss is exploding, aborting.')
            break -- halt
        end
    end

end
