--[[
This model use a maxpooling in the middle
--]]
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

-- there is a better one called llap
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data tinyshakespeare
cmd:option('-data_dir','data/test_','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 32, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, gru or rnn')
cmd:option('-n_class', 10, 'number of categories')
cmd:option('-nbatches', 5000, 'number of training batches loader prepare')
-- optimization
cmd:option('-learning_rate',1e-2,'learning rate')
cmd:option('-learning_rate_decay',0.1,'learning rate decay')
cmd:option('-learning_rate_decay_every', 4,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0.5,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length', 4,'last layer"s number of timesteps to unroll for')
cmd:option('-batch_size', 512,'number of sequences to train on in parallel')
cmd:option('-max_epochs', 4,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',5,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every', 2 ,'every how many epochs should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv4', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',2,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

trainLogger = optim.Logger('train_mp.log')

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully

if opt.gpuid >= 0 then
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

-- create the data loader class
local loader = DataLoader.create(opt.data_dir, opt.batch_size, opt.seq_length^2, split_sizes, opt.n_class, opt.nbatches)
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
    protos = {}
    if opt.model == 'lstm' then
        interm_size = opt.rnn_size
        protos.rnn1 = LSTM.lstm(vocab_size, interm_size, opt.rnn_size, opt.num_layers, opt.dropout)
        protos.rnn2 = LSTM.lstm(interm_size, opt.n_class, opt.rnn_size, opt.num_layers, opt.dropout, true)
    --[[
    -- discard gru and rnn temporarily
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    --]]
    end
    protos.criterion = nn.BCECriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do 
        if torch.type(v) == 'table' then
            for _, vv in pairs(v) do
                vv:cuda()
            end
        else
            v:cuda() 
        end
    end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn1, protos.rnn2)

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers  -- just uniform sampling
end


local num_level = 2
local window_size = 4
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
    for level_ind = 1, num_level do
        local rnn = level_ind == 1 and protos.rnn1 or protos.rnn2
        for layer_idx = 1, opt.num_layers do
            --print(rnn.forwardnodes)
            for _,node in ipairs(rnn.forwardnodes) do -- where to get forwardnodes? in nngraph
                if node.data.annotations.name == "i2h_" .. layer_idx then
                    print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                    -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                    -- which means f is from 128+1 to 256
                    node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
                end
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
    local rep_num = 0
    if name == 'rnn1' then rep_num = opt.seq_length^2
    else rep_num = opt.seq_length end
    clones[name] = model_utils.clone_many_times(proto, rep_num)
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {}
    local level_output = {}
    for l = 1, num_level do
        rnn_state[l] = {[0] = clone_list(init_state, true)}
        level_output[l] = {}
    end
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        
        -- forward pass
        
        -- first level
        for t=1,  opt.seq_length^2 do 
            clones.rnn1[t]:evaluate() 
            local x_OneHot = OneHot(vocab_size):forward(x[{{}, t}]):cuda()
            local lst = clones.rnn1[t]:forward{x_OneHot, unpack(rnn_state[1][t-1])}
            rnn_state[1][t] = {}
            for i=1,#init_state do table.insert(rnn_state[1][t], lst[i]) end 
            level_output[1][t] = lst[#lst]
        end

        local temporalPooling = nn.TemporalMaxPooling(3):cuda()
        
        -- merge
        local merged_output = {}
        local concate_res = {}
        for t=1, opt.seq_length do
            merged_output[t] = torch.zeros(opt.batch_size, interm_size):cuda()  
            local out_sz = level_output[1][1]:size()
            for tt=1, opt.seq_length do
                local out_t = torch.reshape(level_output[1][(t-1)*opt.seq_length+tt], 1, out_sz[1], out_sz[2])
                if tt==1 then
                    concate_res[t] = out_t
                else
                    concate_res[t] = torch.cat(concate_res[t], out_t, 1)
                end
            end
            concate_res[t] = concate_res[t]:transpose(1, 2)
            local maxpooling_input = temporalPooling:forward(concate_res[t])
            merged_output[t]:copy(torch.squeeze(maxpooling_input))
        end
        
        -- second level
        for t=1,  opt.seq_length do 
            clones.rnn2[t]:evaluate() 
            local lst = clones.rnn2[t]:forward{merged_output[t], unpack(rnn_state[2][t-1])}
            rnn_state[2][t] = {}
            for i=1,#init_state do table.insert(rnn_state[2][t], lst[i]) end 
            level_output[2][t] = lst[#lst]
            loss = loss + clones.criterion[t]:forward(level_output[2][t], y)
        end
        
        -- carry over lstm state
        rnn_state[1][0] = rnn_state[1][#(rnn_state[1])]
        rnn_state[2][0] = rnn_state[2][#(rnn_state[2])]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

local init_state_global = {clone_list(init_state, true), clone_list(init_state, true)}

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then 
        x = x:float():cuda()
        y = y:float():cuda()
    end

    local temporalPooling = {}
    for ind = 1, opt.seq_length do
        temporalPooling[ind] = nn.TemporalMaxPooling(3):cuda()
    end


    -- this is for random dropping a few entries' gradients
    d_rate = 0.5
    randdroping_mask = y:clone()
    chosen_mask = torch.randperm(10)[{{1,math.floor(opt.n_class*d_rate)}}]:cuda()
    chosen_mask = chosen_mask:repeatTensor(y:size(1), 1)
    randdroping_mask:scatter(2, chosen_mask, 1)

    ------------------- forward pass -------------------
    local rnn_state = {}
    local level_output = {}
    for l = 1, num_level do
        rnn_state[l] = {[0] = init_state_global[l]}
        level_output[l] = {}
    end
    local loss = 0

    -- first level
    for t=1,  opt.seq_length^2 do 
        clones.rnn1[t]:training() 
        local x_OneHot = OneHot(vocab_size):forward(x[{{}, t}]):cuda()
        local lst = clones.rnn1[t]:forward{x_OneHot, unpack(rnn_state[1][t-1])}
        rnn_state[1][t] = {}
        for i=1,#init_state do table.insert(rnn_state[1][t], lst[i]) end 
        level_output[1][t] = lst[#init_state]
    end

    -- merge
    local merged_output = {}
    local concate_res = {}
    for t=1, opt.seq_length do
        merged_output[t] = torch.zeros(opt.batch_size, interm_size):cuda()  
        local out_sz = level_output[1][1]:size()
        for tt=1, opt.seq_length do
            local out_t = torch.reshape(level_output[1][(t-1)*opt.seq_length+tt], 1, out_sz[1], out_sz[2])
            if tt==1 then
                concate_res[t] = out_t
            else
                concate_res[t] = torch.cat(concate_res[t], out_t, 1)
            end
        end
        concate_res[t] = concate_res[t]:transpose(1, 2)
        local maxpooling_input = temporalPooling[t]:forward(concate_res[t])
        merged_output[t]:copy(torch.squeeze(maxpooling_input))
    end
    
    -- second level
    for t=1, opt.seq_length do 
        clones.rnn2[t]:training() 
        local lst = clones.rnn2[t]:forward{merged_output[t], unpack(rnn_state[2][t-1])}
        rnn_state[2][t] = {}
        for i=1,#init_state do table.insert(rnn_state[2][t], lst[i]) end 
        level_output[2][t] = lst[#lst]
        loss = loss + clones.criterion[t]:forward(level_output[2][t]:cmul(randdroping_mask), y)
    end
    
    -- the loss is the average loss across time steps
    loss = loss / opt.seq_length
    
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {}
    drnn_state[1] = {[opt.seq_length^2] = clone_list(init_state, true)} -- clones the size and zeros it
    drnn_state[2] = {[opt.seq_length] = clone_list(init_state, true)} -- clones the size and zeros it

    local dinterm_out = {[opt.seq_length] = torch.zeros(opt.batch_size, interm_size*opt.seq_length)}

    -- second level
    for t=opt.seq_length,1,-1 do
        local doutput_t = clones.criterion[t]:backward(level_output[num_level][t], y)
        table.insert(drnn_state[2][t], doutput_t)
        local dlst = clones.rnn2[t]:backward({merged_output[t], unpack(rnn_state[2][t-1])}, drnn_state[2][t])
        -- dlst is dlst_dI, need to feed to the previous time step
        drnn_state[2][t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k >= 1 is gradient on states
                drnn_state[2][t-1][k-1] = v -- reverse as the forward one
            else
                dinterm_out[t] = v
            end
        end
    end

    -- maxpooling level
    local dinterm_in = {}
    for t=1, opt.seq_length do
        local dout_mp = dinterm_out[t]
        local din_mp = temporalPooling[t]:backward(concate_res[t], dout_mp):transpose(1, 2)
        local din_mp_split = din_mp:split(1, 1)
        for i = 1, opt.seq_length do
            dinterm_in[(t-1)*opt.seq_length+i] = torch.squeeze(din_mp_split[i])
        end
    end

    -- first level
    for t=opt.seq_length^2,1,-1 do
        -- local derv_ind = math.floor((t-1)/opt.seq_length) + 1
        -- local doutput_t = dinterm_out[derv_ind][{{}, {interm_size*((t-1)%opt.seq_length)+1, interm_size*((t-1)%opt.seq_length+1)}}]:clone()
        local doutput_t = dinterm_in[t]
        --table.insert(drnn_state[1][t], doutput_t)
        local rs_size = #drnn_state[1][t]
        drnn_state[1][t][rs_size] = drnn_state[1][t][rs_size]+doutput_t
        local dlst = clones.rnn1[t]:backward({x[{{}, t}], unpack(rnn_state[1][t-1])}, drnn_state[1][t])
        -- dlst is dlst_dI, need to feed to the previous time step
        drnn_state[1][t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k >= 1 is gradient on states
                drnn_state[1][t-1][k-1] = v -- reverse as the forward one
            end
        end
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global[1] = rnn_state[1][#rnn_state]
    init_state_global[2] = rnn_state[2][#rnn_state]
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    collectgarbage()
    return loss, grad_params
end

-- start optimization here

print("start training:")
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
local epoch = 1
for i = 1, iterations do
    local new_epoch = math.ceil(i / iterations_per_epoch)
    local is_new_epoch = false
    if new_epoch > epoch then 
        epoch = new_epoch
        is_new_epoch = true
    end

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it

    trainLogger:add{
        ['Loss'] = train_loss
    }
    trainLogger:style{'-'}
    trainLogger.showPlot = false
    trainLogger:plot()
    os.execute('convert -density 200 train_mp.log.eps train_mp.png')

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch % opt.learning_rate_decay_every == 0 then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end
    
    -- every now and then or on last iteration
    if is_new_epoch and epoch % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation

        local savefile = string.format('%s/mp_%s_epoch%d_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("[%d][%d/%d] train_loss: %6.8f grad/param norm = %6.4e time/batch = %.2fs", epoch, i%iterations_per_epoch, iterations_per_epoch, train_loss, grad_params:norm() / params:norm(), time))
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

--]]
