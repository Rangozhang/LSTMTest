require 'util.OneHot'
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local DataLoader = {}
DataLoader.__index = DataLoader 

function split(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

function DataLoader.create(data_dir, batch_size, seq_length, split_fractions, n_class, nbatches, isOverlappingData, isBatchEvenly)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}
    local self = {}
    self.n_class = n_class
    self.nbatches = nbatches or 1000

    setmetatable(self, DataLoader)

    local input_file = path.join(data_dir, 'input.txt')
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, 'data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)
        if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        DataLoader.text_to_tensor(input_file, vocab_file, tensor_file)
    end

    print('loading data files...')
    local saveddata = torch.load(tensor_file)
    self.vocab_mapping = torch.load(vocab_file)
    --print(saveddata)
    --print(saveddata.label)
    --io.read()

    -----------------------------------------------
    -- for test set
    if isOverlappingData then
        test_file = 'overlapping_test.txt'
    else
        test_file = 'test.txt'
    end

    local test_input_file = path.join(data_dir, test_file)
    local test_tensor_file = path.join(data_dir, 'test_data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local test_run_prepro = true
    if not ((path.exists(vocab_file) and path.exists(test_tensor_file))) then
        -- prepro files do not exist, generate them
        print('vocab.t7 and test_data.t7 do not exist. Running preprocessing...')
        test_run_prepro = true
    end
    if test_run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. test_input_file .. '...')
        DataLoader.text_to_tensor(test_input_file, vocab_file, test_tensor_file)
    end

    print('loading data files...')
    local test_saveddata = torch.load(test_tensor_file)

    self.test_x = test_saveddata.data
    self.test_y = test_saveddata.label
    self.test_n_data = test_saveddata.n_data

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local data = saveddata.data
    local label = saveddata.label
    local n_data = saveddata.n_data
    local n_data_per_class = n_data / n_class
    self.n_data = n_data

    self.x_batches = {}
    self.y_batches = {}
    ind_batch = torch.Tensor(batch_size)
    local n_data_class_batch = math.ceil(batch_size / n_class)
    for j = 1, batch_size do
        if isBatchEvenly then
            ind_batch[j] = n_data_per_class * math.floor((j-1)/n_data_class_batch) + math.ceil(torch.uniform()*n_data_per_class)
        else
            ind_batch[j] = math.ceil(torch.uniform()*n_data)
        end
    end
    --print(ind_batch)
    --io.read()
    strt_batch = torch.Tensor(batch_size):fill(1)
    for i = 1, self.nbatches do
        local batch_data = torch.Tensor(batch_size, seq_length)
        local batch_label = torch.Tensor(batch_size)
        for j = 1, batch_size do
            local n_char = data[ind_batch[j]]:size(1)
            if n_char < strt_batch[j] + seq_length then
                strt_batch[j] = 1
                if isBatchEvenly then
                    ind_batch[j] = ind_batch[j] + 1
                    if ind_batch[j] > n_data_per_class then
                        ind_batch[j] = n_data_per_class * math.floor((j-1)/n_data_class_batch) + math.ceil(torch.uniform()*n_data_per_class)
                    end
                else
                    ind_batch[j] = math.ceil(torch.uniform()*n_data)
                end
            end
            --tmp[ind_batch[j]] = tmp[ind_batch[j]] + 1

            batch_data[{j, {}}]:copy(data[ind_batch[j]]:sub(strt_batch[j], strt_batch[j]+seq_length-1))
            strt_batch[j] = strt_batch[j] + seq_length
            batch_label[j] = label[ind_batch[j]]
       end
        self.x_batches[i] = batch_data:clone()
        self.y_batches[i] = OneHot(n_class):forward(batch_label:clone())
    end

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}
    self.test_batch_ix = 0

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function DataLoader:next_test_data()
    self.test_batch_ix = self.test_batch_ix + 1
    if self.test_batch_ix > #self.test_y then
        self.test_batch_ix = 1
    end
    return self.test_x[self.test_batch_ix], self.test_y[self.test_batch_ix]
end

function DataLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function DataLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]
end


function DataLoader:next_batch_wrt_label(split_index, cur_label)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    x_batch = nil
    y_batch = nil
    local y = self.y_batches[ix]
    local tmp_y = torch.Tensor(y:size(1)):fill(0)
    for y_ind = 1, y:size(1) do
        if y[y_ind][cur_label] == 1 then
               tmp_y[y_ind] = 1
        else
               tmp_y[y_ind] = 0
        end
    end
    y = tmp_y
    for cls_ind = 1, self.n_class do
        local tmp_y_ = torch.Tensor(y:size(1)):fill(0)
        for y_ind = 1, y:size(1) do
            if self.y_batches[ix][y_ind][cls_ind] == 1 then
                   tmp_y_[y_ind] = 1
            else
                   tmp_y_[y_ind] = 0
            end
        end
        local ind = torch.range(1, y:size(1)):maskedSelect(tmp_y_:eq(1):byte()):long()
        if cls_ind ~= cur_label then
            ind = ind[{{1, math.ceil(self.batch_size/self.n_class)}}]
        end
        if x_batch then
            x_batch = torch.cat(x_batch, self.x_batches[ix]:index(1, ind), 1)
            y_batch = torch.cat(y_batch, y:index(1, ind), 1)
        else
            x_batch = self.x_batches[ix]:index(1, ind)
            y_batch = y:index(1, ind)
        end
    end
    return x_batch, y_batch
end

-- *** STATIC method ***
function DataLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile)
    local timer = torch.Timer()
    print('loading text file...')

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    -- record all characters to a set
    local ordered = {}
    local lowercase = 'abcdefghijklmnopqrstuvwxyz'
    for c in (lowercase .. string.upper(lowercase)):gmatch"." do
        ordered[#ordered+1] = c
    end
    table.sort(ordered)
    local n_line = 0
    local n_chars = {}
    for line in io.lines(in_textfile) do
        -- print(line)
        local count_chars = 0
        for char in line:gmatch'[%a_]' do
            count_chars = count_chars+1 
        end
        n_line = n_line + 1
        n_chars[n_line] = count_chars
    end
    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
    -- construct a tensor with all the data
    print('putting data into tensor...')
    --local data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
    local data = {}
    local label = {}
    -- f = io.open(in_textfile, "r")
    local cur_line = 0
    --print(n_chars)
    for line in io.lines(in_textfile) do
        cur_line = cur_line+1
        local data_per_line = torch.Tensor(n_chars[cur_line])
        local tmp_n = 0
        string_list = split(line, ' ')
        data_sequence = string_list[#string_list]
        for char in data_sequence:gmatch'.' do
            tmp_n = tmp_n + 1
            data_per_line[tmp_n] = vocab_mapping[char]
        end
        --print(data_per_line)
        --[[
        for char in line:gmatch'[%a_]' do
           tmp_n = tmp_n + 1 
           data_per_line[tmp_n] = vocab_mapping[char]
        end
        --]]
        data[cur_line] = data_per_line:clone()
        local label_per_line = torch.Tensor(#string_list-1)
        for label_ind = 1, #string_list-1 do
            label_per_line[label_ind] = tonumber(string_list[label_ind])
        end
        --print(label_per_line)
        --io.read()
        label[cur_line] = label_per_line:clone()
        --[[
        for char in line:gmatch'%d' do
           label[cur_line] = (label[cur_line] or 0)*10 + tonumber(char)
        end  
        --]]
        --[[
        --local currlen = 0
        --rawdata = f:read(cache_len)
        --repeat
            for i=1, #rawdata do
                data[currlen+i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
            end
            -- currlen = currlen + #rawdata
            -- rawdata = f:read(cache_len)
        --until not rawdata
        --f:close()
        --]]
    end
    --[[
    print(data[1])
    print(data[2])
    print(label)
    io.read()
    --]]
    saved_data = {
        data = data,
        label = label,
        n_data = n_line
    }

    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, saved_data)
end

return DataLoader

