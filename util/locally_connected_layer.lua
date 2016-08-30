local LocallyConnected, parent = torch.class('nn.LocallyConnected', 'nn.Module')

function LocallyConnected:__init(inputSize, outputSize, ngroup)
    parent.__init(self)
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.ngroup = ngroup

    assert(inputSize % ngroup == 0, "invalid input size and group number")
    assert(outputSize % ngroup == 0, "invalid input size and group number")

    local igroup_size = inputSize / ngroup
    local ogroup_size = outputSize / ngroup
    
    self.model = nn.Concat(2)
    for n = 1, ngroup do
        local partial_model = nn.Sequential()
        partial_model:add(nn.Narrow(2, (n-1)*igroup_size+1, igroup_size))
        partial_model:add(nn.Linear(igroup_size, ogroup_size))
        self.model:add(partial_model)
    end
end

function LocallyConnected:parameters()
  return self.model:parameters()
end

function LocallyConnected:training()
    self.model:training()
end

function LocallyConnected:evaluate()
    self.model:evaluate()
end

function LocallyConnected:updateOutput(input)
  return self.model:updateOutput(input)
end

function LocallyConnected:updateGradInput(input, gradOutput)
  return self.model:updateGradInput(input, gradOutput)
end

function LocallyConnected:accGradParameters(input, gradOutput)
  return self.model:accGradParameters(input, gradOutput)
end
