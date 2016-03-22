
local PrintLayer, parent = torch.class('PrintLayer', 'nn.Module')

function PrintLayer:updateOutput(input)
    print(input)
    self.output = input
    return self.output
end
