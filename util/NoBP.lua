local NoBP, parent = torch.class('nn.NoBP', 'nn.Module')

function NoBP:__init()
    parent.__init(self)
end

function NoBP:updateOutput(input)
    self.input = input
    return self.input
end

function NoBP:updateGradInput(input, gradOutput)
    local dinputs = input:clone():zero()
    self.gradInput = dinputs
    return self.gradInput
end
