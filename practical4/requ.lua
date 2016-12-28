require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:cmul(torch.gt(input,0):double()):cmul(input)
   return self.output
end

function ReQU:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(torch.gt(input,0):double()):cmul(2*input)
   return self.gradInput
end

