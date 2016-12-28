require 'nngraph'

-- module definition

local function makeGraphModule(hiddenSize, bias, weightVal)
 
   local x1 = nn.Identity()()
   local x2 = nn.Identity()()
   local x3 = nn.Identity()()

   local linear = nn.Linear(hiddenSize, hiddenSize, bias == nil and true or bias)
   if(weightVal) then
      linear.weight:fill(weightVal)
   end
   
   local z = nn.CAddTable()({x1, nn.CMulTable()({x2, linear(x3)})})
   
   return nn.gModule({x1, x2, x3}, {z})
end

-- module test

local g = makeGraphModule(3, false, 1) -- no bias, all weights equal 1
local x = {torch.Tensor{1, 2, 3},
           torch.Tensor{1, 2, 3},
           torch.Tensor{1, 2, 3}}

print(g:forward(x))

