require 'nngraph'

-- input definition

local inputSize = 5
local hiddenSize = 10

local c = torch.rand(hiddenSize)
local h = torch.rand(hiddenSize)
local x = torch.rand(inputSize)

-- graph definition

local ctPrev = nn.Identity()()
local htPrev = nn.Identity()()
local xt     = nn.Identity()()

local it = nn.Sigmoid()(
   nn.CAddTable()({
         nn.Linear(inputSize, hiddenSize)(xt),
         nn.Linear(hiddenSize, hiddenSize)(htPrev)}))

local ft = nn.Sigmoid()(
   nn.CAddTable()({
         nn.Linear(inputSize, hiddenSize)(xt),
         nn.Linear(hiddenSize, hiddenSize)(htPrev)}))

local ct = nn.CAddTable()({
      nn.CMulTable()({ft, ctPrev}),
      nn.CMulTable()({
            it,
            nn.Tanh()(nn.CAddTable()({
                            nn.Linear(inputSize, hiddenSize)(xt),
                            nn.Linear(hiddenSize, hiddenSize)(htPrev)}))})})

local ot = nn.Sigmoid()(
   nn.CAddTable()({
         nn.Linear(inputSize, hiddenSize)(xt),
         nn.Linear(hiddenSize, hiddenSize)(htPrev),
         nn.Linear(hiddenSize, hiddenSize)(ct)}))

local ht = nn.CMulTable()({ot, nn.Tanh()(ct)})

local lstm = nn.gModule({ctPrev, htPrev, xt}, {ct, ht})

-- test

local res = lstm:forward({c, h, x})
print(res[1])
print(res[2])

