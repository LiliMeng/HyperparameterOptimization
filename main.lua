require 'rnn'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'randomkit'
require 'nngraph'
require 'torch'

local dataLoader = require 'dataLoader'
local ardUtils = require 'ardUtils'


----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
-f,--freq          (default "16")       determines which dataset to use
-h,--nhidden       (default 50)          number of hidden neurons for LSTM
-m,--model         (default "nn")       type of model "nn" or "rnn"
-p,--order         (default 16)           order of autoregressive model
-b,--batchSize     (default 40)          batchSize for NN model
-r,--rho           (default 14)          time steps for BPTT for RNN model
-e,--epoches       (default 250)         number of epoches for training
-r,--learningRate  (default "5e-3")      learning rate
-d,--decay         (default 0.96)        learning rate decay
-v,--eval_every    (default 5)           evaluate full training set every so epoches
-s,--seed          (default 12345)       seed for random
--save                                   flag for whether we should save the model and evals
--ard                                    use the differentials as input
]]

math.randomseed(opt.seed)
torch.manualSeed(opt.seed)



-- These indicate which dataset to use!
local frequency = opt.freq
local ninputs = 2
local noutputs = 2


--[[ HyperParameters ]]--
local nhidden = opt.nhidden
local batchSize = opt.batchSize
local rho = opt.rho
local lr = opt.learningRate
local decay = opt.decay
local epoches = opt.epoches
local seed = opt.seed


-- Create hash
local hash = frequency .. 'f_' .. opt.order .. 'p_'

if opt.ard then
  hash = hash .. 'diff_'
end

if opt.model == "rnn" or opt.model == "lrnn" then
  hash = hash .. opt.model .. '_' .. rho .. 'r_'
elseif opt.model == "nn" then
  hash = hash .. opt.model .. '_' .. batchSize .. 'b_'
else
  error('Unknown model.')
end
hash = hash .. nhidden .. 'h_' .. epoches .. 'ep_' .. lr .. '_' .. decay .. '_' .. seed

print('Hash:', hash)



local get_inputs

if opt.ard then
  get_inputs = function(inputs, targets, t, order)
    return ardUtils.create_ard_sequence(inputs, targets, t, order)
  end
else
  get_inputs = function(inputs, targets, t, order)
    return ardUtils.create_ar_sequence(inputs, targets, t, order)
  end
end

print('Loading and Pre-processing Dataset.')

--[[ Load Data ]]--
local ignore = 1 -- time column
local seqs = dataLoader.load('/home/lili/workspace/RNN/rnn_lili/data/freq' .. frequency .. '/', ignore, ninputs, noutputs)
print('Loaded sequences', seqs)
print('Loaded %d sequences together',#seqs)
assert(#seqs > 0, 'Failed to load any data. Aborting.')

--[[ Process Data ]]--
-- scale all joystick and odometry data to within [-1,1]
-- while preserving the meaning of 0 to indicate no movement.
scale_in = 0
scale_out = 0
for i=1,#seqs do
  max_in = torch.max(torch.abs(seqs[i]['inputs']))
  if scale_in < max_in then
    scale_in = max_in
  end
end

local all_outputs = seqs[1]['outputs']

for i=2,#seqs do
  all_outputs = torch.cat(all_outputs, seqs[i]['outputs'],1)
end

scale_out = torch.std(all_outputs)

all_outputs = nil

for i=1,#seqs do
  seqs[i]['inputs']:div(scale_in)
  seqs[i]['outputs']:div(scale_out)
end

print('Loaded ' .. #seqs .. ' sequences.')

-- take one sequence as test sequence
local testSeqNum=3
testSeq = seqs[testSeqNum]
table.remove(seqs,testSeqNum)
 
local testInputs = testSeq['inputs']
local testTargets =testSeq['outputs']

local testLength = testInputs:size(1)
     if testTargets:size(1) ~= testLength then
        print('Input and target sequences not of same length. Aborting.')
        os.exit()
     end

     print(string.format('testSeq=%02d, len=%04d', testSeqNum , testLength))


print ('testing sequence is seq is ', testSeqNum)

local modelinputs = ninputs
if opt.order >= 1 then
  modelinputs = (ninputs + noutputs)*opt.order
end

--[[ Model ]]--
local criterion = nn.MSECriterion()
model = nn.Sequential()
local rnn
local rnn_criterion
-- model:add(nn.MulConstant(1/scale_in))
if opt.model == "lrnn" then

  model:add(nn.Linear(modelinputs, nhidden))
  model:add(nn.ReLU())
  model:add(nn.FastLSTM(nhidden, nhidden))
　　model:add(nn.ReLU())
  model:add(nn.FastLSTM(nhidden, nhidden))
  model:add(nn.Linear(nhidden, noutputs))

  rnn = nn.Sequencer(model)
  rnn_criterion = nn.SequencerCriterion(criterion)

elseif opt.model == "rnn" then
  
  --model:add(nn.Linear(modelinputs, nhidden))
 -- model:add(nn.LSTM(nhidden, nhidden, rho))
 -- model:add(nn.LSTM(nhidden, nhidden, rho))
  --model:add(nn.LSTM(nhidden, nhidden, rho))
  model:add(nn.FastLSTM(modelinputs, nhidden))
  --model:add(nn.Droptout(dropout))
  model:add(nn.ReLU())
  model:add(nn.Linear(nhidden, noutputs))

  rnn = nn.Sequencer(model)
  rnn_criterion = nn.SequencerCriterion(criterion)

elseif opt.model == "nn" then
  
  print(modelinputs)
  model:add(nn.Linear(modelinputs, nhidden))
  model:add(nn.ReLU())
  model:add(nn.Linear(nhidden, nhidden2))
  model:add(nn.ReLU())
  model:add(nn.Linear(nhidden, noutputs))
  
end


-- model:add(nn.MulConstant(scale_out))

-- Get parameters and parameter gradients.
local w, dl_dw = model:getParameters()

print('Chosen Model: \n', model)

print('With Criterion:\n', criterion)

--[[ Evaluate Test Sequence ]]--
local evalTest = function(ep)
  model:evaluate()

  local T = 5;

  if opt.model == "rnn" or opt.model == "lrnn" then
    model:forget()
  end

  local inputs = testSeq['inputs']
  local targets = testSeq['outputs']
  local n = inputs:size(1)
  print('Evaluating test error on sequence of length ', n)

--  local predictions = torch.Tensor(n,2*T)
--  local seqErrors = torch.Tensor(n,1*T)
--  local absErrors = torch.Tensor(n,2*T)
--  local relErrors = torch.Tensor(n,2*T)

--  for i=1,n-T do
--    local ctargets:copy(targets)
    
--    for j=1,T do
--      local ar_input = get_inputs(inputs, ctargets, i, opt.order)
--      local target = targets[i]

--      local idx = 2*(j-1)

--      predictions[{i,{idx, idx+1}}] = model:forward(ar_input)
--      seqErrors[{i, {j-1}}] = criterion:forward(predictions[{i,{idx, idx+1}}], target)
--      absErrors[{i,{idx, idx+1}}] = torch.cmul(torch.abs(predictions[{i,{idx, idx+1}}] - target), torch.sign(target))
--      relErrors[{i,{idx, idx+1}}] = torch.cdiv(torch.abs(absErrors[i]), target)
--    end
--  end

  local predictions = torch.Tensor(n,2)
  local seqErrors = torch.Tensor(n,1)
  local absErrors = torch.Tensor(n,2)
  local relErrors = torch.Tensor(n,2)

  predictions[1] = targets[1]
  seqErrors[1] = torch.zeros(1)
  absErrors[1] = torch.zeros(2)
  relErrors[1] = torch.zeros(2)

  for i=2,n do 
    -- prepare input

    local ar_input = get_inputs(inputs, targets, i, opt.order)
    local target = targets[i]
 
    predictions[i] = model:forward(ar_input)
    seqErrors[i] = criterion:forward(predictions[i], target)
    absErrors[i] = torch.cmul(torch.abs(predictions[i] - target), torch.sign(target))
    relErrors[i] = torch.cdiv(torch.abs(absErrors[i]), target)
  end

  if opt.model == "rnn" then
    model:forget()
  end

  -- save evaluation
  if opt.save then
    lfs.mkdir('　evals2NNtestSeq3')
    lfs.mkdir('　evals2NNtestSeq3/' .. hash)
    gnuplot.pngfigure('　evals2NNtestSeq3/' .. hash .. string.format('/test_ep%02d.png', ep))
    gnuplot.raw('set terminal png size 1100,400 enhanced font "Helvetica,16"')
    gnuplot.raw('set grid')
    gnuplot.raw('set xtics -1,0.2,1')
    gnuplot.hist(relErrors, 300, -1, 1)
    gnuplot.xlabel('Signed Relative Error')
    gnuplot.title('Relative Test Error Histogram')
    gnuplot.plotflush()

    torch.save('　evals2NNtestSeq3/' .. hash .. string.format('/predictions_ep%02d.t7', ep),
    {predictions = predictions, inputs = inputs, targets = targets})
  end

  print('Epoch:', ep, 'Test Criterion Error:', torch.mean(torch.abs(seqErrors)))
  print('Epoch:', ep, 'Test Absolute Error:', torch.mean(torch.abs(absErrors)))
  print('Epoch:', ep, 'Test Relative Error:', torch.mean(torch.abs(relErrors)))
  print('================================================================================')
end


--[[ Evaluate All Training Sequences ]]--
local evalTraining = function(ep)
  print('Evaluating full training error...')
  model:evaluate()
  
  if opt.model == "rnn" then
    model:forget()
  end

  -- count full dataset size
  local N=0
  for i=1, #seqs do
    N = N + seqs[i]['inputs']:size(1)
  end
  
  local seqErrors = torch.Tensor(N,1)
  local absErrors = torch.Tensor(N,2)
  local relErrors = torch.Tensor(N,2)

  seqErrors[1] = torch.zeros(1)
  absErrors[1] = torch.zeros(2)
  relErrors[1] = torch.zeros(2)
  
  local ii=0
  for i=1,#seqs do
    local inputs = seqs[i]['inputs']
    local targets = seqs[i]['outputs']
    local n = inputs:size(1)
    
    for i=2,n do 
      -- prepare input
      local ar_input = get_inputs(inputs, targets, i, opt.order)
      local target = targets[i]

      prediction = model:forward(ar_input)
      ii = ii+1
      seqErrors[ii] = criterion:forward(prediction, target)
      absErrors[ii] = torch.cmul(torch.abs(prediction - target), torch.sign(target))
      relErrors[ii] = torch.cdiv(torch.abs(absErrors[ii]), target)
    end
    
    if opt.model == "rnn" or opt.model == "lrnn" then
      model:forget()
    end
    collectgarbage()
  end
  
  local errs = {seq=seqErrors, abs=absErrors, rel=relErrors}   
 
  -- save model and evaluation
  if opt.save then
    lfs.mkdir('models2NNtestSeq3')
    lfs.mkdir('models2NNtestSeq3/' .. hash)
    torch.save('models2NNtestSeq3/' .. hash .. string.format('/model_ep%02d.t7', ep),
    {model=model, dataset={training=seqs,test=testSeq}, scale_in=scale_in, scale_out=scale_out, errs=errs})

    lfs.mkdir('　evals2NNtestSeq3')
    lfs.mkdir('　evals2NNtestSeq3/' .. hash)
    gnuplot.pngfigure('　evals2NNtestSeq3/' .. hash .. string.format('/train_ep%02d.png', ep))
    gnuplot.raw('set terminal png size 1100,400 enhanced font "Helvetica,16"')
    gnuplot.raw('set grid')
    gnuplot.raw('set xtics -1,0.2,1')
    gnuplot.hist(relErrors, 300, -1, 1)
    gnuplot.xlabel('Signed Relative Error')
    gnuplot.title('Relative Training Error Histogram')
    gnuplot.plotflush()
  end

  print('Epoch:', ep, 'Training Criterion Error:', torch.mean(torch.abs(seqErrors)))
  print('Epoch:', ep, 'Training Absolute Error:', torch.mean(torch.abs(absErrors)))
  print('Epoch:', ep, 'Training Relative Error:', torch.mean(torch.abs(relErrors)))
  print('================================================================================')
end


--[[ Train One Epoch ]]--
local train

local sgd_config = {
  learningRate = opt.learningRate,
  learningRateDecay = opt.decay,
  momentum = 0
}

local adam_config = {
  learningRate = opt.learningRate,
  beta1 = 0.88,
  beta2 = 0.99,
  epsilon = 1e-8
}

if opt.model == "rnn" or opt.model == "lrnn" then

  train = function(ep, aLearningRate)

    local adam_config = {
      learningRate = aLearningRate,
      beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 1e-8
    }

    model:training()
    collectgarbage()

    -- train sequences in random order
    local inds = torch.randperm(#seqs)
    print ('training sequence number ',#seqs)
    for iseq=1,#seqs do

      local seq = seqs[inds[iseq]]
      local inputs = seq['inputs']
      local targets = seq['outputs']

      local n = inputs:size(1)
      if targets:size(1) ~= n then
        print('Input and target sequences not of same length. Aborting.')
        os.exit()
      end

      print(string.format('epoch=%02d, seq=%02d, len=%04d', ep, iseq, n))
      local loss = 0

      i=2

      local feval = function(w_new)

        if w ~= w_new then
          w:copy(w_new)
        end

        -- sample how many steps we should perform from a geometric dist'n.
        local steps = randomkit.geometric(1/(rho-1))

        -- if somehow this happens.
        if steps < 1 then
          print('Sampled ' .. steps .. ' steps. Something wrong?')
          steps = 1
        end

        -- don't step more than the sequence length
        if not (i + steps <= n) then steps = n-i+1 end

        -- forget old time steps if we're going to forward more steps than rho.
        if steps > rho+2 then model:forget() end

        dl_dw:zero()

        local ar_inputs = {}
        local ar_targets = {}

        for j=1,steps do

          -- prepare input
          ar_inputs[j] = get_inputs(inputs, targets, i, opt.order)
          ar_targets[j] = targets[i]

          i=i+1

        end
	
	local predictions = rnn:forward(ar_inputs)
	local sampleloss = rnn_criterion:forward(predictions, ar_targets)
	loss = loss + sampleloss
	local gradOutputs = rnn_criterion:backward(predictions, ar_targets)
	rnn:backward(ar_inputs,gradOutputs)

        return sampleloss/steps, dl_dw/steps
      end

      -- optimize
      while i<=n do
        local optim_w, optim_loss = optim.adam(feval, w, adam_config)

        if optim_loss[1] ~= optim_loss[1] then
          error('Sample error =', optim_loss[1])
        end

        if i % 100 == 0 then
          print(string.format('i=%04d', i) ,'Sample mean error=' .. optim_loss[1])
        end
      end

      model:forget()
      collectgarbage()

      print(string.format('<rnn> epoch=%02d, seq=%02d, Sequence mean error=', ep, iseq, ' ') .. loss/(n-1))
      print('================================================================================')
    end
  end

elseif opt.model == "nn" then
  train = function(ep, aLearningRate)

    local adam_config = {
      learningRate = aLearningRate,
      beta1 = 0.85,
      beta2 = 0.99,
      epsilon = 1e-8
    }

    -- train sequences in random order
    local inds = torch.randperm(#seqs)
    print ('training sequence number ',#seqs)
    print ('training sequence order: ', inds)
    for iseq=1,#seqs do

      local seq = seqs[inds[iseq]]
      local inputs = seq['inputs']
      local targets = seq['outputs']

      local n = inputs:size(1)
      if targets:size(1) ~= n then
        error('Input and target sequences not of same length. Aborting.')
      end

      print(string.format('epoch=%02d, seq=%02d, len=%04d', ep, iseq, n))

      local feval = function(w_new)

        if w ~= w_new then
          w:copy(w_new)
        end

        dl_dw:zero()

        local sampleloss = 0

        for b=1,batchSize do
          -- uniformly sample an end
          local i = math.ceil(math.random() * (n-1)) + 1

          -- prepare input
          local ar_input = get_inputs(inputs, targets, i, opt.order)
          local target = targets[i]

          -- forward and backward
          local prediction = model:forward(ar_input)
          sampleloss = sampleloss + criterion:forward(prediction, target)
          local gradOutput = criterion:backward(prediction, target)
          model:backward(ar_input, gradOutput)

        end

        return sampleloss/batchSize, dl_dw/batchSize

      end

      local loss = 0

      -- optimize
      for i=1,n/batchSize do
        local optim_w, optim_loss = optim.adam(feval, w, adam_config)

        loss = loss + optim_loss[1]

        if i % 100 == 0 then
          print(string.format('i=%04d', i) ,'Sample mean error=' .. optim_loss[1])
        end
      end

      local seqErr = loss/n*batchSize

      print(string.format('epoch=%02d, seq=%02d, Sequence mean error=', ep, iseq, ' ') .. seqErr)
      print('================================================================================')

    end
  end
end


print('Training...')
print('================================================================================')

local timer = torch.Timer()

for ep=1,epoches do

  collectgarbage()

  -- training
  train(ep, lr)

  lr = lr * decay
  local time =  timer:time()
  print('Time elapsed:', time['real'])
  print('CPU time:', time['user'])
  print('================================================================================')

  -- evaluate error
  if ep % opt.eval_every == 0 then
    evalTraining(ep)
    evalTest(ep)
  end
end

local tminus = 3
print(string.format('Training has ended. Exiting in %02d seconds.', tminus))
local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end
sleep(tminus)




--torch.save('weights.txt', model.modules[1].weight,'ascii')


