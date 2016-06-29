require 'gnuplot'
require 'rnn'

local dataLoader = require 'dataLoader'
local ardUtils = require 'ardUtils'

local opt = lapp[[
-m, --modelLoc         (default "nil")    determines which dataset to use
-o, --order            (default "nil")    order of the inputs
--ard                                     use the differentials as input
]]

local ninputs=2
local noutputs=2
local frequency=16


local modelLoc = opt.modelLoc
local order = tonumber(opt.order)

if modelLoc == "nil" then
	error('Model location must be provided.')
end

if order == nil then
	error('Model order must be provided.')
end

print('Loading model save from '.. modelLoc)

save = torch.load(modelLoc)


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


local inputs = testSeq['inputs']
local targets = testSeq['outputs']



N = targets:size(1)
d = targets:size(2)

model = save['model']

predictions = torch.zeros(N, d)

for i=10,N do

  local ar_inputs

  if opt.ard then
    ar_inputs = ardUtils.create_ard_sequence(inputs, predictions, i, order)
  else
    ar_inputs = ardUtils.create_ar_sequence(inputs, predictions, i, order) 
  end

  predictions[i] = model:forward(ar_inputs)
  
end



predictions = predictions:mul(save['scale_out'])


-- rnn_model = torch.load('models/0.1f_0p_diff_rnn_14r_50h_100ep_5e-3_0.9_12345/model_ep100.t7')['model']
-- rnn_predictions = torch.zeros(N, d)
-- for i=2,N do
--   local ar_inputs
--   if opt.ard then
--     ar_inputs = ardUtils.create_ard_sequence(inputs, rnn_predictions, i, 0)
--   else
--     ar_inputs = ardUtils.create_ar_sequence(inputs, rnn_predictions, i, 0)
--   end
--   rnn_predictions[i] = rnn_model:forward(ar_inputs)
-- end
-- rnn_predictions = rnn_predictions * save['scale_out']

targets:mul(save['scale_out'])
t = torch.linspace(1/16, N*1/16, N)


--here we just plot the linear velocity error, and use the absolute error 

predictions1 = predictions[{{},2}]
targets1= targets[{{},2}]
errors1 = predictions1 - targets1


--errors = predictions - targets


gnuplot.pngfigure('seqpredict.png')
gnuplot.raw('set terminal png size 1100,400 enhanced font "Helvetica,16"')
gnuplot.raw('set grid')
gnuplot.raw('set style line 1 lc rgb "#8FBC8F"')
gnuplot.raw('set style line 2 lc rgb "#7B68EE"')
gnuplot.raw('set style line 3 lc rgb "#DC143C"')
gnuplot.raw(string.format('set xtics 0, 100, %d', N/10))

--gnuplot.plot({'Error', t, torch.cat(errors, torch.zeros(N),2), 'filledcurves'})
gnuplot.plot({'Actual', t, targets1, '-'}, {'Prediction', t, predictions1, '-'})--, {'Error', t, torch.cat(errors, torch.zeros(N),2), 'filledcurves'})
--gnuplot.plot({'Actual', t, targets, '-'}, {'Prediction', t, predictions, '-'},{'Velocity', t, -velocities,'-'})--, {'Error', t, torch.cat(errors, torch.zeros(N),2), 'filledcurves'})
gnuplot.xlabel('Time (seconds)')
gnuplot.ylabel('Joystick forward backward deflection')
gnuplot.title('Sequence Prediction on Test Sequence 3 [MLP-50-50 With Odometry-only]')
gnuplot.plotflush()

criterion = nn.MSECriterion()
scale_out = save['scale_out']

--print(criterion:forward(predictions, targets))
print(criterion:forward(predictions:div(scale_out), targets:div(scale_out)))
--print(criterion:forward(predictions1:div(scale_out), targets1:div(scale_out)))

