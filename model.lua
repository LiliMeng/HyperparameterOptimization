require 'nn'
require 'cephes'
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



--[[ HyperParameters ]]--
function trainHyper(tab_params) -- change after to just trainHyper... no returning model

  tab_params['numHidden1'] = math.pow(10,tab_params['numHidden1'])




	--math.randomseed(opt.seed)
	--torch.manualSeed(opt.seed)
	math.randomseed(12345)
	torch.manualSeed(12345)



	-- These indicate which dataset to use!
	--local frequency = opt.freq
	local frequency=16
	local ninputs = 2
	local noutputs = 2
    local nhidden=tab_params['numHidden3']
   
	local batchSize = tab_params['numHidden2']
	--local batchSize = opt.batchSize
	--local lr = opt.learningRate
	local lr= tab_params['numHidden1']

	
	local decay = 0.96
	local epoches = tab_params['numHidden5']
	local seed=12345
	local model1 ='nn'
	local order=16
	local eval_every=5  -- evaluate full training set every so epoches

	-- Create hash
	local hash = frequency .. 'f_' .. order .. 'p_'.. model1 .. '_' .. batchSize .. 'b_' .. nhidden .. 'h_' .. epoches .. 'ep_' .. lr .. '_' .. decay .. '_' .. '_' .. seed

	print('Hash:', hash)

	get_inputs = function(inputs, targets, t, order)
	  return ardUtils.create_ar_sequence(inputs, targets, t, order)
	end

	print('Loading and Preprocessing Dataset.')

	-- [[ Load Data ]]--
	local ignore = 1 -- time column
	local seqs = dataLoader.load('/home/lili/workspace/RNN/rnn_lili/data/freq' .. frequency .. '/', ignore, ninputs, noutputs)
	print('Loaded sequences', seqs)
	print('Loaded %d sequences together', #seqs)
	assert(#seqs > 0, 'Failed to load any data. Aborting.')

	--[[ Process Data ]]--
	-- scale all joystick and odometry data to within [-1, 1]
	-- while preserving the meaning of 0 to indicate no movement.
	scale_in = 0
	scale_out = 0
	for i=1, #seqs do
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
	local testTargets = testSeq['outputs'] 

	local testLength = testInputs:size(1)
		if testTargets:size(1) ~= testLength then
			print('Input and target sequences not of the same length. Aborting.')
			os.exit()
		end 

		print(string.format('testSeq=%02d, len=%04d', testSeqNum, testLength))

	print ('testing sequence is seq ', testSeqNum)

	local modelinputs = ninputs
	if order >=1 then 
		modelinputs = (ninputs + noutputs)*order 
	end 

	print('order is', order)

	-- [[ Model ]]--
	local criterion = nn.MSECriterion()

	local model = nn.Sequential()

	print('model:sequential()')

	-- MLP Construction Logic
  	
    
    model:add(nn.Linear(modelinputs, tab_params['numHidden3']))
    model:add(nn.ReLU())
    model:add(nn.Linear(tab_params['numHidden3'], tab_params['numHidden4']))
    model:add(nn.ReLU())
    model:add(nn.Linear(tab_params['numHidden4'], noutputs))
  	
    
	-- Get parameters and parameter gradients.
	local w, dl_dw = model:getParameters()

	print('Chosen Model: \n', model)
	print('With Criterion: \n', criterion)


	-- [[ Evaluate Test Sequence ]] --
	local evalTest = function(ep)
		model:evaluate()

		local inputs = testSeq['inputs']
		local targets = testSeq['outputs']
		local n = inputs:size(1)
		print('Evaluating test error on sequence of length ', n)

		local predictions = torch.Tensor(n,2)
		local seqErrors = torch.Tensor(n,1)
		local absErrors = torch.Tensor(n,2)
		local relErrors = torch.Tensor(n,2)

	    predictions[1] = targets[1]
	    seqErrors[1] = torch.zeros(1)
	    absErrors[1] = torch.zeros(2)
	    relErrors[1] = torch.zeros(2)

	    for i=2, n do
	    	--prepare input
	    	local ar_input = get_inputs(inputs, targets, i, order)
	    	local target = targets[i]

	    	predictions[i] = model:forward(ar_input)
	    	seqErrors[i] = criterion:forward(predictions[i], target)
	    	absErrors[i] = torch.cmul(torch.abs(predictions[i] - target), torch.sign(target))
	    	relErrors[i] = torch.cdiv(torch.abs(absErrors[i]), target)

	    end 
	 
        
	    print('Epoch:', ep, 'Test Criterion Error:', torch.mean(torch.abs(seqErrors)))
	    print('Epoch:', ep, 'Test Absolute Error:', torch.mean(torch.abs(absErrors)))
	    print('Epoch:', ep, 'Test Relative Error:', torch.mean(torch.abs(relErrors)))
	    print('==============================================================================')
	    

	end

	--[[ Evaluate All Training Sequences ]]--
	local evalTraining = function(ep)
		print('Evaluating full training error...')
  		model:evaluate()

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
      			local ar_input = get_inputs(inputs, targets, i, order)
      			local target = targets[i]

      			prediction = model:forward(ar_input)
      			ii = ii+1
      			seqErrors[ii] = criterion:forward(prediction, target)
      			absErrors[ii] = torch.cmul(torch.abs(prediction - target), torch.sign(target))
      			relErrors[ii] = torch.cdiv(torch.abs(absErrors[ii]), target)
    		end
    
         	collectgarbage()
  		end
  
  	   	
	  	print('Epoch:', ep, 'Training Criterion Error:', torch.mean(torch.abs(seqErrors)))
	  	print('Epoch:', ep, 'Training Absolute Error:', torch.mean(torch.abs(absErrors)))
	  	print('Epoch:', ep, 'Training Relative Error:', torch.mean(torch.abs(relErrors)))
	  	print('===============================================================================')
	end


	--[[ Train One Epoch ]]--

	local train 

	local sgd_config = {
		learningRate = lr,
		learningRateDecay = decay,
		momentum = 0
	}

	local adam_config = {
		learningRate = lr,
		beta1 = 0.9,
		beta2 = 0.999,
		epsilon = 1e-8
	}

	train = function(ep, aLearningRate)

	    local adam_config = {
	      learningRate = aLearningRate,
	      beta1 = 0.9,
	      beta2 = 0.999,
	      epsilon = 1e-8
	    }

	    -- train sequences in random order
    	local inds = torch.randperm(#seqs)
    	--print ('training sequence number ',#seqs)
    	--print ('training sequence order: ', inds)
    	for iseq=1,#seqs do

      		local seq = seqs[inds[iseq]]
      		local inputs = seq['inputs']
      		local targets = seq['outputs']

      		local n = inputs:size(1)
      		if targets:size(1) ~= n then
        		error('Input and target sequences not of same length. Aborting.')
      		end

      		--print(string.format('epoch=%02d, seq=%02d, len=%04d', ep, iseq, n))

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
          			local ar_input = get_inputs(inputs, targets, i, order)
          			local target = targets[i]

          			-- forward and backward
          			local prediction = model:forward(ar_input)
          			sampleloss = sampleloss + criterion:forward(prediction, target)
          			local gradOutput = criterion:backward(prediction, target)
          			model:backward(ar_input, gradOutput)

        		end

        		return sampleloss/batchSize, dl_dw/batchSize

      		end

      		local loss=0

	   		-- optimize
	   		for i=1, n/batchSize do
	   			local optim_w, optim_loss = optim.adam(feval, w, adam_config)

	   			loss = loss + optim_loss[1]

	   			if i % 100 == 0 then
	   				--print(string.format('i=%04d', i), 'Sample mean error=' .. optim_loss[1])
	   			end
	   		end 

	   		local seqErr = loss/n*batchSize

	   		--print(string.format('epoch=%02d, seq=%02d, Sequence mean error=', ep, iseq, ' ') .. seqErr)
	   		--print('==================================================================================')


	    end
	end

	print('Training...')
	print('================================================================================')

	local timer = torch.Timer()

	
	local mse_avg

	for ep=1,epoches do

		collectgarbage()

		--training
		train(ep, lr)

		lr = lr * decay

		local time = timer:time()
		--print('Time elapsed: ', time['real'])
		--print('CPU time:', time['user'])
		--print('==============================================================================')

	end 

	-- evaluate training and testing error
	evalTraining(ep)
	model:evaluate()

	local inputs = testSeq['inputs']
	local targets = testSeq['outputs']
	local n = inputs:size(1)
	print('Evaluating test error on sequence of length ', n)

	local predictions = torch.Tensor(n,2)
	local seqErrors = torch.Tensor(n,1)
	local absErrors = torch.Tensor(n,2)
	local relErrors = torch.Tensor(n,2)

	predictions[1] = targets[1]
	seqErrors[1] = torch.zeros(1)
	absErrors[1] = torch.zeros(2)
	relErrors[1] = torch.zeros(2)

	for i=2, n do
	    --prepare input
	    local ar_input = get_inputs(inputs, targets, i, order)
	    local target = targets[i]

	    predictions[i] = model:forward(ar_input)
	    seqErrors[i] = criterion:forward(predictions[i], target)
	    absErrors[i] = torch.cmul(torch.abs(predictions[i] - target), torch.sign(target))
	    relErrors[i] = torch.cdiv(torch.abs(absErrors[i]), target)

	 end 
	 
        
	print('Epoch:', ep, 'Test Criterion Error:', torch.mean(torch.abs(seqErrors)))
	print('Epoch:', ep, 'Test Absolute Error:', torch.mean(torch.abs(absErrors)))
	print('Epoch:', ep, 'Test Relative Error:', torch.mean(torch.abs(relErrors)))
	print('==============================================================================')
	mse_avg=torch.mean(torch.abs(seqErrors))
	 
	print("Average Testing MSE: "..mse_avg)	
	return mse_avg
	
end
