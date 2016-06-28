require 'torch'
require 'lfs'

local dataLoader = {}

--[[
ignore - number of cols to ignore at start of csv
numInputs - number of cols for input
numOutputs - num of cols for output
]]--
function dataLoader.getSeq(fileLoc, ignore, numInputs, numOutputs)
  local num_line =0
  local file0 = io.open(fileLoc, 'r')
  local file1 = io.open(fileLoc, 'r')
  local file2 = io.open(fileLoc, 'r')
  
  while true do
    line0 = file0:read()
    
    if line0==nil then break end

    num_line=num_line+1
  end 
  file0.close()
  print(num_line)

  
  local file1 = io.open(fileLoc, 'r')
  
  input1 = {}
  input = {}
  output = {}

  local i = 0
  while true do
    local line1 = file1:read()
   -- print(line1)
    if i == num_line then break end
    i = i +1
    ll = line1:split(',')
    
    input1[i] = {}
    for col=ignore+numOutputs+1,ignore+numOutputs+numInputs do
      table.insert(input1[i], ll[col])
    end
  
  end
  
  for m=1, num_line-12 do
    input[m]=input1[m+12]
  end
  input[num_line] = nil


  file1.close()
  
  local j=0
  while true do
    local line2 = file2:read()

   -- if line == nil then break end
    if j == num_line-12 then break end
    j = j +1
    ll = line2:split(',')

    output[j] = {}
    for col=ignore+1, ignore+numInputs do
      table.insert(output[j], ll[col])
    end
  --  print(output)
  end
  file2.close()
  print(inputs)
  --print(outputs)  
  return {inputs=torch.Tensor(input), outputs=torch.Tensor(output)}

end

-- Load all seq*.txt files in a directory.
function dataLoader.load(dir, ignore, numInputs, numOutputs)

  local seqs = {}
  
  for file in lfs.dir(dir) do
    print( "Found file: " .. file )
   
    if file:match('seq.*.txt') ~= nil then
      table.insert(seqs, dataLoader.getSeq(dir .. file, ignore, numInputs, numOutputs))
   end
  end

  return(seqs)
end

return dataLoader
