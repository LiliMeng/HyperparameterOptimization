require 'nn'

ardUtils = {}

-- creates the ARD input if the target is at time t
function ardUtils.create_ard_sequence(inputs, targets, t, order)

  local dim_in = inputs:size(2)
  local dim_out = targets:size(2)

  if order >= 1 then

    if t==1 then
      return torch.zeros(order * (dim_in+dim_out))
    end

    -- creates inputs[t-p:t-1] inclusive.
    -- pads the sequence with zeros if t-p < 1
    local chunk = inputs:narrow(1, math.max(1, t-order), math.min(t-1, order))
    if t - order < 1 then
      chunk = torch.cat(torch.zeros(order-t+1, dim_in), chunk, 1)
    end

    local ar_input = chunk[-1]

    for j=2,order do
      chunk = diff(chunk)
      ar_input = torch.cat(ar_input, chunk[-1])
    end

    -- do the same thing with targets
    local chunk = targets:narrow(1, math.max(1, t-order), math.min(t-1, order))
    if t - order < 1 then
      chunk = torch.cat(torch.zeros(order-t+1, dim_out), chunk, 1)
    end

    local ar_input = torch.cat(ar_input, chunk[-1])

    for j=2,order do
      chunk = diff(chunk)
      ar_input = torch.cat(ar_input, chunk[-1])
    end

    return ar_input

  else

    return inputs[t-1]

  end
end

function ardUtils.create_ar_sequence(inputs, targets, t, order)
  local dim_in = inputs:size(2)
  local dim_out = targets:size(2)

  if order >= 1 then

    if t==1 then
      return torch.zeros(order * (dim_in+dim_out))
    end
    -- creates inputs[t-p:t-1] inclusive.
    -- pads the sequence with zeros if t-p < 1
    local chunk = inputs:narrow(1, math.max(1, t-order), math.min(t-1, order))
    if t - order < 1 then
      chunk = torch.cat(torch.zeros(order-t+1, dim_in), chunk, 1)
    end

    local t_chunk = targets:narrow(1, math.max(1, t-order), math.min(t-1, order))
    if t - order < 1 then
      t_chunk = torch.cat(torch.zeros(order-t+1, dim_out), t_chunk, 1)
    end

    return torch.cat(chunk:reshape(order * dim_in), t_chunk:reshape(order * dim_out), 1)

  else
    return inputs[t-1]
  end
end

function diff(x)
  local n = x:size(1)
  return x:narrow(1,2,n-1) - x:narrow(1,1,n-1)
end

return ardUtils
