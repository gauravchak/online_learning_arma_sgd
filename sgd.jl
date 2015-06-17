using DataFrames

# compute the hypothesis of linear function
function hypothesis(theta::Array{Float64}, x::Array{Float64})
    return (theta' * x)[1]
end

# compute stochastic gradient
function gradient(theta::Array{Float64}, x::Array{Float64}, y::Float64)
    hyp = hypothesis(theta, x)
    return (hyp - y) .* x
end

# update the weights using gradient descent
function updateWeights(theta::Array{Float64}, x::Array{Float64}, y::Float64, learningRate::Real)
    theta = theta - learningRate .* gradient(theta, x, y)
end

# return predictions for a column
function getPred( signalCol::DataArray{Float64,1}, windowSize::Int64, learningRate::Real, intercept::Bool, errorWindow::Int64, logBias::Bool)
    # initializing the Auto Regressive Part
    if intercept
        theta = ones(windowSize+1)
        x = zeros(windowSize+1)
        x[windowSize+1] = 1
        theta[windowSize+1] = 0
    else
        theta = ones(windowSize)
        x = zeros(windowSize)
    end
    
    # intializing the moving average part
    phi = ones(errorWindow)
    e   = zeros(errorWindow)
    predCol = zeros(length(signalCol))
    
    theta = [theta, phi]
    for i in 1:(length(signalCol) - 1)
        if (windowSize > 0)
            if intercept
                x = [signalCol[i], x[1:(windowSize-1)], x[windowSize + 1]]
            else
                # removing the last element and right shifting the array by one element
                x = [signalCol[i], x[1:(windowSize-1)]]
            end
        end
        
        if (errorWindow > 0)
            e = [(signalCol[i] - predCol[i]), e[1:(errorWindow-1)]]
        end

        predCol[i+1] = hypothesis(theta, [x, e])   
        theta = updateWeights(theta, [x, e], signalCol[i+1], learningRate)
        
        # log bias Correction
        if logBias & (i > 3)
            mse = sum((predCol[2:i] - signalCol[2:i]).^2)/(i-3)
            predCol[i+1] = predCol[i+1] + mse/2
        end
    end
    return predCol
end

# Predict log returns for each signal
function getPredSignals(dt::DataFrame, windowSize::Int64, learningRate::Real, intercept::Bool, errorWindow::Int64, logBias::Bool)
    dt = sort(dt, cols = :date)
    colNames = names(dt)
    nsignals = size(dt)[2] - 1
    colNames = colNames[2:(nsignals+1)]
    
    predSignal = DataFrame()
    #getting predicted log returns for each signal
    for col in colNames
        predSignal[col] = getPred(dt[col], windowSize, learningRate, intercept, errorWindow, logBias)
    end
    
    return predSignal
end

# Evaluate the weight of each signal at a given date
function evalWeights(signalVec::Array{Float64}, balancing_Factor::Real)
    pos_loc = signalVec .>= 0
    PositiveSignal = signalVec[pos_loc]
    NegativeSignal = signalVec[!pos_loc]
    
    weight = zeros(length(signalVec))'
    
    if length(NegativeSignal) == 0
        balancing_Factor = 1
    elseif length(PositiveSignal) == 0
        balancing_Factor = 0
    end
    
    sumPositive = sum(PositiveSignal)
    weight[pos_loc] =  sumPositive > 0 ? (balancing_Factor * PositiveSignal / sumPositive) : (balancing_Factor/length(PositiveSignal))
    weight[!pos_loc] = 1 ./ (NegativeSignal*sum(1 ./ NegativeSignal)) * (1 - balancing_Factor)
    
    return weight
end

function main(dt::DataFrame, windowSize::Int64, learningRate::Real, intercept::Bool, errorWindow::Int64, logBias::Bool, balancingFactor::Real)
    predSignal = getPredSignals(dt, windowSize, learningRate, intercept, errorWindow, logBias)
    shape_predSignal = size(predSignal)
    weights = zeros(shape_predSignal[1], shape_predSignal[2])
    
    for i in 1:size(predSignal)[1]
        weights[i,:] = evalWeights(array(predSignal[i,:]), balancingFactor)
    end
    return weights
end