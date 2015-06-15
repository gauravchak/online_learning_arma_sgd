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
function updateWeights(theta::Array{Float64}, x::Array{Float64}, y::Float64, learningRate::Float64)
    theta = theta - learningRate .* gradient(theta, x, y)
end

# return predictions for a column
function getPred( signalCol::DataArray{Float64,1}, windowSize::Int64, learningRate::Float64, intercept::Bool, decayfactor::Float64, errorWindow::Int64)
    if intercept
        theta = ones(windowSize+1)
        x = zeros(windowSize+1)
        #decay = ones(windowSize)
        x[windowSize+1] = 1
        theta[windowSize+1] = 0
        #decay[windowSize+1] = 1
    else
        theta = ones(windowSize)
        x = zeros(windowSize)
        decay = ones(windowSize)
    end
    
    phi = ones(errorWindow)
    e   = zeros(errorWindow)
    predCol = zeros(length(signalCol))
    
    theta = [theta, phi]
    for i in 1:(length(signalCol) - 1)
        if i <= windowSize
            x[i] = signalCol[i]
            #decay[i] = decayfactor ^ i
        else
            if intercept
                x = [x[2:windowSize], signalCol[i], x[windowSize + 1]]
            else
                x = [x[2:windowSize], signalCol[i]]
            end
        end
        
        if i <= errorWindow
            e[i] = signalCol[i] - predCol[i]
        else
            e = [e[2:errorWindow], (signalCol[i] - predCol[i])]
        end

        predCol[i+1] = hypothesis(theta, [x, e])   
        theta = updateWeights(theta, [x, e], signalCol[i+1], learningRate)
        
        if i > 3
            mse = sum((predCol[2:i] - signalCol[2:i]).^2)/(i-3)
            predCol[i+1] = predCol[i+1] + mse/2
        end
    end
    return predCol
end

function getPredSignals(dt::DataFrame, windowSize::Int64, learningRate::Float64, intercept::Bool, decayFactor::Float64, errorWindow::Int64)
    dt = sort(dt, cols = :date)
    colNames = names(dt)
    nsignals = size(dt)[2] - 1
    colNames = colNames[2:(nsignals+1)]
    
    predSignal = DataFrame()
    for col in colNames
        predSignal[col] = getPred(dt[col], windowSize, learningRate, intercept, decayFactor, errorWindow::Int64)
    end
    
    return predSignal
    #w = ones(n) ./ n #initialising with equal weight
    
end

function evalWeights(signalVec::Array{Float64}, balancing_Factor::Float64)
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

function main(dt)
    #dt = DataFrames.readtable("F:\input_combination_signals.csv")
    predSignal = getPredSignals(dt, 252, 0.5, false, 1.0, 50)
    shape_predSignal = size(predSignal)
    weights = zeros(shape_predSignal[1], shape_predSignal[2])
    
    for i in 1:size(predSignal)[1]
        weights[i,:] = evalWeights(array(predSignal[i,:]), 0.8)
    end
    return weights
end