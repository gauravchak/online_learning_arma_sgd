using DataFrames

@doc doc"""
@Function: hypothesis
find the value of the linear hypothesis function trans(theta) * x

@param theta: Array of theta vector, i.e., [a_1, a_2, ....., a_p, c, b_1, b_2, ......, b_q]
@param     x: Array consisting of previous values of signal and previous prediction errors, i.e., [x[t-1], x[t-2], ......., x[t-p], 1, e[t-1], e[t-2], ........., e[t-q]]

@output returns trans(theta) * x
""" -> 
function hypothesis(theta::Array{Float64}, x::Array{Float64})
    return (theta' * x)[1]
end

@doc doc"""
@Function: gradient
compute the stochastic gradient of L2-norm loss function

@param theta: Array of theta vector, i.e., [a_1, a_2, ....., a_p, c, b_1, b_2, ......, b_q]
@param     x: Array consisting of previous values of signal and previous prediction errors, i.e., [x[t-1], x[t-2], ......., x[t-p], 1, e[t-1], e[t-2], ........., e[t-q]]
@param     y: Float value containing actual value of the signal

@function calls: hypothesis

@output returns an Array of gradients for each predictor variables x[t-1], x[t-2], ...x[t-p], intercept, e[t-1], e[t-2], ......., e[t-q]
""" -> 
function gradient(theta::Array{Float64}, x::Array{Float64}, y::Float64)
    hyp = hypothesis(theta, x)
    return (hyp - y) .* x
end

@doc doc"""
@Function: updateWeights
Update the theta vector using stochastic gradient descent

@param theta: Array of theta vector, i.e., [a_1, a_2, ....., a_p, c, b_1, b_2, ......, b_q]
@param     x: Array consisting of previous values of signal and previous prediction errors, i.e., [x[t-1], x[t-2], ......., x[t-p], 1, e[t-1], e[t-2], ........., e[t-q]]
@param     y: Float value containing actual value of the signal
@param learningRate

@function calls: gradient

@output returns updated theta vector theta = theta - learningRate * gradient(theta, x, y)
""" -> 
function updateWeights(theta::Array{Float64}, x::Array{Float64}, y::Float64, learningRate::Real)
    theta = theta - learningRate .* gradient(theta, x, y)
end

@doc doc"""
@Function: getPred
Computes and rerturns the predicted values of the log returns of the input signal

@param   signalCol: Data Array containing actual values of log return of the signal on each date
@param  windowSize: Order of the Auto-Regressive Model, i.e., the value of p in the model equation
@param learningRate
@param   intercept: Boolean variable to select if intercept is part of model equation, i.e., if set to false the model equation won't have c in it.
@param errorWindow: Order of the Moving-Average Model, i.e., the value of q in the model equation
@param 	   logBias: Boolean Operator to select if log bias correction has to be used

@function calls: hypothesis: calculate the predicted value of the log returns of the signal
@function calls: updateWeights

@output predicted log returns on each date
""" -> 
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

@doc doc"""
@Function getPredSignals
Computes the predicted log returns of each signal

@param  		dt: Data Frame of signals
@param  windowSize: Order of the Auto-Regressive Model, i.e., the value of p in the model equation
@param learningRate
@param   intercept: Boolean variable to select if intercept is part of model equation, i.e., if set to false the model equation won't have c in it.
@param errorWindow: Order of the Moving-Average Model, i.e., the value of q in the model equation
@param 	   logBias: Boolean Operator to select if log bias correction has to be used

@function calls: getPred: compute the predicted log returns of each signal

@output Data Frame of predicted values of the signals on each date
""" -> 
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

@doc doc"""
@Function evalWeights
Evaluate the weights of various signals of the portfolio, on every date

@param 		  signalVec: Array of values of log returns of all signals on a date
@param balancing_Factor

@output returns the array of weights of all signals on the date
""" ->
function evalWeights(signalVec::Array{Float64}, balancing_Factor::Real)
    pos_loc = ( signalVec .>= 0 )
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

@doc doc"""
@Function main
A wrapper function which calls getPredSignals and evalWeights

@param  		dt: Data Frame of signals
@param  windowSize: Order of the Auto-Regressive Model, i.e., the value of p in the model equation
@param learningRate
@param   intercept: Boolean variable to select if intercept is part of model equation, i.e., if set to false the model equation won't have c in it.
@param errorWindow: Order of the Moving-Average Model, i.e., the value of q in the model equation
@param 	   logBias: Boolean Operator to select if log bias correction has to be used
@param balancing_Factor

@function calls: getPredSignals
@function calls: evalWeights

@output returns the weight of each signal at all dates
""" ->
function main(dt::DataFrame, windowSize::Int64, learningRate::Real, intercept::Bool, errorWindow::Int64, logBias::Bool, balancingFactor::Real)
    predSignal = getPredSignals(dt, windowSize, learningRate, intercept, errorWindow, logBias)
    shape_predSignal = size(predSignal)
    weights = zeros(shape_predSignal[1], shape_predSignal[2])
    
    for i in 1:size(predSignal)[1]
        weights[i,:] = evalWeights(convert(Array, (predSignal[i,:])), balancingFactor)
    end
    return weights
end