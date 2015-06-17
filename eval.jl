using DataFrames
function compute_log_returns(_logret_matrix, _weight_matrix)
    _log_returns = zeros(size(_logret_matrix)[1])
    for i in 1:size(_logret_matrix)[1]
        _log_return = log(sum(_weight_matrix[i,:] .* exp(_logret_matrix[i,:])))
        #Append to the series of returns
        _log_returns[i] = _log_return
    end
    return _log_returns
end

function perfstats(_log_returns)
    _annualized_percent_returns = (exp(252*mean(_log_returns))-1)*100
    _estimate_of_annual_range = sqrt(252.0) * std(_log_returns)
    _annualized_percent_stdev = ((exp(_estimate_of_annual_range) - 1) + (1 - exp(-_estimate_of_annual_range)))/2.0 * 100.0
    _sharpe = _annualized_percent_returns/_annualized_percent_stdev
    return [_annualized_percent_returns, _annualized_percent_stdev, _sharpe]
end

function getPerformanceStats(_logret_matrix, _weight_matrix)
    _log_returns = compute_log_returns(_logret_matrix, _weight_matrix)
    _net_log_returns = sum(_log_returns)
    perfStats = perfstats(_log_returns)
    _annualized_percent_returns = perfStats[1]
    _annualized_percent_stdev = perfStats[2]
    _sharpe = perfStats[3]
    
    println("Annualise_percent_returns are $_annualized_percent_returns Annualise_percent_stdev is $_annualized_percent_stdev Sharpe is $_sharpe")
end

function evaluate(_ret_frame)
    # Implement/call your functions here, feel free to import other modules
	_logret_matrix = array(_ret_frame[2:size(_ret_frame)[2]])
	_weights_matrix = main(_ret_frame, 200, 0.5, false, 50, true, 0.8)
    #_weights_matrix = ones(size(_logret_matrix)[1], size(_logret_matrix)[2]) ./ size(_logret_matrix)[2] # Equal weight by default, implement your own method to compute weights here
    return _weights_matrix
end

function __init__(path)
        _returns_data_filename = path
        _ret_frame = DataFrames.readtable(_returns_data_filename)
        _logret_matrix = array(_ret_frame[2:size(_ret_frame)[2]])
        
        for name in names(_ret_frame)[2:length(_ret_frame)]
            perfStats = perfstats(_ret_frame[name])
            _annualized_percent_returns = perfStats[1]
            _annualized_percent_stdev = perfStats[2]
            _sharpe = perfStats[3]
            
            println("$name Annualise_percent_returns are $_annualized_percent_returns Annualise_percent_stdev is $_annualized_percent_stdev Sharpe is $_sharpe")
        end
        
        _weight_matrix = evaluate(_ret_frame)
        getPerformanceStats(_logret_matrix, _weight_matrix)

end