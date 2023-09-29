using Printf


function init_output(len::Int64)
    output = Dict()
    output["n_epoch"]      = zeros(Float64, len)
    output["elapsed_time"] = zeros(Float64, len)
    output["fval"]         = zeros(Float64, len)
    output["normalized_l2"] = zeros(Float64, len)
    return output
end


function update_output!(output, index, t, time, fval, error)
    output["n_epoch"][index] = t
    output["elapsed_time"][index] = time
    output["fval"][index] = fval
    output["normalized_l2"][index] = error
end


function print_output(io, output, index, verbose)
    @printf(io, "%.1f\t%E\t%E\t%E\n", output["n_epoch"][index], output["elapsed_time"][index], output["fval"][index], output["normalized_l2"][index])
    if verbose
        @printf("%.1f\t%E\t%E\t%E\n", output["n_epoch"][index], output["elapsed_time"][index], output["fval"][index], output["normalized_l2"][index])
    end
    flush(io)
end


function print_signal(io, λ)
    d = size(λ)[1]
    @printf(io, "%d\n", d)
    for i = 1:d
        @printf(io, "%E\n", λ[i])
    end
end