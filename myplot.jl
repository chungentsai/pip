using PyPlot
using DelimitedFiles


function myPlot(filename, N_EPOCH = 200)
    A = readdlm(filename, '\t', Any, '\n')
    n_line = size(A)[1]
    algs = String[]
    results = Dict()
    approx_opt = Inf

    i = 1
    d = A[i, 1]
    λ_true::Vector{Float64} = A[i+1:i+d, 1] 

    i += d+1
    while i <= n_line
        alg = A[i, 1]
        n_epoch = A[i+1, 1]
        n_rate = A[i+2, 1]

        push!(algs, alg)
        results[alg] = Dict()
        s = i+2+1
        t = i+2+n_epoch*n_rate
        results[alg]["n_epoch"] = A[s:t, 1]
        results[alg]["elapsed_time"] = A[s:t, 2]
        results[alg]["fval"] = A[s:t, 3]
        results[alg]["normalized_l2"] = A[s:t, 4]
        results[alg]["signal"] = A[t+2:t+d+1, 1]
        approx_opt = minimum([minimum(results[alg]["fval"]), approx_opt])
        i = t+d+2
    end

    path = replace(filename, "records"=> "figures") * "/"
    mkpath(path)
    close("all")


    figure(1)
    for alg in algs
        semilogy(results[alg]["n_epoch"], results[alg]["fval"] .- approx_opt, linewidth=2)
        hold
    end
    legend(algs)
    xlabel("Number of epochs")
    ylabel("Approximate optimization error")
    xlim([0, N_EPOCH])
    grid("on")
    savefig(path * "/epoch-error.png")


    figure(2)
    for alg in algs
        loglog(results[alg]["elapsed_time"], results[alg]["fval"] .- approx_opt, linewidth=2)
        hold
    end
    legend(algs)
    xlabel("Elapsed time (seconds)")
    ylabel("Approximate optimization error")
    grid("on")
    savefig(path * "/time-error.png")


    figure(3)
    for alg in algs
        semilogy(results[alg]["n_epoch"], results[alg]["normalized_l2"], linewidth=2)
        hold
    end
    legend(algs)
    xlabel("Number of epochs")
    ylabel("Normalized L2 distance")
    xlim([0, N_EPOCH])
    grid("on")
    savefig(path * "/epoch-distance.png")


    figure(4)
    for alg in algs
        loglog(results[alg]["elapsed_time"], results[alg]["normalized_l2"], linewidth=2)
        hold
    end
    legend(algs)
    xlabel("Elapsed time (seconds)")
    ylabel("Normalized L2 distance")
    grid("on")
    savefig(path * "/time-distance.png")


    figure(5)
    plot(1:d, λ_true, linewidth=2)
    hold
    for alg in algs
        plot(1:d, results[alg]["signal"], linewidth=2)
        hold
    end
    legend(vcat(["true"], algs))
    xlabel("Index")
    ylabel("Intensity")
    grid("on")
    savefig(path * "/signal.png")

end