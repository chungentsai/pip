include("./myplot.jl")
include("./functions.jl")
include("./utils.jl")
include("./batch_algorithms.jl")
include("./stochastic_algorithms.jl")


using MKL
using TimerOutputs;
using Random
using Dates


BLAS.set_num_threads(8);


mkpath("./records")
const filename = "./records/" * Dates.format(now(), "yyyy-mm-dd-HH-MM-SS")
const io       = open(filename, "a")
const to = TimerOutput();
reset_timer!(to)

# setup
const w = 256      # width
const d = w*w      # dimension
const n = w*w    # number of measurements
const p = 0.1
const λ_true = true_parameters(w) # Poisson parameters
print_signal(io, λ_true)
const A = sensing_matrix(n, d, p) # sensing matrix
const y = generate_observations(A, λ_true)
f(λ) = f(A, y, λ)
∇f(λ) = ∇f(A, y, λ)

# transform to kelly form
const A_csum = vec(sum(A,dims=1))
const Y = sum(y)
const B = Y * A ./ transpose(A_csum)
const P = y / Y
Kellyf(x) = Kellyf(B, P, x)
∇Kellyf(x) = ∇Kellyf(B, P, x)
∇2Kellyf(x) = ∇2Kellyf(B, P, x)
x_to_λ(x) = x_to_λ(A_csum, Y, x)

# algorithms
const batch_algs = [EMD, NoLips, FW, EM]
const stochastic_algs = [SPDHG, LB_SDA, d_sample_LB_SDA, SLBOMD, SSB]
const N_EPOCH_S = 200
const N_RATE_S = 1
const N_EPOCH_B = 600
const N_RATE_B = 1
const VERBOSE = false
run_alg(alg, n_epoch, n_rate) = alg(n_epoch, n_rate)

try
    global results = Dict()

    for alg in batch_algs
        results[alg] = run_alg(alg, N_EPOCH_B, N_RATE_B)
    end

    for alg in stochastic_algs
        results[alg] = run_alg(alg, N_EPOCH_S, N_RATE_S)
    end

finally
    close(io)

end


myPlot(filename)