using Arpack
using TimerOutputs
using ExponentialUtilities
using Printf
using LinearAlgebra
using StatsBase


function SSB(n_epoch::Int64, n_rate::Int64)
    name = "SSB"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    len_output = n_epoch * n_rate
    output = init_output(len_output)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    x_bar::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x_bar)

    n_iter::Int64 = n_epoch * n
    period::Int64 = n ÷ n_rate
    @timeit to "iteration" begin
        idx = sample(1:n, Weights(P), n_iter)
        η::Float64 = sqrt( log( d ) / n_iter / d )
        η = η / ( 1.0 + η )
        y = ( 1.0 - η ) * ones(Float64, d)
    end
 
    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin
            grad = - view(B, idx[iter], :) / dot(view(B, idx[iter], :), x)
            x = ( 1.0 - η ) * x -  η * x .* grad
            x_bar = (iter * x_bar + x) / (iter + 1.0)
        end

        if mod(iter, period) == 0
            λ = x_to_λ(x_bar)
            update_output!(output, iter÷period, iter/n, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    print_signal(io, λ)

    return output
end


function SLBOMD(n_epoch::Int64, n_rate::Int64)
    name = "SLBOMD"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    len_output = n_epoch * n_rate
    output = init_output(len_output)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    x_bar::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x_bar)

    n_iter::Int64 = n_epoch * n
    period::Int64 = n ÷ n_rate
    @timeit to "iteration" begin
        idx = sample(1:n, Weights(P), n_iter)
        η = sqrt( d * log( n_iter ) )
        η = η / ( sqrt( n_iter ) + η )
    end
 
    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin
            grad = - view(B, idx[iter], :) / dot(view(B, idx[iter], :), x)
            x_half = 1 ./ (1 ./ x + η * grad)
            x = log_barrier_projection(x_half, 1e-5)
            x_bar = (iter * x_bar + x) / (iter + 1.0)
        end

        if mod(iter, period) == 0
            λ = x_to_λ(x_bar)
            update_output!(output, iter÷period, iter/n, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    print_signal(io, λ)

    return output 
end


function relSGD(n_epoch::Int64, n_rate::Int64)
    name = "relSGD"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    len_output = n_epoch * n_rate
    output = init_output(len_output)
    to = TimerOutput()

    λ::Vector{Float64} = x_to_λ(ones(Float64, d) / d)
    λ_bar::Vector{Float64} = x_to_λ(ones(Float64, d) / d)
    grad::Vector{Float64} = zeros(Float64, d)

    n_iter::Int64 = n_epoch * n
    period::Int64 = n ÷ n_rate
    @timeit to "iteration" begin
        idx = rand(1:n, n_iter)

        # this learning rate is not explicitly stated in their paper
        # since their Corollary 4.6 has lots of typos.
        # I apply their theorem and use the best learning rates I found
        L = sqrt(2 * n_iter) / sqrt(d) / 5
    end
 
    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin

            a = view(A, idx[iter], :)
            grad = a - y[idx[iter]] * a / dot(a, λ)

            λ .= @. 1 / (1 / λ + grad / L)
            λ_bar .= @. (λ_bar * iter + λ) / (iter + 1.0)
        end

        if mod(iter, period) == 0
            update_output!(output, iter÷period, iter/n, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ_bar), normalized_l2(λ_bar, λ_true))
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    print_signal(io, λ_bar)

    return output 
end


function LB_SDA(n_epoch::Int64, n_rate::Int64)
    name = "1-sample LB-SDA"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    len_output = n_epoch * n_rate
    output = init_output(len_output)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    x_bar::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x_bar)
    ∑grad::Vector{Float64} = zeros(Float64, d)
    ∑dual_norm2::Float64 = 0.0
    
    n_iter::Int64 = n_epoch * n
    period::Int64 = n ÷ n_rate
    
    @timeit to "iteration" begin
        idx = sample(1:n, Weights(P), n_iter)
    end
 
    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin

            grad = - view(B, idx[iter], :) / dot(view(B, idx[iter], :), x)
            ∑grad += grad
            ∑dual_norm2 += dual_norm2(x, grad + α(x, grad) * ones(Float64, d))
            η = sqrt(d) / sqrt(4 * d + 1 + ∑dual_norm2)
            
            x = log_barrier_projection(1 ./ (η * ∑grad), 1e-5)

            x_bar = (iter * x_bar + x) / (iter + 1.0)
        end

        if mod(iter, period) == 0
            λ = x_to_λ(x_bar)
            update_output!(output, iter÷period, iter/n, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    print_signal(io, λ)

    return output 
end


function d_sample_LB_SDA(n_epoch::Int64, n_rate::Int64)
    name = "d-sample LB-SDA"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    len_output = n_epoch * n_rate
    output = init_output(len_output)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    x_bar::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x_bar)
    ∑grad::Vector{Float64} = zeros(Float64, d)
    ∑dual_norm2::Float64 = 0.0
    batch_size::Int64 = d
    
    n_iter::Int64 = n_epoch * (n ÷ batch_size)
    period::Int64 = (n ÷ batch_size) ÷ n_rate
    
    @timeit to "iteration" begin
        idx = sample(1:n, Weights(P), (batch_size, n_iter))
    end
 
    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin
            grad::Vector{Float64} = zeros(Float64, d)
            grad = vec( sum(- view(B, idx[:, iter], :) ./ (view(B, idx[:,iter], :) * x), dims = 1) / batch_size )
            ∑grad += grad

            ∑dual_norm2 += dual_norm2(x, grad + α(x, grad) * ones(Float64, d))
            η = sqrt(d) / sqrt(4 * d + 1 + ∑dual_norm2)
            
            x = log_barrier_projection(1 ./ (η * ∑grad), 1e-5)
            x_bar = (iter * x_bar + x) / (iter + 1.0)
        end

        if mod(iter, period) == 0
            λ = x_to_λ(x_bar)
            update_output!(output, iter÷period, iter/(n÷batch_size), TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    print_signal(io, λ)

    return output 
end


function SPDHG(n_epoch::Int64, n_rate::Int64)
    name = "SPDHG"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()
    θ = 1

    n_iter::Int64 = n_epoch * n
    period::Int64 = n ÷ n_rate
    
    λ = x_to_λ(ones(Float64, d)/d)
    z = zeros(Float64, n) # dual variable
    z_bar = zeros(Float64, n)

    maxnorm = 0
    for i = 1:n
        maxnorm = max(maxnorm, norm(view(A, i, :)))
    end
    τ = 0.99 / (n * maxnorm)
    
    @timeit to "iteration" begin
        idx = rand(1:n, n_iter)
    end

    direction = transpose(A) * z_bar

    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin
            λ = max.(λ - τ * direction, zeros(Float64, d)) # prox g
        
            # only update z[idx[iter]]
            a = view(A, idx[iter], :)
            σ = 0.99 / norm(a)

            prev_z_idx = z[idx[iter]]
            temp = z[idx[iter]] + σ*dot(a, λ)
            z[idx[iter]] = temp + 1 - sqrt( (temp-1)^2 + 4*σ*y[idx[iter]] ) # prox f^∗
            z[idx[iter]] /= 2

            prev_z_bar_idx = z_bar[idx[iter]]
            z_bar[idx[iter]] = z[idx[iter]] + θ * (z[idx[iter]] - prev_z_idx)
            
            direction += (z_bar[idx[iter]] - prev_z_bar_idx) * a
        end

        if mod(iter, period) == 0
            update_output!(output, iter÷period, iter/n, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    print_signal(io, λ)

    return output
end
