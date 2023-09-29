using Arpack
using TimerOutputs
using ExponentialUtilities
using Printf
using LinearAlgebra


function EM(
    n_epoch::Int64, 
    n_rate::Int64, 
    verbose
    )
    # Thomas M. Cover, An algorithm for maximizing expected log investment return (https://ieeexplore.ieee.org/abstract/document/1056869?casa_token=y0cA70bABs0AAAAA:zpsP7RfwrwwlhCno5liSw3OU1Fha6yFDdJk9UDvLCSAMnC0teguSlZx31ILbUc1SoVk0bep2yw)
    name = "EM"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x)

    @inbounds for t = 1: n_epoch
        # update iterate
        @timeit to "iteration" begin
            grad = ∇Kellyf(x)
            x = x .* -grad
        end

        λ = x_to_λ(x)
        update_output!(output, t, t, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
        print_output(io, output, t, verbose)
    end

    print_signal(io, λ)

    return output
end


function BPG(
    n_epoch::Int64, 
    n_rate::Int64, 
    verbose
    )
    # Heinz H. Bauschke, Jérôme Bolte, Marc Teboulle, A Descent Lemma Beyond Lipschitz Gradient Continuity: First-Order Methods Revisited and Applications, 2017 (https://pubsonline.informs.org/doi/abs/10.1287/moor.2016.0817)

    name = "BPG"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x)
    η = 1

    @inbounds for t = 1: n_epoch
        # update iterate
        @timeit to "iteration" begin        
            # update
            grad = ∇Kellyf(x)
            x_half = 1 ./ (1 ./ x + η * grad)
            x = log_barrier_projection(x_half, 1e-5)
        end

        λ = x_to_λ(x)
        update_output!(output, t, t, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
        print_output(io, output, t, verbose)
    end

    print_signal(io, λ)

    return output
end


function FW(
    n_epoch::Int64, 
    n_rate::Int64, 
    verbose
    )
    # Renbo Zhao and Robert M. Freund, Analysis of the Frank–Wolfe method for convex composite optimization involving a logarithmically-homogeneous barrier, 2023 (https://link.springer.com/article/10.1007/s10107-022-01820-9)
    name = "Frank-Wolfe"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x)
    pmin = minimum(P[P.>0])
    
    @inbounds for t = 1: n_epoch

        # update iterate
        @timeit to "iteration" begin
            invBx = 1 ./ (B * x) 
            grad = P .* -invBx / pmin # ∇_y f(Bx), n×1

            v = zeros(Float64, d)
            v[argmin(transpose(B) * grad)] = 1
            direction = v - x

            Bdir = B * -direction
            G = dot(grad, Bdir)
            D = ( sum( Bdir.^2 .* P .* invBx.^2 ) / pmin )^0.5
            η = min(G / (D*(G+D)), 1)

            x += η * direction
        end

        λ = x_to_λ(x)
        update_output!(output, t, t, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
        print_output(io, output, t, verbose)
    end

    print_signal(io, λ)

    return output
end


function EMD(
    n_epoch::Int64, 
    n_rate::Int64, 
    verbose
    )

    name = "EMD"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x)
    α0 = 10
    r = 0.5
    τ = 0.8

    @inbounds for t = 1: n_epoch
        # update iterate
        @timeit to "iteration" begin
            grad = ∇Kellyf(x)
            
            # Armijo line search
            α = α0
            xα = x .* exp.(-α * grad)
            xα /= sum(xα)
            while τ*dot(grad, xα-x) + Kellyf(x) < Kellyf(xα)
                α *= r
                xα = x .* exp.(-α * grad)
                xα /= sum(xα)
            end
            x = xα
        end

        λ = x_to_λ(x)
        update_output!(output, t, t, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
        print_output(io, output, t, verbose)
    end

    print_signal(io, λ)

    return output
end


function PDHG(
    n_epoch::Int64, 
    n_rate::Int64,
    verbose
    )

    name = "PDHG"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    σ = 0.99 / norm(A)
    τ = 0.99 / norm(A)
    θ = 1

    λ::Vector{Float64} = x_to_λ(ones(Float64, d))
    z::Vector{Float64} = zeros(Float64, n) # dual variable
    z_bar::Vector{Float64} = zeros(Float64, n)

    @inbounds for t = 1: n_epoch
        # update iterate
        @timeit to "iteration" begin
            λ = max.(λ - τ * transpose(A) * z_bar, zeros(Float64, d)) # prox g
        
            prev_z = z
            temp = z + σ*A*λ
            z = temp .+ 1 - sqrt.( (temp.-1).^2 + 4*σ*y ) # prox f^∗
            z /= 2

            z_bar = z + θ * (z - prev_z)
        end

        update_output!(output, t, t, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
        print_output(io, output, t, verbose)
    end

    print_signal(io, λ)

    return output
end