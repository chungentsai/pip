using Arpack
using TimerOutputs
using ExponentialUtilities
using Printf
using LinearAlgebra


function EM(n_epoch::Int64, n_rate::Int64)
    name = "EM"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x)

    @inbounds for t = 1: n_epoch
        @timeit to "iteration" begin
            grad = ∇Kellyf(x)
            x = x .* -grad
        end

        λ = x_to_λ(x)
        update_output!(output, t, t, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
        print_output(io, output, t, VERBOSE)
    end

    print_signal(io, λ)

    return output
end


function BPG(n_epoch::Int64, n_rate::Int64)
    name = "BPG"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x)
    η::Float64 = 1

    @inbounds for t = 1: n_epoch
        @timeit to "iteration" begin
            grad = ∇Kellyf(x)
            x_half = 1 ./ (1 ./ x + η * grad)
            x = log_barrier_projection(x_half, 1e-5)
        end

        λ = x_to_λ(x)
        update_output!(output, t, t, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
        print_output(io, output, t, VERBOSE)
    end

    print_signal(io, λ)

    return output
end


function FW(n_epoch::Int64, n_rate::Int64)
    name = "Frank-Wolfe"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x)
    pmin::Float64 = minimum(P[P.>0])
    
    @inbounds for t = 1: n_epoch
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
        print_output(io, output, t, VERBOSE)
    end

    print_signal(io, λ)

    return output
end


function EMD(n_epoch::Int64, n_rate::Int64)
    name = "EMD"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    x::Vector{Float64} = ones(Float64, d) / d
    λ::Vector{Float64} = x_to_λ(x)
    α0::Float64 = 10
    r::Float64 = 0.5
    τ::Float64 = 0.8

    @inbounds for t = 1: n_epoch
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
        print_output(io, output, t, VERBOSE)
    end

    print_signal(io, λ)

    return output
end


function PDHG(n_epoch::Int64, n_rate::Int64)

    name = "PDHG"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    σ::Float64 = 0.99 / norm(A)
    τ::Float64 = 0.99 / norm(A)
    θ::Float64 = 1

    λ::Vector{Float64} = x_to_λ(ones(Float64, d))
    z::Vector{Float64} = zeros(Float64, n) # dual variable
    z_bar::Vector{Float64} = zeros(Float64, n)

    @inbounds for t = 1: n_epoch
        @timeit to "iteration" begin
            λ = max.(λ - τ * transpose(A) * z_bar, zeros(Float64, d)) # prox g
        
            prev_z = z
            temp = z + σ*A*λ
            z = temp .+ 1 - sqrt.( (temp.-1).^2 + 4*σ*y ) # prox f^∗
            z /= 2

            z_bar = z + θ * (z - prev_z)
        end

        update_output!(output, t, t, TimerOutputs.time(to["iteration"]) * 1e-9, f(λ), normalized_l2(λ, λ_true))
        print_output(io, output, t, VERBOSE)
    end

    print_signal(io, λ)

    return output
end