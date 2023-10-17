using Distributions
using StatsBase
using LinearAlgebra
using ImagePhantoms


function true_parameters(w::Int64)
    d = w*w
    image = ImagePhantoms.shepp_logan(w) * 1000
    λ::Vector{Float64} = reshape(image, d)

    #λ::Vector{Float64} = ones(Float64, d) * 10000
    #λ[1*d÷5 : 1*d÷4] .= 10000
    #λ[4*d÷9 : 5*d÷9] .= 10000
    #λ[3*d÷4 : 4*d÷5] .= 10000
    return λ
end


function sensing_matrix(n::Int64, d::Int64, p::Float64)
    #A = ones(Float64, n, d) / n

    A = sample([0.0, 1.0], Weights([1-p, p]), (n, d))
    A = A ./ sum(A, dims=1)

    #λminus = -((1-p)/p)^0.5
    #λplus = (p/(1-p))^0.5
    #@inbounds for i in 1:n
    #    @inbounds for j in 1:d
    #        if rand() <= p
    #            A[i, j] = λminus
    #        else
    #            A[i, j] = λplus
    #        end
    #    end
    #end
    #A = ((p*(1-p))^0.5 * A + (1-p) * ones(Float64, n, d)) / n
    #A = max.(A, zeros(Float64, n, d))
    
    return A
end


function generate_observations(A::Array{Float64, 2}, λ_true::Array{Float64, 1})
    n = size(A)[1]
    λ = A * λ_true

    y = zeros(Int64, n)
    for i = 1:n
        y[i] = rand(Poisson(λ[i]))
    end

    return y
end


function x_to_λ(A_csum::Array{Float64, 1}, Y::Int64, x::Array{Float64, 1})
    λ::Vector{Float64} = Y .* x ./ A_csum
    return λ
end


function f(A::Array{Float64, 2}, y::Array{Int64, 1}, λ::Array{Float64, 1})
    ip = A * λ
    return sum(ip .- y .* log.(ip))
end


function ∇f(A::Array{Float64, 2}, y::Array{Int64, 1}, λ::Array{Float64, 1})
    return vec(- sum(A - y .* A ./ (A * λ), dims=1))
end


function Kellyf(B::Array{Float64, 2}, P::Array{Float64, 1}, x::Array{Float64, 1})
    return -sum(P .* log.(B * x))
end


function ∇Kellyf(B::Array{Float64, 2}, P::Array{Float64, 1}, x::Array{Float64, 1})
    return vec(- sum(P .* B ./ (B * x), dims=1))
end


function ∇2Kellyf(B::Array{Float64, 2}, P::Array{Float64, 1}, x::Array{Float64, 1})
    n, d = size(B)
    hess = zeros(Float64, d, d)

    temp = B ./ (B * x)
    for i = 1:n
        hess += P[i] * temp[i, :] * temp[i, :]'
    end
    return hess
end


function log_barrier_projection(
    u::Array{Float64, 1},
    ε::Float64
    )
    # compute argmin_{x∈Δ} D_h(x,u) where h(x)=∑_{i=1}^d -log(x_i)
    # minimize ϕ(θ) = θ - ∑_i log(θ + u_i^{-1})

    uinv::Vector{Float64} = 1 ./ u
    θ::Float64 = 1 - minimum(uinv)
    a::Vector{Float64} = @. 1 / (uinv + θ)
    ∇::Float64 = 1 - sum(a)
    ∇2::Float64 = a ⋅ a
    λt::Float64 = abs(∇) / sqrt(∇2)

    while λt > ε
        a .= @. 1 / (uinv + θ)
        ∇ = 1 - norm(a, 1)
        ∇2 =  a ⋅ a
        θ -= ∇ / ∇2
        λt = abs(∇) / sqrt(∇2)
    end

    return a
end


function α(x::Array{Float64, 1}, v::Array{Float64, 1})
    return - sum(x.^2 .* v) / sum(x.^2)
end


function dual_norm2(x::Array{Float64, 1}, v::Array{Float64, 1})
    return sum((x.*v).^2)
end


function normalized_l2(λ::Array{Float64, 1}, λ_true::Array{Float64, 1})
    return norm(λ - λ_true) / norm(λ_true)
end
