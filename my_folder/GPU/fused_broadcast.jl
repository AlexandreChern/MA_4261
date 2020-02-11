using CUDAnative
using CuArrays
using BenchmarkTools

X = CuArray(rand(42))

f(x) = 3x^2 + 5x + 2

Y = f.(2 .* X.^2 .+ 6 .* X.^3 .- sqrt.(X))

@benchmark Y = f.(2 .* X.^2 .+ 6 .* X.^3 .- sqrt.(X))
