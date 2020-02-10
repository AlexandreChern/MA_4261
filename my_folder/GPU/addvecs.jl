using CUDAnative


function addvecs!(c,a,b)
    N = length(a);
    @inbounds for i = 1:N
        c[i] = a[i] + b[i]
    end
end

let
    N = 1000
    b = rand(N)
    a = rand(N)
    c = similar(a)

    addvecs!(c,a,b)

    c0 = a + b
    @assert isapprox(c0,c)
end
