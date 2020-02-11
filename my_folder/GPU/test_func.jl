# This file is created from paper https://arxiv.org/pdf/1712.03112.pdf
# The function provided from the paper doesn't run successfully, need to look into it


using CUDAnative
using CUDAdrv

function vadd(a,b,c)
    i = (blockIdx().x - 1) * blockDim().x  + threadIdx().x
    c[i] = a[i] + b[i]
    return
end

len = 100

a = rand(Float32, len)
b = rand(Float32, len)

d_a = CUDAdrv.Array(a)
d_b = CUDAdrv.Array(b)
d_c = similar(d_a)


@cuda (1,len) vadd(d_a, d_b, d_c)

@cuda threads=10 vadd(d_a, d_b, d_c)
c = Base.Array(d_c)

c
