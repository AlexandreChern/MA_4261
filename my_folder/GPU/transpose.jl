using CUDAnative
using CuArrays
using CUDAdrv: synchronize, device
using LinearAlgebra
using Printf

function transpose_cpu!(b,a)
    N = size(a,1)
    for j = 1:N
        for i = 1:N
            b[j,i] = a[i,j]
        end
    end
end


function transpose_naive!(b,a)
    # Which threads are we in our blocks
    tidx = threadIdx().x
    tidy = ThreadIdx().y

    # Which block of threads are we in
    bidx = blockIdx().x
    bidy = blockIdx().y

    # What is the size of the thread block
    dimx = blockDim().x
    dimy = blockDim().y

    # What index am I in the global thread space
    i = tidx + dimx * (bidx - 1)
    j = tidy + dimy * (bidy - 1)

    if i <= N && j <= N
        b[j,i] = a[i,j]
    end
    return nothing
end


function main(; N=1024, FT=Float32, tile_dim=32)
    println(device())
    memsize = N * N * sizeof(FT) / 1024^3
    @printf("Float type:        %s\n", FT)
    @printf("Matrix size:       %4d x %4d\n", N, N)
    @printf("Memory required:   %f GiB\n", memsize)

    # Host Arrays
    a = rand(FT,N,N)
    b = similar(a)
    transpose_cpu!(b,a)
    @assert b == a'

    # Device Arrays
    d_a = CuArray(a)
    d_b = similar(d_a)

    nblocks = (cld(N,tile_dim),cld(N, tile_dim))
    @cuda threads=(tile_dim, tile_dim) blocks=nblocks transpose_naive!(d_b,d_a)

end
