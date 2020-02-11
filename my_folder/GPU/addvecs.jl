using CUDAnative


function addvecs!(c,a,b)
    N = length(a);
    @inbounds for i = 1:N
        c[i] = a[i] + b[i]
    end
end


function fake_knl_addvecs(c,a,b,num_threads_block, num_blocks)
    for bid = 1:num_blocks
        for tid = 1:num_threads_per_block
            i = tid + dim *(bid - 1) # unique global num_threads_block
            c[i] = a[i] + b[i]
        end
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
    fake_knl_addvecs(c,a,b,4,5)

    c .= 0;
    threads = 64

    blocks =
end
