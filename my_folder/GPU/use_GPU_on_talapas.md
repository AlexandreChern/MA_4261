## Section 1 Configuration on the servers

##### 1. Request a gpu node

`srun --pty --account=erickson  --gres=gpu:1 --mem=8G --time=240 --partition=testgpu bash`

This can be stored into a shell script and reused later 

##### 2. Load cuda library this is very important !
`module load cuda/10.1` 

 This need to be executed before launching Julia and installing Julia packages, older version like cuda/9.0 might work but it would give warnings about cudnn library not compatible with CuArrays package

##### 3. Create a directory in your home directory to store the julia-1.3.1 version release
`mkdir julia1.3;
 cd julia1.3`

##### 4. Download latest julia1.3 from official website into julia1.3 directory
`wget https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.1-linux-x86\_64.tar.gz`

Note: `\_ can be replaced with simply _`

##### 5. Unzip it 
`tar -xf julia-1.3.1-linux-x86\_64.tar.gz ` 

##### 6. Change your bash setting to use this julia1.3 as default

`vim ~/.bashrc # I would suggest using .bashrc instead of .bash_profile,`

Add this line to the end of the file

`alias julia=~/julia1.3/julia-1.3.1/bin/julia # use this julia as the default version`

And then source this file

`source ~/.bashrc`



## Section 2 Install Julia packages

##### 1. Launch Julia

`julia`     you will see the version  number is now 1.3.1

##### 2.  Enter Package Manager Mode

`]`

##### 3.  Install related packages

```add CUDAdrv
add CUDAdrv # usually this is the dependency of CUDAnative.jl, but just to make sure
add CUDAnative
add CuArrays
```





##### 4. Test packages

`using CUDAnative `  You might need to use `using CUDAdrv`  and specify CUDA device with 

``` 
using CUDAnative
using CuArrays

a = cu(randn(3,3)) # will give you a randomized cuda array
```



Sometimes when you are switched to a node with different GPU, you might need to precompile CUDAnative again to make sure it works.

Sometimes you might need to use CUDAdrv and specify the device that you are using before using CUDAnative. This often will be prompted in warning message that CUDAnative or errors like this one `ERROR: CUDA error: invalid device context (code 201, ERROR_INVALID_CONTEXT)`. What you can do is run this code before `using CUDAnative`

```
using CUDAdrv
gpuid = 0
dev = CuDevice(gpuid) # Or the ID of the GPU you want, if you have many of them
ctx = CuContext(dev)
```

From my previous experience, I would always add these lines in the script even many times it worked without these lines.



With the steps above, I got my julia1.3 working with both K80 (on @120 node) and v100 (on @245 node).

It is frustrating that when you are assigned a different node, usually you will receive errors when trying to use CUDAnative package and needs to rebuild. It's still not clear to me what I need to do exactly, although most of the times what I did worked.  (rebuilding CUDAnative.jl, uisng CUDAdrv, etc)

