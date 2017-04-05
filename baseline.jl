using Knet
using Images,MAT

function main(args="")
    batchsize = 10
    xtrn,ytrn,xtst,ytst = loaddata("cifar10")
    dtrn = minibatch(xtrn,ytrn,batchsize)
    dtst = minibatch(xtst,ytst,batchsize)
    w,ms = init_weights("cifar10")
    prms = init_opt_param(w)
   
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,ms),:tst,accuracy(w,dtst,ms)))
    @time for i=1:1
        train(w,dtrn,ms,prms)
        report(epoch)
    end
end


function loaddata(dataset)
    path = "../dataset/cifar-10-batches-mat/"
    
    #path = "data/"
    if dataset == "cifar10"
        xtrn = Array{UInt8}(5*10000,3072)
        ytrn = Array{UInt8}(5*10000)
        xtst = Array{UInt8}(10000,3072)
        ytst = Array{UInt8}(10000)
        for i=1:5
            filename = string("data_batch_",i,".mat")
            data = matread(string(path,filename))
            xtrn[(i-1)*10000+1:i*10000,:] = data["data"]
            ytrn[(i-1)*10000+1:i*10000] = data["labels"]
        end
        filename = string("test_batch.mat")
        data = matread(string(path,filename))
        xtst = data["data"]
        ytst = data["labels"]
    end
    #Remember for ImageNet
    #separate{C<:Colorant}(img::AbstractArray{C,2}) is deprecated, use permuteddimsview(channelview(img), (2,3,1)) instead.
    return xtrn,ytrn,xtst,ytst
end


function minibatch(x,y,batchsize; atype=Array{Float32}, xrows=32, yrows=32, xscale=255)
    row2im(a) = permutedims(convert(atype, reshape(a, 32, 32, 3)), (2,1,3))
    n_data = size(x,1)
    n_data == length(y) || throw(DimensionMismatch())
    
    all_data = Array{Float32}(32,32,3,length(y))
    all_labels = zeros(Float32, (10,length(y)))
    
    for i=1:n_data
        all_data[:,:,:,i] = row2im(x[i,:])
        all_labels[y[i]+1,i] = 1        
    end
    data = Any[]
    for i=1:batchsize:n_data-batchsize+1
        push!(data,(all_data[:,:,:,i:i+batchsize-1], all_labels[:,i:i+batchsize-1]))
    end
    return data
end

#function generates random classes at the moment.
function predict(x,nclasses)
    nInstances = size(x,4)
    output = randn(nclasses, nInstances) * 0.1
end

function loss(w,x,ms,ygold;mode=1)
    ypred = resnet_cifar(w,x,ms;mode=mode)
    ynorm = logp(ypred,1)
    return -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function accuracy(w,dtst,ms, pred=resnet_cifar;mode=1)
    ncorrect = ninstance = nloss = 0
    for (x, ygold) in dtst
        ygold_gpu = convert(KnetArray{Float32},ygold)
        ypred = pred(w,x,ms;mode=mode)
        ynorm = logp(ypred,1)
        nloss += -sum(ygold_gpu .* ynorm)
        ncorrect += sum(ygold_gpu .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold,2)
    end
    return (ncorrect/ninstance, nloss/ninstance)
end

function train(w,dtrn,ms,prms)
    for (x,y) in dtrn
        x = convert(KnetArray{Float32}, x)
        y = convert(KnetArray{Float32}, y)
        g = lossgradient(w,x,ms,y;mode=0)
        for k=1:length(prms)
          update!(w[k],g[k],prms[k])
        end
    end
end

function resnet_cifar(w,x,ms;mode=1)
    x_gpu = convert(KnetArray{Float32},x)
    z = conv4(w[1],x_gpu; padding=1, stride=1) .+ w[2]
    z = batchnorm(w[3:4],z,ms; mode=mode)
    z = reslayerx5(w[5:43], z, ms; strides=[1,1,1,1], mode=mode)
    z = reslayerx5(w[44:82], z, ms; mode=mode)
    z = reslayerx5(w[83:121], z, ms; mode=mode)
    
    z  = pool(z; stride=1, window=8, mode=2)
    z = w[122] * mat(z) .+ w[123]
end

function init_weights(dataset;s=0.1)
    w = Any[]
    ms = Any[]
    if dataset == "cifar10"
        #block 1
        push!(w,randn(Float32, (3,3,3,64))*s) #1
        push!(w, zeros(Float32,1))
        push!(w,randn(Float32, 1)*s)
        push!(w, zeros(Float32,1)) #4
        push!(ms,zeros(Float32,1,1,64,1))
        push!(ms,zeros(Float32,1,1,64,1));#(0.00316)*ones
        #block 2
        for i=1:4
            if i== 1
                bottleneck_single_layer(w,ms,1,s) #5 to 7
            end
            #8 to 16, 17 to 25, 26 to 34, 35 to 43
            bottleneck_full_layer(w,ms,s)
        end
        #block 3
        for i=1:4
            if i== 1
                bottleneck_single_layer(w,ms,1,s) #44 to 46
            end
            #47 to 55, 56 to 64, 65 to 73, 74 to 82
            bottleneck_full_layer(w,ms,s)
        end
        #block 4
        for i=1:4
            if i== 1
                bottleneck_single_layer(w,ms,1,s) #83 to 85
            end
            #86 to 94, 95 to 103, 104 to 112, 113 to 121
            bottleneck_full_layer(w,ms,s)
        end
        #fc 121 to 123
        push!(w,randn(Float32, (10,64))*s)
        push!(w,zeros(Float32, (10,1))*s)
    end

    return map(x->convert(KnetArray{Float32},x),w),map(KnetArray,ms)
end

function bottleneck_single_layer(w,ms,kernel_size,s)
    push!(w,randn(Float32, (kernel_size,kernel_size,64,64))*s) #5
    push!(w,randn(Float32,1)*s)
    push!(w,zeros(Float32,1)) #7

    push!(ms,zeros(Float32,1,1,64,1))
    push!(ms,zeros(Float32,1,1,64,1))#(0.00316)*ones
end

function bottleneck_full_layer(w,ms,s)
    bottleneck_single_layer(w,ms,1,s)
    bottleneck_single_layer(w,ms,3,s)
    bottleneck_single_layer(w,ms,1,s)
end

function batchnorm(w, x, ms; mode=1, avg_decay=0.997,epsilon=1e-5)
    mu, sigma = nothing, nothing
    if mode == 0
        d = ndims(x) == 4 ? (1,2,4) : (2,)
        s = prod(size(x)[[d...]])
        mu = sum(x,d) / s
        xshift = x.-mu
        sigma_sq = (sum(xshift.*xshift, d)) / s # NOTE: x.^2 gives NAN FOR WHATEVER REASON

        mu_old = shift!(ms)
        sigma_old = shift!(ms)

        mu = avg_decay * mu_old + (1-avg_decay) * mu
        sigma_sq = avg_decay * (sigma_old.^2) + (1-avg_decay) *sigma_sq
        sigma = sqrt(sigma_sq + epsilon)

    elseif mode == 1
        mu = shift!(ms)
        sigma = shift!(ms)
    end

    # we need getval in backpropagation
    push!(ms, AutoGrad.getval(mu), AutoGrad.getval(sigma))
    xhat = (x.-mu) ./ sigma   
    return w[1] .* xhat .+ w[2]
end

function reslayerx0(w,x,ms; padding=0, stride=1, mode=1)
    b  = conv4(w[1],x; padding=padding, stride=stride)
    bx = batchnorm(w[2:3],b,ms; mode=mode)
end

function reslayerx1(w,x,ms; padding=0, stride=1, mode=1)
    relu(reslayerx0(w,x,ms; padding=padding, stride=stride, mode=mode))
end

function reslayerx2(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    ba = reslayerx1(w[1:3],x,ms; padding=pads[1], stride=strides[1], mode=mode)
    bb = reslayerx1(w[4:6],ba,ms; padding=pads[2], stride=strides[2], mode=mode)
    bc = reslayerx0(w[7:9],bb,ms; padding=pads[3], stride=strides[3], mode=mode)
end

function reslayerx3(w,x,ms; pads=[0,0,1,0], strides=[2,2,1,1], mode=1) # 12
    a = reslayerx0(w[1:3],x,ms; stride=strides[1], padding=pads[1], mode=mode)
    b = reslayerx2(w[4:12],x,ms; strides=strides[2:4], pads=pads[2:4], mode=mode)
    relu(a .+ b)
end

function reslayerx4(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    relu(x .+ reslayerx2(w,x,ms; pads=pads, strides=strides, mode=mode))
end

function reslayerx5(w,x,ms; strides=[2,2,1,1], mode=1)
    x = reslayerx3(w[1:12],x,ms; strides=strides, mode=mode)
    for k = 13:9:length(w)
        x = reslayerx4(w[k:k+8],x,ms; mode=mode)
    end
    return x
end

function init_opt_param(weights)
    prms = Any[]
    for k=1:length(weights)
        push!(prms, Momentum(;lr=0.001, gclip=0, gamma=0.9))
    end
    return prms
end
main()