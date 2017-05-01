using Knet
using Images,MAT
using JLD

function main(args="")
    batchsize = 128
    block_size = 6
    lr=0.1
    l2reg = 0.0001
    aug = true

    xtrn,ytrn,xtst,ytst,mean_im = loaddata("cifar10")
    dtrn = minibatch(xtrn,ytrn,mean_im,batchsize)
    dtst = minibatch(xtst,ytst,mean_im,batchsize)
    w,ms = init_weights("cifar10",block_size)
    prms = init_opt_param(w,lr)
    #Knet.knetgc(); gc()
    println("batchsize= $(batchsize), blocksize= $(block_size), lr=$(lr), l2reg=$(l2reg), augmentation=$(aug)")
    report(epoch,ac1,ac2,n1)=println((:epoch,epoch,:trn,ac1,:tst,ac2,:norm,n1))
    
    println(accuracy(w,dtrn,ms,block_size))
    @time for epoch=1:200
        train(w,dtrn,ms,block_size,prms;l2=l2reg,aug=aug)
        ac1 = accuracy(w,dtrn,ms,block_size)
        ac2 = accuracy(w,dtst,ms,block_size)
        if epoch > 80
            change_lr(prms,lr/10)
        elseif epoch > 120
            change_lr(prms,lr/100)
        end
        if ac1[1] >= 0.9
            savename = string("weights_bottleneck_epoch",epoch,".jld")
            save_model(w,ms,savename)
        end
        n1 = squared_sum_weights(w)
        report(epoch,ac1,ac2,n1)
        if n1 == NaN32
            break
        end
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

        mean_im = load("models/mean_cifar.jld","mean_image")
    end
    #Remember for ImageNet
    #separate{C<:Colorant}(img::AbstractArray{C,2}) is deprecated, use permuteddimsview(channelview(img), (2,3,1)) instead.
    return xtrn,ytrn,xtst,ytst,mean_im
end


function minibatch(x,y,mean_im,batchsize; atype=Array{Float32}, xrows=32, yrows=32, xscale=255)
    row2im(a) = permutedims(convert(atype, reshape(a, 32, 32, 3))./xscale, (2,1,3))
    n_data = size(x,1)
    n_data == length(y) || throw(DimensionMismatch())
    
    all_data = Array{Float32}(32,32,3,length(y))
    all_labels = zeros(Float32, (10,length(y)))
    
    for i=1:n_data
        all_data[:,:,:,i] = row2im(x[i,:])
        all_labels[y[i]+1,i] = 1        
    end
    all_data = all_data .- (mean_im ./xscale)
    data = Any[]
    #n_data = n_data > 20000 ? 32: n_data #for small experiments
    n_batches = Int(floor(n_data / batchsize))
    for i=1:batchsize:n_batches*batchsize
        push!(data,(all_data[:,:,:,i:i+batchsize-1], all_labels[:,i:i+batchsize-1]))
    end
    if n_batches != n_data/batchsize
        push!(data,(all_data[:,:,:,n_batches*batchsize+1:n_data], all_labels[:,n_batches*batchsize+1:n_data]))
    end
    return data
end

#function generates random classes at the moment.
function predict(x,nclasses)
    nInstances = size(x,4)
    output = randn(nclasses, nInstances) * 0.1
end

function loss(w,x,ms,block_size,ygold;l2=0,mode=1)
    ypred = resnet_cifar(w,x,ms,block_size;mode=mode)
    ynorm = logp(ypred,1)
    J = -sum(ygold .* ynorm) / size(ygold,2)
    if l2 != 0
        J += l2 * squared_sum_weights(w)
    end
    return J
end

function squared_sum_weights(w)
    return sum(sumabs2(wi) for wi in w)
end

lossgradient = grad(loss)

function accuracy(w,dtst,ms, block_size,pred=resnet_cifar;mode=1)
    ncorrect = ninstance = nloss = 0
    for (x, ygold) in dtst
        ygold = convert(KnetArray{Float32},ygold)
        x = convert(KnetArray{Float32},x)
        ypred = pred(w,x,ms,block_size;mode=mode)
        ynorm = logp(ypred,1)
        nloss += -sum(ygold .* ynorm)
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold,2)
    end
    acc = ncorrect/ninstance
    J = nloss/ninstance

    return (acc, J)
end

function train(w,dtrn,ms,block_size,prms;l2=0,aug=true)
    for (x,y) in dtrn
        if aug
            x = augment_cifar10(x)
        end
        x = convert(KnetArray{Float32}, aug_x)
        y = convert(KnetArray{Float32}, y)
        g = lossgradient(w,x,ms,block_size,y;l2=l2,mode=0)
        for k=1:length(prms)
          update!(w[k],g[k],prms[k])
        end
    end
end

function resnet_cifar(w,x,ms,block_size;mode=1)
    z = conv4(w[1],x; padding=1, stride=1) .+ w[2]
    z = batchnorm(w[3:4],z,ms; mode=mode)
    
    fin1 = 5+3+block_size*9-1
    fin2 = fin1 + 3 + block_size*9
    fin3 = fin2 + 3 + block_size*9
    z = reslayerx5(w[5:fin1], z, ms; strides=[1,1,1,1], mode=mode)
    z = reslayerx5(w[fin1+1:fin2], z, ms; mode=mode)
    z = reslayerx5(w[fin2+1:fin3], z, ms; mode=mode)
    
    z  = pool(z; stride=1, window=8, mode=2)
    z = w[end-1] * mat(z) .+ w[end]
end

function init_weights(dataset,block_size;s=0.1)
    w = Any[]
    ms = Any[]
    filt_size = 16
    if dataset == "cifar10"
        #block 1
        push!(w,randn(Float32,3,3,3,filt_size)*sqrt(1/27)) #1
        push!(w,zeros(Float32,1,1,filt_size,1))
        push!(w,ones(Float32, 1,1,filt_size,1))
        push!(w,zeros(Float32,1,1,filt_size,1)) #4
        push!(ms,zeros(Float32,1,1,filt_size,1))
        push!(ms,ones(Float32,1,1,filt_size,1));
        #block 2
        for i=1:block_size
            if i== 1
                bottleneck_single_layer(w,ms,(1,1,filt_size,filt_size)) #5 to 7
                bottleneck_full_layer(w,ms,filt_size,filt_size,filt_size) #8 to 16
            else
                #17 to 25, 26 to 34, 35 to 43
                bottleneck_full_layer(w,ms,filt_size,filt_size,filt_size)
            end
        end
        input_size = filt_size
        #block 3
        for i=1:block_size
            if i== 1
                bottleneck_single_layer(w,ms,(1,1,input_size,input_size*2)) #44 to 46
                bottleneck_full_layer(w,ms,input_size,Int(input_size/2),input_size*2) # 47 to 55
            else
                # 56 to 64, 65 to 73, 74 to 82
                bottleneck_full_layer(w,ms,input_size*2,Int(input_size/2),input_size*2)
            end
        end
        input_size = input_size * 2
        #block 4
        for i=1:block_size
            if i== 1
                bottleneck_single_layer(w,ms,(1,1,input_size,input_size*2)) #83 to 85
                bottleneck_full_layer(w,ms,input_size,Int(input_size/2),input_size*2) # 86 to 94
            else
                # 95 to 103, 104 to 112, 113 to 121
                bottleneck_full_layer(w,ms,input_size*2,Int(input_size/2),input_size*2)
            end
        end
        input_size = input_size * 2
        #fc 121 to 123
        push!(w,randn(Float32, (10,input_size))*sqrt(1.0/input_size))
        push!(w,zeros(Float32, (10,1)))
    end

    return map(x->convert(KnetArray{Float32},x),w),map(KnetArray,ms)
end

function bottleneck_single_layer(w,ms,tensor_size)
    push!(w,generate_resnet_weights(tensor_size)) #5
    push!(w,ones(Float32,1,1,tensor_size[4],1))
    push!(w,zeros(Float32,1,1,tensor_size[4],1)) #7

    push!(ms,zeros(Float32,1,1,tensor_size[4],1))
    push!(ms,ones(Float32,1,1,tensor_size[4],1))
end

function bottleneck_full_layer(w,ms,channel_size,filter_size,out_size)
    bottleneck_single_layer(w,ms,(1,1,channel_size,filter_size))
    bottleneck_single_layer(w,ms,(3,3,filter_size,filter_size))
    bottleneck_single_layer(w,ms,(1,1,filter_size,out_size))
end

function generate_resnet_weights(tensor_size)
    n = tensor_size[1]*tensor_size[2]*tensor_size[3]
    w = randn(Float32, tensor_size) * sqrt(2/n)
    return w
end

function batchnorm(w, x, ms; mode=1, avg_decay=0.997,epsilon=1e-5)
    mu, sigma = nothing, nothing
    if mode == 0
        d = ndims(x) == 4 ? (1,2,4) : (2,)
        s = prod(size(x)[[d...]])
        mu = sum(x,d) / s
        xshift = x.-mu
        sigma_sq = (sum(xshift.*xshift, d)) / s # NOTE: x.^2 gives NAN FOR WHATEVER REASON

        xhat = (x.-mu) ./ sqrt(sigma_sq + epsilon)

        mu_old = shift!(ms)
        sigma_old = shift!(ms)

        mu = avg_decay * mu_old + (1-avg_decay) * mu
        sigma_sq = avg_decay * (sigma_old.*sigma_old) + (1-avg_decay) *sigma_sq
        sigma = sqrt(sigma_sq + epsilon)
        push!(ms, AutoGrad.getval(mu), AutoGrad.getval(sigma))

    elseif mode == 1
        mu = shift!(ms)
        sigma = shift!(ms)
        d = ndims(x) == 4 ? (1,2,4) : (2,)
        s = prod(size(x)[[d...]])
        xhat = (x.-mu) ./ (sqrt(s/(s-1))*sigma)
        # we need getval in backpropagation
        push!(ms, AutoGrad.getval(mu), AutoGrad.getval(sigma))
    end
   
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

function init_opt_param(weights,lr)
    prms = Any[]
    for k=1:length(weights)
        push!(prms, Momentum(;lr=lr, gamma=0.9))
    end
    return prms
end

function change_lr(prms,new_lr)
    for k=1:length(prms)
        prms[k].lr = new_lr
    end
end

function augment_cifar10(x)
    y = zeros(Float32,size(x))
    padded = zeros(size(x,1)+8,size(x,2)+8,size(x,3))
    h = size(x,1)
    w = size(x,2)
    c = size(x,3)
    b = size(x,4)
    hflip = rand([false,true],b)
    xi = rand(collect(1:9),b)
    xj = rand(collect(1:9),b)

    for i=1:size(y,4)
        if hflip[i]
            padded[4:3+h,4:3+w,:] = flipdim(x[:,:,:,i],2)
        else
            padded[4:3+h,4:3+w,:] = x[:,:,:,i]
        end
        y[:,:,:,i] = padded[xi[i]:xi[i]+31, xj[i]:xj[i]+31, :]
    end
    return y
end

function load_model(filename)
    model = load(string("models/",filename))
    w = model["w"]
    ms = model["ms"]
    w = map(x->convert(KnetArray,x),w)
    ms = map(KnetArray,ms)
    return w,ms
end

function save_model(w,ms,filename)
    weight = map(Array, w)
    moments = map(Array, ms)
    save(string("models/",filename),"w",weight,"ms",moments)
end

main()
