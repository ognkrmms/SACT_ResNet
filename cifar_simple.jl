using Knet
using Images,MAT
using JLD

function main(args="")
    batchsize = 128
    lr=0.1
    l2reg = 0.00001
    aug = true

    xtrn,ytrn,xtst,ytst,mean_im = loaddata("cifar10")
    dtrn = minibatch(xtrn,ytrn,mean_im,batchsize)
    dtst = minibatch(xtst,ytst,mean_im,batchsize)
    w,ms = init_weights("cifar10")
    #w,ms = load_model("weights_keras_epoch128.jld")
    prms = init_opt_param(w,lr)
    #prms = init_opt_param_adam(w)
    #Knet.knetgc(); gc()
    
    println("batchsize= $(batchsize), lr=$(lr), l2reg=$(l2reg),aug=$(aug)")
    report(epoch,ac1,ac2,n1)=println((:epoch,epoch,:trn,ac1,:tst,ac2,:norm,n1))    
    println((:epoch,0,:trn,accuracy(w,dtrn,ms),:tst,accuracy(w,dtst,ms),:wnorm,squared_sum_weights(w)))
    epoch=1
    @time for epoch=1:300
        train(w,dtrn,ms,prms;l2=l2reg,aug=aug)
        ac1 = accuracy(w,dtrn,ms)
        ac2 = accuracy(w,dtst,ms)
        if epoch > 80
            change_lr(prms,lr/10)
        elseif epoch > 120
            change_lr(prms,lr/100)
        end

        if ac1[1] >=0.9
            savename = string("weights_keras_epoch",epoch,".jld")
            save_model(w,ms,savename)
        end
        
        n1 = squared_sum_weights(w)
        report(epoch,ac1,ac2,n1)

        if n1 == NaN32 || ac1[1] == 1.0
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
    all_data = all_data .- (mean_im./xscale)
    data = Any[]
    #n_data = n_data > 20000 ? 64: n_data #for small experiments
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

function loss(w,x,ms,ygold;l2=0, mode=1)
    ypred = resnet_cifar(w,x,ms;mode=mode)
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

function accuracy(w,dtst,ms,pred=resnet_cifar;mode=1)
    ncorrect = ninstance = nloss = 0
    for (x, ygold) in dtst
        ygold = convert(KnetArray{Float32},ygold)
        x = convert(KnetArray{Float32},x)
        ypred = pred(w,x,ms;mode=mode)
        ynorm = logp(ypred,1)      
        nloss += -sum(ygold .* ynorm)
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold,2)
    end
    return (ncorrect/ninstance, nloss/ninstance)
end

function train(w,dtrn,ms,prms;l2=0,aug=true)
    for (x,y) in dtrn
        if aug
            x = augment_cifar10(x)
        end
        x = convert(KnetArray{Float32}, x)
        y = convert(KnetArray{Float32}, y)
        g = lossgradient(w,x,ms,y;l2=l2,mode=0)
        for k=1:length(prms)
          update!(w[k],g[k],prms[k])
        end
    end
end

function resnet_cifar(w,x,ms;mode=1)
    z = conv4(w[1],x; padding=1, stride=2)
    z = batchnorm(w[2:3],z,ms,1; mode=mode)

    for i=1:10
        z = reslayerx4(w[4+(i-1)*9:4+i*9-1],z,ms,3+(i-1)*6;mode=mode)
    end
    
    z  = pool(z; stride=1, window=16, mode=2)
    z = w[end-1] * mat(z) .+ w[end]
    
end

function init_weights(dataset;s=0.01)
    w = Any[]
    ms = Any[]
    filt_size = 128
    if dataset == "cifar10"
        #block 1
        push!(w,randn(Float32,3,3,3,filt_size)*sqrt(1.0/27)) #1
        push!(w,ones(Float32, 1,1,filt_size,1))
        push!(w,zeros(Float32,1,1,filt_size,1)) #3
        push!(ms,zeros(Float32,1,1,filt_size,1))
        push!(ms,ones(Float32,1,1,filt_size,1));
        #block 2,3,4
        for i=1:10
            bottleneck_full_layer(w,ms,filt_size,Int(filt_size/4),filt_size)
        end
    end
    push!(w,randn(Float32,10,filt_size)*sqrt(1.0/filt_size))
    push!(w,zeros(Float32,10,1))
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

function batchnorm(w, x, ms,idx; mode=1, avg_decay=0.997,epsilon=1e-5)
    if mode == 0
        d = ndims(x) == 4 ? (1,2,4) : (2,)
        s = prod(size(x)[[d...]])
        mu = sum(x,d) / s
        xshift = x.-mu
        sigma_sq = (sum(xshift.*xshift, d)) / s # NOTE: x.^2 gives NAN FOR WHATEVER REASON

        xhat = (x.-mu) ./ sqrt(sigma_sq + epsilon)

        mu = avg_decay * ms[idx] + (1-avg_decay) * mu
        sigma_sq = avg_decay * (ms[idx+1].*ms[idx+1]) + (1-avg_decay) *sigma_sq
        sigma = sqrt(sigma_sq + epsilon)
        ms[idx] = AutoGrad.getval(mu)
        ms[idx+1] = AutoGrad.getval(sigma)
    elseif mode == 1
        d = ndims(x) == 4 ? (1,2,4) : (2,)
        s = prod(size(x)[[d...]])
        xhat = (x.-ms[idx]) ./ (sqrt(s/(s-1))*ms[idx+1])
    end    
    
    return w[1] .* xhat .+ w[2]
end

function reslayerx0(w,x,ms,ms_idx; padding=0, stride=1, mode=1)
    b  = conv4(w[1],x; padding=padding, stride=stride)
    bx = batchnorm(w[2:3],b,ms,ms_idx; mode=mode)
end

function reslayerx1(w,x,ms,ms_idx; padding=0, stride=1, mode=1)
    relu(reslayerx0(w,x,ms,ms_idx; padding=padding, stride=stride, mode=mode))
end

function reslayerx2(w,x,ms,ms_idx; pads=[0,1,0], strides=[1,1,1], mode=1)
    ba = reslayerx1(w[1:3],x,ms,ms_idx; padding=pads[1], stride=strides[1], mode=mode)
    bb = reslayerx1(w[4:6],ba,ms,ms_idx+2; padding=pads[2], stride=strides[2], mode=mode)
    bc = reslayerx0(w[7:9],bb,ms,ms_idx+4; padding=pads[3], stride=strides[3], mode=mode)
end

function reslayerx3(w,x,ms,ms_idx; pads=[0,0,1,0], strides=[2,2,1,1], mode=1) # 12
    a = reslayerx0(w[1:3],x,ms,ms_idx; stride=strides[1], padding=pads[1], mode=mode)
    b = reslayerx2(w[4:12],x,ms,ms_idx+2; strides=strides[2:4], pads=pads[2:4], mode=mode)
    relu(a .+ b)
end

function reslayerx4(w,x,ms,ms_idx; pads=[0,1,0], strides=[1,1,1], mode=1)
    relu(x .+ reslayerx2(w,x,ms,ms_idx; pads=pads, strides=strides, mode=mode))
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

function init_opt_param_adam(weights)
    prms = Any[]
    for k=1:length(weights)
        push!(prms, Adam())
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
