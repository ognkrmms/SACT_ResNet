using Images,MAT

function main(args="")
    batchsize = 10
    xtrn,ytrn,xtst,ytst = loaddata("cifar10")
    dtrn = minibatch(xtrn,ytrn,batchsize)
    dtst = minibatch(xtst,ytst,batchsize)
    println(string(size(dtrn[1][1])))

end


function loaddata(dataset)
    path = "/media/ogn/Data/datasets/cifar-10-batches-mat/"
    
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
    

    #img[:,:,1] = reshape(img_data[1:1024],32,32)'
    #img[:,:,2] = reshape(img_data[1025:2048],32,32)'
    #img[:,:,3] = reshape(img_data[2049:3072],32,32)'
    
    

    #separate{C<:Colorant}(img::AbstractArray{C,2}) is deprecated, use permuteddimsview(channelview(img), (2,3,1)) instead.
    return xtrn,ytrn,xtst,ytst
end


function minibatch(x,y,batchsize; atype=Array{Float32}, xrows=32, yrows=32, xscale=255)
    row2im(a) = permutedims(convert(atype, reshape(a./xscale, 32, 32, 3)), (2,1,3))
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

main()