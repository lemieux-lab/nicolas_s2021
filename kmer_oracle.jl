using CUDA
using Flux
using Gadfly
using ProgressBars
using Statistics
using HDF5
using Cairo
using Dates

include("kmer_utils.jl")
CUDA.allowscalar(false)

function evaluate_loss(loss::Function, kmers::Array{Bool, 2}, counts::Array{Int32, 1}; device::Function=cpu)
    dl = Flux.Data.DataLoader((kmers, counts), batchsize=1000)
    total = Array{Float32, 1}()
    for (i, e) in dl
        push!(total, loss(device(i), device(e)))
    end
    return mean(total)
end

function evaluate_results(network, testing_set::Array{Bool, 2}; device::Function=cpu)
    o_test = Array{Float32, 1}()
    dl = Flux.Data.DataLoader(testing_set, batchsize=1000)
    for i in dl
        push!(o_test, cpu(network(device(i)))...)
    end
    return o_test
end

function neural_network(k::Int64=31)
    return Chain(
            Dense(4*k, 4*k÷2, x->σ.(x)),
            Dense(4*k÷2, 1, identity)
            )
end

function train_and_plot(; plot_every::Int64=10, use_log_count::Bool=false, device::Function=cpu)

    # @time raw_counts = DataFrame(JDF.load("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_DataFrame_min-5_300k.jld2"))
    h5_file = h5open("/home/golem/rpool/scratch/jacquinn/data/13H107-k31.h5", "r")
    all_kmers = read(h5_file["kmers/13H107-k31_min-5_ALL"])
    all_counts = read(h5_file["counts/13H107-k31_min-5_ALL"])
    close(h5_file)
    i_train, e_train, i_test, e_test = split_kmer_data(all_kmers, all_counts, 75)
    # println(mean(e_train))

    network = neural_network() |> device
    # println(network[1])
    # println(network[2].W)

    function L2_penalty() # = sum(sum.(abs2, network.W)) + sum(sum.(abs2, network.b))
        penalty = 0
        for layer in network
            penalty += sum(layer.W) + sum(layer.b)
        end
        return penalty
    end

    # println(L2_penalty())
    loss(input, expected) = Flux.Losses.mse(network(input), expected) + L2_penalty()
    dl_train = Flux.Data.DataLoader((i_train, e_train), batchsize=1000, shuffle=true)
    opt = Flux.ADAM()
    ps = Flux.params(network)

    # Empty lists that will contain loss values over iterations (for plotting)
    losses = Float32[]
    testing_losses = Float32[]
    
    #=
    These 3 are here so I can quickly see what the newtork outputs for a 113k count kmer,
    a 14 counts kmer and a kmer that's not in the dataset
    =#
    prepared_test_kmers= gpu([onehot_kmer("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                          onehot_kmer("ACTGCTTATATATGTATCCTTCAACAATATA"),
                          onehot_kmer("ACATGTAACAGTAGATACACAAGAATACAAC")])

    # Main training loop starts
    epoch = 0
    start_time = now()
    while true
        epoch_start_time = now()
        epoch += 1

        # Evaluates loss on training & testing sets
        push!(losses, evaluate_loss(loss, i_train, e_train, device=device))
        push!(testing_losses, evaluate_loss(loss, i_test, e_test, device=device))

        # Iterates over the dataset to train the NN
        for (i, e) in ProgressBar(dl_train)
            if use_log_count  # BAD
                e = log.(e)
            end
            gs = gradient(ps) do
                loss(device(i), device(e))
            end
            Flux.update!(opt, ps, gs)
        end

        # Tests for the 3 prepped kmers
        for prep in prepared_test_kmers
            println(network(prep))
        end

        # Plots loss tracking data every X epoch
        if epoch % plot_every == 0
            results = plot(layer(x=1:length(losses), y=losses, Geom.line),
            layer(x=1:length(testing_losses), y=testing_losses, Geom.line, Theme(default_color=color("orange"))),
            Guide.xlabel("Iter"), Guide.ylabel("loss"))
            draw(PNG("/u/jacquinn/graphs/L2_and_log_counts/results_13H107_min-5_ALL_epoch-$(epoch).png", 50cm, 50cm), results)
        end

        println("current epoch: $(epoch)\nepoch runtime: $(now()-epoch_start_time)\nglobal runtime: $(now()-start_time)\n\n\n")
        flush(stdout)
    end

    # o_test = evaluate_results(network, i_test, device=device)

    # Making a plot to visualise results
    
    # results = vstack(
    # plot(layer(x=1:length(losses), y=losses, Geom.line),
    #     layer(x=1:length(testing_losses), y=testing_losses, Geom.line, Theme(default_color=color("orange"))),
    #     Guide.xlabel("Iter"), Guide.ylabel("loss")),
    # plot(x=hcat(e_test...), y=hcat(o_test...), Geom.point,
    #     Guide.xlabel("expected"), Guide.ylabel("obtained"))
    # )

end

train_and_plot(plot_every=5, device=gpu, use_log_count=true)
