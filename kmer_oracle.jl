using CUDA
using Flux
using Gadfly
using ProgressBars
using Statistics
using HDF5
using Cairo

include("kmer_utils.jl")
CUDA.allowscalar(false)

function evaluate_loss(loss::Function, kmers::Array{Bool, 2}, counts::Array{Int32, 1}; device::Function=cpu)
    dl = Flux.Data.DataLoader((kmers, counts), batchsize=1000)
    total = 0
    for (i, e) in ProgressBar(dl)
        total+=sum(loss(device(i), device(e)))
    end
    return total
end

function evaluate_results(network, testing_set::Array{Bool, 2}; device::Function=cpu)
    o_test = Array{Float64, 1}[]
    dl = Flux.Data.DataLoader(kmers, batchsize=1000)
    for i in ProgressBar(dl)
        push!(o_test, network(device(i)))
    end
    return o_test
end

function train_and_plot()
    # device = cpu
    device = gpu
    k = 31

    function neural_network()
        return Chain(
                Dense(4*k, 25, x->σ.(x)),
                Dense(25, 25, x->σ.(x)),
                Dense(25, 1, identity)
                )
    end

    # @time raw_counts = DataFrame(JDF.load("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_DataFrame_min-5_300k.jld2"))
    @time h5_file = h5open("/home/golem/rpool/scratch/jacquinn/data/encoded_kmer_counts.h5", "r")
    @time all_kmers = read(h5_file["kmers/kmers"])
    @time all_counts = read(h5_file["counts/counts"])
    close(h5_file)
    @time i_train, e_train, i_test, e_test = split_kmer_data(all_kmers, all_counts, 75)
    # @time training_df, testing_df = split_kmer_df(raw_counts, 75)

    # @time i_train, e_train = create_flux_sets(training_df)
    # @time i_test, e_test = create_flux_sets(testing_df)
    # @time i_train, e_train = onehot_kmer.(training_df[!, "kmers"]), training_df[!, "counts"]
    # @time i_test, e_test = onehot_kmer.(testing_df[!, "kmers"]), testing_df[!, "counts"]
    # @time i_train, e_train = parse_saved_onehot_kmers(training_df[!, "kmers"]), training_df[!, "counts"]
    # @time i_test, e_test = parse_saved_onehot_kmers(testing_df[!, "kmers"]), testing_df[!, "counts"]

    # i_train |> device
    # e_train |> device
    # i_test |> device
    # e_test |> device

    network = neural_network() |> device
    loss(input, expected) = Flux.Losses.mse(network(input), expected)
    # function loss(input, expected)
    #     println("input:", input)
    #     l = Flux.Losses.mse(network(input), expected)
    #     return l
    # end
    dl_train = Flux.Data.DataLoader((i_train, e_train), batchsize=1000, shuffle=true)
    opt = Flux.ADAM()
    ps = Flux.params(network)

    # Empty lists that will contain loss values over iterations (for plotting)
    losses = Int64[]
    testing_losses = Int64[]

    # println(loss.(eachcol(i_train), eachcol(e_train)))

    # Main training loop
    epochs = 2
    cur_loss = "N/A"
    iter = ProgressBar(1:epochs)
    for i in iter
        set_postfix(iter, Loss=cur_loss)
        for (i, e) in ProgressBar(dl_train)
            gs = gradient(params(network)) do
                loss(device(i), device(e))
            end 
            Flux.Optimise.update!(opt, ps, gs)
        end
        cur_loss = evaluate_loss(loss, i_train, e_train, device=device)
        push!(losses, cur_loss)
        push!(testing_losses, evaluate_loss(loss, i_test, e_test, device=device))
    end

    o_test = evaluate_results(network, i_test, device=device)

    # Making a plot to visualise results
    set_default_plot_size(30cm, 30cm)
    results = vstack(
    plot(layer(x=1:length(losses), y=losses, Geom.line),
        layer(x=1:length(testing_losses), y=testing_losses, Geom.line, Theme(default_color=color("orange"))),
        Guide.xlabel("Iter"), Guide.ylabel("loss")),
    plot(x=hcat(e_test...), y=hcat(o_test...), Geom.point,
        Guide.xlabel("expected"), Guide.ylabel("obtained"))
    )
    draw(PNG("/u/jacquinn/graphs/results_13H107_min-5_ALL_epoch-2.png", 50cm, 50cm), results)

end

@time train_and_plot()
