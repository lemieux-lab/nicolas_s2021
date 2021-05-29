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
    total = 0
    for (i, e) in dl
        total+=sum(loss(device(i), device(e)))
    end
    return total
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
            Dense(4*k, 4*k, x->σ.(x)),
            Dense(4*k, 100, x->σ.(x)),
            Dense(100, 50, x->σ.(x)),
            Dense(50, 25, x->σ.(x)),
            Dense(25, 1, identity)
            )
end

function train_and_plot(; plot_every::Int64=10, device::Function=cpu)

    # @time raw_counts = DataFrame(JDF.load("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_DataFrame_min-5_300k.jld2"))
    h5_file = h5open("/home/golem/rpool/scratch/jacquinn/data/encoded_kmer_counts.h5", "r")
    all_kmers = read(h5_file["kmers/kmers"])
    all_counts = read(h5_file["counts/counts"])
    close(h5_file)
    i_train, e_train, i_test, e_test = split_kmer_data(all_kmers, all_counts, 75)
    set_default_plot_size(30cm, 30cm)
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
    epoch = 0
    cur_loss = "N/A"
    start_time = now()
    while true
        epoch_start_time = now()
        epoch += 1
        for (i, e) in dl_train
            gs = gradient(params(network)) do
                loss(device(i), device(e))
            end 
            Flux.Optimise.update!(opt, ps, gs)
        end
        cur_loss = evaluate_loss(loss, i_train, e_train, device=device)
        push!(losses, cur_loss)
        push!(testing_losses, evaluate_loss(loss, i_test, e_test, device=device))

        if epoch % plot_every == 0
            results = plot(layer(x=1:length(losses), y=losses, Geom.line),
            layer(x=1:length(testing_losses), y=testing_losses, Geom.line, Theme(default_color=color("orange"))),
            Guide.xlabel("Iter"), Guide.ylabel("loss"))
            draw(PNG("/u/jacquinn/graphs/week-end_run/results_13H107_min-5_ALL_epoch-$(epoch).png", 50cm, 50cm), results)
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

train_and_plot(device=gpu)
