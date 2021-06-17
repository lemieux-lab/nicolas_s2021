using CUDA
using Flux
using Gadfly
using ProgressBars
using Statistics
using HDF5
using Cairo
using Dates
using JLD2
using BSON: @save
using FileIO

include("kmer_utils.jl")
CUDA.allowscalar(false)

Base.@kwdef struct Hyperparams
    training_rate::Float32=0.001
    batchsize::Int64=2048
    use_log_counts::Bool=true
    substract_average::Bool=true
    l2_multiplier::Float32=1
    use_training_set::Bool=true
end

Base.@kwdef struct Datafile
    paths::Array{String, 1}
    datasets::Array{String, 1}
    save_model::Bool=true
    save_every::Int64=5
    model_path::String=""
end

Base.@kwdef struct Plotparams
    plot_loss::Bool=true
    plot_correlation::Bool=true
    plot_accuracy::Bool=true
    plot_l2::Bool=true
    show_plots::Bool=true
    save_plots::Bool=true
    plot_path::String=""
    plot_every::Int32=5
    accuracy_subset::Int32=100000
end


function evaluate_loss(loss::Function, kmers::Array{Bool, 2}, counts::Array{Float64, 2}; device::Function=cpu)
    dl = Flux.Data.DataLoader((kmers, counts), batchsize=2048)
    total = Array{Float32, 1}()
    for (i, e) in dl
        push!(total, loss(device(i), device(e)))
    end
    return mean(total)
end

function evaluate_loss(loss::Function, kmers::Array{Bool, 2}, counts::Array{Int32, 2}; device::Function=cpu)
    dl = Flux.Data.DataLoader((kmers, counts), batchsize=2048)
    total = Array{Float32, 1}()
    for (i, e) in dl
        push!(total, loss(device(i), device(e)))
    end
    return mean(total)
end

function evaluate_results(network, testing_set::Array{Bool, 2}; device::Function=cpu)
    o_test = Array{Float32, 1}()
    dl = Flux.Data.DataLoader(testing_set, batchsize=2048)
    for i in dl
        push!(o_test, cpu(network(device(i)))...)
    end
    return o_test
end

function generate_fake_data(size::Int64=1000000)
    kmers = Array{Bool, 2}(undef, 124, size)
    values = Array{Float64, 1}(undef, size)
    iter = ProgressBar(1:size)
    set_description(iter, "Generating data...")

    for i in iter
        to_add = rand(Bool, 124)
        f(x) = x
        value = count(f, to_add[1:31]) - count(f, to_add[32:64]) + count(f, to_add[65:95]) - + count(f, to_add[96:end])
        kmers[:,i] = to_add
        values[i] = value
    end
    return (kmers, values)
end

function neural_network(k::Int64=31)
    return Chain(
            Dense(4*k, 700, relu),
            Dense(700, 350, relu),
            # Dense(4*k÷4, 4*k÷8, relu),
            Dense(350, 1, identity)
            )
end

# function neural_network(k::Int64=31)
#     return Chain(
#         Parallel(vcat, 
#             Dense(k, 100),
#             Dense(k, 100),
#             Dense(k, 100),
#             Dense(k, 100)
#         ),
#         Dense(400, 1)
#     )
# end

function plot_loss(train_losses::Vector{Float32}, test_losses::Vector{Float32})
    df = DataFrame(train = train_losses, test = test_losses, epoch=0:length(train_losses)-1)
    graph = plot(stack(df, [:train, :test]), x=:epoch, y=:value, 
            color=:variable, Guide.xlabel("Epoch"), Guide.ylabel("Loss"), 
            Guide.title("Loss per epoch"), Theme(panel_fill="white"), Geom.line)
    return graph
end

function plot_loss_with_l2(train_losses::Vector{Float32}, test_losses::Vector{Float32}, l2_values::Vector{Float32})
    df = DataFrame(train = train_losses, test = test_losses, 
                   l2 = l2_values, epoch=0:length(train_losses)-1)
    df = stack(df, [:train, :test])
    rename!(df, Dict(:variable => "set", :value => "loss"))
    df = stack(df, [:l2, :loss])
    df[df.variable.== "l2", "set"].="l2"
    graph = plot(df, ygroup = "variable", x="epoch", y="value", color="set", 
                 Geom.subplot_grid(Geom.line, free_y_axis=true),
                 Guide.title("Loss & L2 per epoch"), Theme(panel_fill="white"))
    return graph
end

function plot_correlation(correlations::Vector{Float32})
    df = DataFrame(correl = correlations, epoch = 0:length(correlations)-1)
    graph = plot(df, x=:epoch, y=:correl, Guide.xlabel("Epoch"), Guide.ylabel("Corellation"), 
            Guide.title("Corellation per epoch"), Theme(panel_fill="white"), Geom.line)
    return graph
end

function plot_accuracy(obtained, expected, subset, epoch)
    subset_indexes = rand(1:length(obtained), subset)
    sub_obtained = obtained[subset_indexes]
    sub_expected = expected[subset_indexes]
    graph = plot(x=sub_expected, y=sub_obtained, 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"),
            Guide.title("Accuracy at epoch $(epoch-1)"), Theme(panel_fill="white"),
            Geom.point)
    return graph
end

function plot_hex_accuracy(obtained, expected, subset, epoch)
    subset_indexes = rand(1:length(obtained), subset)
    sub_obtained = obtained[subset_indexes]
    sub_expected = expected[subset_indexes]
    graph = plot(x=sub_expected, y=sub_obtained, 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"),
            Guide.title("Accuracy at epoch $(epoch-1)"), Theme(panel_fill="white"),
            Geom.hexbin, Geom.abline)
    return graph
end

function read_files(data::Datafile; k::Int64=31, nohup::Bool=false)
    
    kmers = Array{Bool,  2}(undef, 4*k, 0)
    counts = Int32[]
    iter = zip(data.paths, data.datasets)
    if !nohup
        iter = ProgressBar(iter) 
        set_description(iter, "Loading datafiles...")
    end
    for (path, ds) in iter
        h5open(path, "r") do h5_file
            kmers = hcat(kmers, read(h5_file["kmers/$(ds)"]))
            append!(counts, read(h5_file["counts/$(ds)"]))
        end
    end
    return kmers, counts
end

function train_and_plot(data::Datafile, hyper::Hyperparams, visu::Plotparams; device::Function=cpu, nohup::Bool=false)
    # This calculates an L2 regularization for the loss
    function L2_penalty() # = sum(sum.(abs2, network.W)) + sum(sum.(abs2, network.b))
        penalty = 0
        for layer in network
            penalty += sum(abs2, layer.W) + sum(abs2, layer.b)
        end
        return penalty
    end

    # function L2_penalty()
    #     penalty = 0
    #     for layer in network
    #         penalty += sum(abs2, layer.W)
    #     end
    #     return penalty
    # end

    # Saving params if plots are saved aswell
    if visu.save_plots
        JLD2.save("$(visu.plot_path)params.jld2", 
                  Dict("hyper"=>hyper, "visu"=>visu, "data"=>data))
    end

    all_kmers, all_counts = read_files(data, nohup=nohup)
    # all_kmers, all_counts = generate_fake_data()

    # Apply log counts
    if hyper.use_log_counts
        all_counts = log.(10, all_counts)
    end

    aver = mean(all_counts)
    println("Average exected value: $aver")

    if hyper.substract_average
        all_counts = all_counts .- aver
    end

    # Splitting into sets
    i_train, e_train, i_test, e_test = split_kmer_data(all_kmers, all_counts, 90, hyper.use_training_set)
    e_train = reshape(e_train, 1, length(e_train))
    e_test = reshape(e_test, 1, length(e_test))
    # println(onehot_kmer("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA") in eachcol(i_train))
    # println(onehot_kmer("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA") in eachcol(i_test))

    # function loss(input, expected)
    #     println("input:", input)
    #     obtained = network(input)
    #     println("obtained: ", obtained)
    #     println("expected: ", expected)
    #     val = Flux.Losses.mse(obtained, expected) + (L2_penalty() * hyper.l2_multiplier)
    #     println(val)
    #     return val
    # end

    # Preparing hyperparams
    network = neural_network() |> device
    loss(input, expected) = Flux.Losses.mse(network(input), expected) + (L2_penalty() * hyper.l2_multiplier)
    # loss(input, expected) = Flux.Losses.huber_loss(network(input), expected) + (L2_penalty() * hyper.l2_multiplier)
    dl_train = Flux.Data.DataLoader((i_train, e_train), batchsize=hyper.batchsize, shuffle=true)
    opt = Flux.ADAM(hyper.training_rate)
    ps = Flux.params(network)

    # Empty lists that will contain loss values over iterations (for plotting)
    losses = Float32[]
    testing_losses = Float32[]
    correlations = Float32[]
    obtained = Float32[]
    l2_values = Float32[]
    
    # This is where the fun begins. Main training loop
    epoch = 0
    start_time = now()
    # println(network(device(onehot_kmer("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"))))
    # return
    while true
        epoch_start_time = now()
        epoch += 1

        # Evaluates loss on training & testing sets
        if visu.plot_loss
            push!(losses, evaluate_loss(loss, i_train, e_train, device=device))
            push!(testing_losses, evaluate_loss(loss, i_test, e_test, device=device))
            if visu.plot_l2
                push!(l2_values, hyper.l2_multiplier * L2_penalty())
            end
        end

        if visu.plot_correlation || visu.plot_accuracy
            results = evaluate_results(network, i_test, device=device)
            obtained = results

            if visu.plot_correlation
                push!(correlations, cor(results, e_test))
            end
        end

        if data.save_model
            model = cpu(network)
            @save "$(data.model_path)model_at_epoch-$(epoch-1).bson" model
        end

        if nohup
            iter = dl_train
        else
            iter = ProgressBar(dl_train)
            set_description(iter, "Training...")
            set_postfix(iter, Epoch=string(epoch))
        end

        # Iterates over the dataset to train the NN
        for (i, e) in iter
            # e = reshape(e, 1, length(e))
            gs = gradient(ps) do
                loss(device(i), device(e))
            end
            Flux.update!(opt, ps, gs)
        end

        # Plots loss tracking data every X epoch
        if visu.plot_loss && (epoch % visu.plot_every == 0)
            if visu.plot_l2
                loss_graph = plot_loss_with_l2(losses, testing_losses, l2_values)
            else
                loss_graph = plot_loss(losses, testing_losses)
            end
            if visu.save_plots
                draw(PNG("$(visu.plot_path)loss_at_epoch-$(epoch-1).png"), loss_graph)
            end
            if visu.show_plots
                draw(PNG(), loss_graph)
            end
        end

        if visu.plot_correlation && (epoch % visu.plot_every == 0)
            correl_graph = plot_correlation(correlations)
            if visu.save_plots
                draw(PNG("$(visu.plot_path)correlation_at_epoch-$(epoch-1).png"), correl_graph)
            end
            if visu.show_plots
                draw(PNG(), correl_graph)
            end
        end

        if visu.plot_accuracy && (epoch % visu.plot_every == 0)
            accur_graph = plot_accuracy(obtained, e_test, visu.accuracy_subset, epoch)
            if visu.save_plots
                draw(PNG("$(visu.plot_path)accuracy_at_epoch-$(epoch-1).png"), accur_graph)
            end
            if visu.show_plots
                draw(PNG(), accur_graph)
            end
            # Nan bug with log counts? FIXME
            # accur_graph = plot_hex_accuracy(obtained, e_test, visu.accuracy_subset, epoch)
            # if visu.save_plots
            #     draw(PNG("$(visu.plot_path)hex_accuracy_at_epoch-$(epoch-1).png"), accur_graph)
            # end
            # if visu.show_plots
            #     draw(PNG(), accur_graph)
            # end
        end

        #Show reports
        println("current epoch: $(epoch)\nepoch runtime: $(now()-epoch_start_time)\nglobal runtime: $(now()-start_time)\n\n\n")
        # println(network(device(onehot_kmer("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"))))
        # println(network(device(onehot_kmer("ACTGCTTATATATGTATCCTTCAACAATATA"))))
        if nohup
            flush(stdout)
        end
    end
end

# Datafile struct
files = [# "/home/golem/rpool/scratch/jacquinn/data/14H171.h5",
         "/home/golem/rpool/scratch/jacquinn/data/13H107-k31.h5"
        ]
datasets = [#"14H171_min-5_ALL",
            "13H107-k31_min-5_ALL"
           ]

path = "/u/jacquinn/graphs_fixed_loss/0.00001_sub_mean_no_test/"
data = Datafile(paths=files, datasets=datasets, model_path=path, save_every=5)

# Hyperparams struct
hyper = Hyperparams(training_rate=0.00001, l2_multiplier=0.5, use_training_set=false, use_log_counts=true)

# Plotparams struct
# folder = "/u/jacquinn/graphs_fixed_loss/0.0000001tr_re-run/"
visu = Plotparams(plot_every=5, plot_path=path, plot_correlation=false, show_plots=false)

train_and_plot(data, hyper, visu, device=gpu, nohup=true)

# Try subtracting average count from count values
# Investigate model
