# Shitty implementation of kmer_oracle with data fed from the disk. Will be replaced soon "Will be named Aboleth.jl"

using CUDA
using Flux
using Gadfly
using ProgressBars
using Statistics
using MLDataUtils
# using HDF5
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
    no_testing_set::Bool=true
end

Base.@kwdef struct Datafile
    paths::Array{String, 1}
    save_model::Bool=true
    save_every::Int64=5
    model_path::String=""
end

Base.@kwdef mutable struct Datareader
    datafile::Datafile
    open_files::Array{IOStream, 1}=init_files(datafile)
    current_file::Int64=1
    current_line::Int64=1
    load_interval::Int64=2048
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
    plot_training::Bool=true
    accuracy_subset::Int32=100000
end

# Evaluates the loss values on a whole set, returns average value
function evaluate_loss(loss::Function, kmers::SubArray{Bool, 2}, counts::SubArray{Float64, 2}; device::Function=cpu)
    dl = Flux.Data.DataLoader((kmers, counts), batchsize=2048)
    # total = Array{Float32, 1}()
    total = Float32(0)
    divider = Float32(0)
    for (i, e) in dl
        # push!(total, loss(device(i), device(e)))
        total += loss(device(i), device(e))
        divider += 1
    end
    return total/divider
end

# Evaluates the loss values on a whole set, returns average value
function evaluate_loss(loss::Function, kmers::SubArray{Bool, 2}, counts::SubArray{Int32, 2}; device::Function=cpu)
    dl = Flux.Data.DataLoader((kmers, counts), batchsize=2048)
    # total = Array{Float32, 1}()
    total = Float32(0)
    divider = Float32(0)
    for (i, e) in dl
        # push!(total, loss(device(i), device(e)))
        total += loss(device(i), device(e))
        divider += 1
    end
    return total/divider
end

# Passes every element from the set to the newtork and returns the outputs
function evaluate_results(network, testing_set::SubArray{Bool, 2}; device::Function=cpu)
    o_test = Array{Float32, 1}()
    dl = Flux.Data.DataLoader(testing_set, batchsize=2048)
    for i in dl
        push!(o_test, cpu(network(device(i)))...)
    end
    return o_test
end

# Generates dummy data for testing
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

# Returns the neural network as a Chain of Dense
function neural_network(k::Int64=31)
    return Chain(
            Dense(4*k, 3000, relu),
            # Dense(500, 500, relu),
            Dense(3000, 1500, relu),
            Dense(1500, 1, identity)
            )
end

# ↓ Plot functions for visualizing the results ↓

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
    # df now has a "variable" column for l2 & loss, and a "set" color for train, loss & L2
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
    sub_expected = expected[:, subset_indexes]
    graph = plot(x=sub_expected, y=sub_obtained, 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"),
            Guide.title("Accuracy at epoch $(epoch-1)"), Theme(panel_fill="white"),
            Geom.point)
    return graph
end

function plot_hex_accuracy(obtained, expected, subset, epoch)
    subset_indexes = rand(1:length(obtained), subset)
    sub_obtained = obtained[subset_indexes]
    sub_expected = expected[:, subset_indexes]
    graph = plot(x=sub_expected, y=sub_obtained, 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"),
            Guide.title("Accuracy at epoch $(epoch-1)"), Theme(panel_fill="white"),
            Geom.hexbin, Geom.abline)
    return graph
end

# Reads as many files as provided by the Datafile struct, and returns the vectors
function read_files(data::Datafile; k::Int64=31, nohup::Bool=false)
    kmers = Array{Bool, 2}(undef, 4*k, 0)
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
    return kmers, reshape(counts, 1, length(counts))
end

function init_files(datafile::Datafile)
    streams = Array{IOStream, 1}(undef, length(datafile.paths))
    for (i, path) in enumerate(datafile.paths)
        streams[i] = open(path)
    end
    return streams
end

function load_next_batch(reader::Datareader)
    counter = 0
    kmers = Array{String, 1}(undef, reader.load_interval÷2)
    counts = Array{Any, 1}(undef, reader.load_interval÷2)
    for line in eachline(reader.open_files[reader.current_file])
        if counter % 2 != 0
            kmers[counter÷2+1] = line
        else
            counts[counter÷2+1] = parse(Int32, strip(line, '>'))
        end
        counter += 1
        if counter >= reader.load_interval
            break
        end
    end
    if counter < reader.load_interval
        kmers = [kmers[i] for i in 1:length(kmers) if isassigned(kmers, i)]
        counts = [counts[i] for i in 1:length(counts) if isassigned(counts, i)]

        if reader.current_file < length(reader.open_files)
            reader.current_file += 1
        else
            return (false, kmers, convert(Array{Int32}, counts))
        end
    end
    return (true, kmers, convert(Array{Int32}, counts))
end

# Main function
function train_and_plot(data::Datafile, hyper::Hyperparams, visu::Plotparams; device::Function=cpu, nohup::Bool=false)
    
    function L2_penalty()
        penalty = 0
        for layer in network
            penalty += sum(abs2, layer.W) + sum(abs2, layer.b)
        end
        return penalty
    end

    # Saving params if plots are saved aswell
    if visu.save_plots
        JLD2.save("$(visu.plot_path)params.jld2", 
                  Dict("hyper"=>hyper, "visu"=>visu, "data"=>data))
    end

    # all_kmers, all_counts = read_files(data, nohup=nohup)
    # all_kmers = all_kmers[:, 1:100]
    # all_counts = all_counts[:, 1:100]
    # println(all_kmers)
    # all_kmers, all_counts = generate_fake_data()

    # Apply log counts
    # if hyper.use_log_counts
    #     transforme(x) = log(10, x) + 1
    #     all_counts = transforme.(all_counts)
    # end

    # aver = mean(all_counts)
    # println("Average exected value: $aver")

    # if hyper.substract_average
    #     all_counts = all_counts .- aver
    # end

    # Splitting into sets
    # if hyper.no_testing_set
    #     i_train = all_kmers
    #     e_train = all_counts
    #     i_test = splitobs(all_kmers, at=0.95)[2]
    #     e_test = splitobs(all_counts, at=0.95)[2]
    # else
    #     i_train, i_test = splitobs(all_kmers, at=0.95)
    #     e_train, e_test = splitobs(all_counts, at=0.95)
    #     println(typeof(i_train), typeof(e_train))
    # end
    # i_train, e_train, i_test, e_test = split_kmer_data(all_kmers, all_counts, 99, hyper.no_testing_set)
    # e_train = reshape(e_train, 1, length(e_train))
    # e_test = reshape(e_test, 1, length(e_test))

    network = neural_network() |> device
    # network = device(neural_network())
    loss(input, expected) = Flux.Losses.mse(network(input), expected) + (L2_penalty() * hyper.l2_multiplier)
    # loss(input, expected) = Flux.Losses.huber_loss(network(input), expected) + (L2_penalty() * hyper.l2_multiplier)
    # dl_train = Flux.Data.DataLoader((i_train, e_train), batchsize=hyper.batchsize, shuffle=true)
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

    # Main training loop starts
    while true
        epoch_start_time = now()
        epoch += 1

        # # Evaluates loss on training & testing sets
        # if visu.plot_loss
        #     push!(losses, evaluate_loss(loss, i_train, e_train, device=device))
        #     push!(testing_losses, evaluate_loss(loss, i_test, e_test, device=device))
        #     if visu.plot_l2
        #         push!(l2_values, hyper.l2_multiplier * L2_penalty())
        #     end
        # end

        # if visu.plot_correlation || visu.plot_accuracy
        #     if visu.plot_training
        #         results = evaluate_results(network, i_train, device=device)
        #         obtained = results
        #     else
        #         results = evaluate_results(network, i_test, device=device)
        #         obtained = results
        #     end

        #     if visu.plot_correlation
        #         if visu.plot_training
        #             push!(correlations, cor(results, e_train))
        #         else
        #             push!(correlations, cor(results, e_test))
        #         end
        #     end
        # end

        # if data.save_model && (epoch % data.save_every == 0)
        #     model = cpu(network)
        #     @save "$(data.model_path)model_at_epoch-$(epoch-1).bson" model
        # end


        # if nohup
        #     iter = dl_train
        # else
        #     iter = ProgressBar(dl_train)
        #     set_description(iter, "Training...")
        #     set_postfix(iter, Epoch=string(epoch))
        # end

        epoch_losses = Float32[]
        expected = Float32[]
        obtained = Float32[]

        # Iterates over the dataset to train the NN
        not_done = true
        reader = Datareader(datafile = data)
        while not_done
            not_done, i, e = load_next_batch(reader)
            i = hcat(onehot_kmer.(i)...) |> device
            e = reshape(e, 1, length(e)) |> device
            push!(epoch_losses, loss(i, e))
            if length(obtained) < 100000
                push!(expected, e...)
                push!(obtained, network(i)...)
            end
            # push!(testing_losses, loss(i, e))
            gs = gradient(ps) do
                loss(i, e)
            end
            Flux.update!(opt, ps, gs)
        end
        push!(losses, mean(epoch_losses))
        push!(testing_losses, mean(epoch_losses))

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
            if visu.plot_training
                accur_graph = plot_accuracy(obtained, expected, visu.accuracy_subset, epoch)
            else
                accur_graph = plot_accuracy(obtained, expected, visu.accuracy_subset, epoch)
            end
            if visu.save_plots
                draw(PNG("$(visu.plot_path)accuracy_at_epoch-$(epoch-1).png"), accur_graph)
            end
            if visu.show_plots
                draw(PNG(), accur_graph)
            end
            if visu.plot_training
                accur_graph = plot_hex_accuracy(obtained, expected, visu.accuracy_subset, epoch)
            else
                accur_graph = plot_hex_accuracy(obtained, expected, visu.accuracy_subset, epoch)
            end
            if visu.save_plots
                draw(PNG("$(visu.plot_path)hex_accuracy_at_epoch-$(epoch-1).png"), accur_graph)
            end
            if visu.show_plots
                draw(PNG(), accur_graph)
            end
        end

        #Show reports
        println("current epoch: $(epoch)\nepoch runtime: $(now()-epoch_start_time)\nglobal runtime: $(now()-start_time)\n\n\n")
        if nohup
            flush(stdout)
        end
    end
end

# Datafile struct
paths = [
    "/home/golem/rpool/scratch/jacquinn/data/13H107-k31_min-5.FASTA",
    "/home/golem/rpool/scratch/jacquinn/data/14H171_min-5.FASTA",
    "/home/golem/rpool/scratch/jacquinn/data/16H106_min-5.FASTA",
    "/home/golem/rpool/scratch/jacquinn/data/17H073_min-5.FASTA"
]

path = "/u/jacquinn/graphs_july/from_disk_with_accuracy/"

# Loading data and saving the model
data = Datafile(paths=paths, model_path=path, 
                save_every=5, save_model=false)

# Hyperparams struct
hyper = Hyperparams(training_rate=0.00001, l2_multiplier=1e-5, 
                    use_log_counts=true, batchsize=2048, no_testing_set=false)

# Plotparams struct
visu = Plotparams(plot_every=2, plot_path=path, plot_correlation=false,
                  plot_accuracy=false, plot_l2=false, show_plots=false)

train_and_plot(data, hyper, visu, device=gpu, nohup=true)
