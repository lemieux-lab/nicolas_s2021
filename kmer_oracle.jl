using CUDA
using Flux
using Gadfly
using ProgressBars
using Statistics
using HDF5
using Cairo
using Dates
using JLD2
using FileIO

include("kmer_utils.jl")
CUDA.allowscalar(false)

Base.@kwdef struct Hyperparams
    training_rate::Float32=0.01
    batchsize::Int64=1000
    use_log_counts::Bool=true
end

Base.@kwdef struct Datafile
    path::String
    dataset::String
end

Base.@kwdef struct Plotparams
    plot_loss::Bool=true
    plot_covariance::Bool=true
    show_plots::Bool=true
    save_plots::Bool=true
    plot_path::String=""
    plot_every::Int32=5
end

function evaluate_loss(loss::Function, kmers::Array{Bool, 2}, counts::Array{Float64, 1}; device::Function=cpu)
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
            Dense(4*k÷2, 4*k÷4, x->σ.(x)),
            Dense(4*k÷4, 1, identity)
            )
end

function plot_loss(train_losses::Vector{Float32}, test_losses::Vector{Float32})
    df = DataFrame(train = train_losses, test = test_losses, epoch=0:length(train_losses)-1)
    graph = plot(stack(df, [:train, :test]), x=:epoch, y=:value, 
            color=:variable, Guide.xlabel("Epoch"), Guide.ylabel("Loss"), 
            Guide.title("Loss per epoch"), Theme(panel_fill="white"), Geom.line)
    return graph
end

function plot_covariance(covariances::Vector{Float32})
    df = DataFrame(covar = covariances, epoch = 0:length(covariances)-1)
    graph = plot(df, x=:epoch, y=:covar, Guide.xlabel("Epoch"), Guide.ylabel("Covariance"), 
    Guide.title("Covariance per epoch"), Theme(panel_fill="white"), Geom.line)
    return graph
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

    # Saving params if plots are saved aswell
    if visu.save_plots
        JLD2.save("$(visu.plot_path)params.jld2", 
                  Dict("hyper"=>hyper, "visu"=>visu, "data"=>data))
    end

    # Reading data
    h5_file = h5open(data.path, "r")
    all_kmers = read(h5_file["kmers/$(data.dataset)"])
    all_counts = read(h5_file["counts/$(data.dataset)"])
    close(h5_file)

    # Apply log counts
    if hyper.use_log_counts
        all_counts = log.(all_counts)
    end

    # Splitting into sets
    i_train, e_train, i_test, e_test = split_kmer_data(all_kmers, all_counts, 75)

    # Preparing hyperparams
    network = neural_network() |> device
    loss(input, expected) = Flux.Losses.mse(network(input), expected) + L2_penalty()
    dl_train = Flux.Data.DataLoader((i_train, e_train), batchsize=hyper.batchsize, shuffle=true)
    opt = Flux.ADAM(hyper.training_rate)
    ps = Flux.params(network)

    # Empty lists that will contain loss values over iterations (for plotting)
    losses = Float32[]
    testing_losses = Float32[]
    covariances = Float32[]
    
    # This is where the fun begins. Main training loop
    epoch = 0
    start_time = now()
    while true
        epoch_start_time = now()
        epoch += 1

        # Evaluates loss on training & testing sets
        if visu.plot_loss
            push!(losses, evaluate_loss(loss, i_train, e_train, device=device))
            push!(testing_losses, evaluate_loss(loss, i_test, e_test, device=device))
        end

        if visu.plot_covariance
            push!(covariances, cov(evaluate_results(network, i_test, device=device), e_test))
        end

        if nohup
            iter = dl_train
        else
            iter = ProgressBar(dl_train)
        end

        # Iterates over the dataset to train the NN
        for (i, e) in iter
            gs = gradient(ps) do
                loss(device(i), device(e))
            end
            Flux.update!(opt, ps, gs)
        end

        # Plots loss tracking data every X epoch
        if visu.plot_loss && (epoch % visu.plot_every == 0)
            loss_graph = plot_loss(losses, testing_losses)
            if visu.save_plots
                draw(PNG("$(visu.plot_path)loss_at_epoch-$(epoch).png"), loss_graph)
            end
            if visu.show_plots
                draw(PNG(), loss_graph)
            end
        end

        if visu.plot_covariance && (epoch % visu.plot_every == 0)
            covar_graph = plot_covariance(covariances)
            if visu.save_plots
                draw(PNG("$(visu.plot_path)covariance_at_epoch-$(epoch).png"), covar_graph)
            end
            if visu.show_plots
                draw(PNG(), covar_graph)
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
file = "/home/golem/rpool/scratch/jacquinn/data/13H107-k31.h5"
dataset = "13H107-k31_min-5_ALL"
data = Datafile(file, dataset)

# Hyperparams struct
hyper = Hyperparams()

# Plotparams struct
visu = Plotparams(plot_every=2, plot_path="/u/jacquinn/graphs/L2_and_log/")

train_and_plot(data, hyper, visu, device=gpu)
