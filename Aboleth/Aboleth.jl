# Imports
using CUDA
using Flux
using Dates
using Statistics
using MLDataUtils
using ProgressBars
using Gadfly
using Cairo
using BSON: @save
using HDF5

include("kmer_utils.jl")
include("plot_utils.jl")
include("run_params.jl")
include("disk_utils.jl")
include("reindeer_bridge.jl")
CUDA.allowscalar(false)

function neural_network(sample_nb::Int64=1)
    return Chain(
        Conv((5,), 4=>27, relu),
        Conv((5,), 27=>23, relu),
        # Conv((11,), 400=>700, relu),
        Flux.flatten,
        Dense(23*23, 300, relu),
        Dense(300, sample_nb, identity),
        transpose  # Output matrix needs to be transposed to match expected matrice
    )
end

# Reads as many files as provided by the Datafile struct, and returns the kmers vector
function read_files(paths::Array{String, 1}, datasets::Array{String, 1}; k::Int64=31, nohup::Bool=false)
    kmers = Array{String, 1}(undef, 0)
    iter = zip(paths, datasets)
    if !nohup
        iter = ProgressBar(iter) 
        set_description(iter, "Loading datafiles...")
    end
    for (path, ds) in iter
        h5open(path, "r") do h5_file
            append!(kmers, read(h5_file["kmers/$(ds)"]))
        end
    end
    return kmers
end

function main(; run_path::String="", sample_nb::Int64=1, nohup::Bool=false, device::Function=gpu)
    
    # Calculates and returns the L2 penalty with network's current state
    function l2_penalty()
        penalty = 0
        for layer in network
            if typeof(layer) != typeof(Flux.flatten) && typeof(layer) != typeof(transpose)
                penalty += sum(abs2, layer.weight)
            end
        end
        return penalty
    end

    loss(input, expected) = Flux.Losses.mse(network(input), expected) + (l2_penalty() * hp.l2_multiplier)

    # Loading a Reindeer index. Note that the passed sample_nb must match the amount of smamples in index
    println("Loading Reindeer index...")
    @time load_reindeer_index("/home/golem/scratch/jacquinn/data/reindeer_files/13H107/output_reindeer")

    # Initialising the run's params
    hp, df, pp = generate_params(run_path)

    # Initialising the network and training tools
    network = device(neural_network(sample_nb))
    save_model_architecture(network, run_path)
    opt = Flux.ADAM(hp.training_rate)
    ps = cu(Flux.params(network))

    # These will be used to record results and track network performances
    training_losses = Float32[]
    testing_losses = Float32[]
    l2_values = Float32[]
    epoch = 0
    start_time = now()

    # Main training loop
    while true
        epoch_start_time = now()
        epoch += 1

        # Saves model if nescessary
        if df.save_model && epoch % df.save_every == 0
            model = cpu(network)
            @save "$(df.model_path)network_at_epoch-$(epoch).bson" model
        end

        # Used to track the network performances over this specific epoch
        epoch_losses =  Float32[]
        track_expected = Array{Float32, 2}(undef, 0, sample_nb)
        track_obtained = Array{Float32, 2}(undef, 0, sample_nb)
        kmers_to_query = read_files(["/home/golem/scratch/jacquinn/data/reindeer_query_data/kmers.hdf5"], ["17H073"], nohup=nohup)
        kmer_list = Flux.DataLoader(kmers_to_query, batchsize=512, shuffle=true)

        # kmers are on RAM. Loops through kmer bathes to be queried
        for kmer_batch in kmer_list
            
            # Using the Reindeer API to get the counts of the current kmer batch
            e = reindeer_query_counts(kmer_batch, sample_nb)
            
            # onehot-encoding the kmers for nn processing
            hot_kmer_batch = Array{Float32, 3}(undef, 31, 4, length(kmer_batch))
            for (j, t) in enumerate(onehot_kmer.(kmer_batch))
                hot_kmer_batch[:, :, j] = t
            end
            if hp.use_log_counts
                e = log10.(e.+1)
            end
            e = convert(Matrix{Float32}, e)
            hot_kmer_batch, e = device(hot_kmer_batch), device(e)

            # evaluates the network's performances
            obtained = network(hot_kmer_batch)
            l2_value = l2_penalty() * hp.l2_multiplier
            loss_value = Flux.Losses.mse(obtained, e) + l2_value
            push!(epoch_losses, cpu(loss_value))
            
            # Makes sure the number of points on the graph isn't too big
            if length(track_expected) < pp.accuracy_subset
                track_expected = vcat(track_expected, cpu(e))
                track_obtained = vcat(track_obtained, cpu(obtained))
            end

            # Calculating gradient, then descend
            gs = gradient(() -> loss(hot_kmer_batch, e), ps)
            Flux.update!(opt, ps, gs)
        end
        push!(l2_values, cpu(l2_penalty() * hp.l2_multiplier))
        push!(training_losses, mean(epoch_losses))
        push!(testing_losses, mean(epoch_losses))

        # Handles plotting whenever it's time
        if epoch % pp.plot_every == 0

            if pp.plot_loss
                if pp.plot_l2
                    loss_graph = plot_loss_with_l2(training_losses, testing_losses, l2_values)
                else
                    loss_graph = plot_loss(training_losses, testing_losses)
                end
                if pp.save_plots
                    draw(PNG("$(pp.plot_path)loss_at_epoch-$(epoch-1).png"), loss_graph)
                end
                if pp.show_plots
                    draw(PNG(), loss_graph)
                end
            end
            
            if pp.plot_accuracy
                accur_graph = plot_accuracy(track_obtained, track_expected, epoch)
                if pp.save_plots
                    draw(PNG("$(pp.plot_path)accuracy_at_epoch-$(epoch-1).png"), accur_graph)
                end
                if pp.show_plots
                    draw(PNG(), accur_graph)
                end
                accur_graph = plot_hex_accuracy(track_obtained, track_expected, epoch)
                if pp.save_plots
                    draw(PNG("$(pp.plot_path)hex_accuracy_at_epoch-$(epoch-1).png"), accur_graph)
                end
                if pp.show_plots
                    draw(PNG(), accur_graph)
                end
            end
        end
        
        #Show reports
        println("current epoch: $(epoch)\nepoch runtime: $(now()-epoch_start_time)\nglobal runtime: $(now()-start_time)\n\n\n")
        if nohup
            flush(stdout)
        end

    end
end

main(run_path="/u/jacquinn/Aboleth/CNN_reindeer_1_sample", sample_nb=1, nohup=true, device=gpu)
