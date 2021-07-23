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

include("kmer_utils.jl")
include("plot_utils.jl")
include("run_params.jl")
include("disk_utils.jl")
CUDA.allowscalar(true)

function neural_network(k::Int64=31)
    return Chain(
        Conv((11,), 4=>200, relu),
        Conv((11,), 200=>400, relu),
        Conv((11,), 400=>200, relu),
        Flux.flatten,
        Dense(200, 100, relu),
        Dense(100, 1, identity)
    )
end

function main(; run_path::String="", nohup::Bool=false, device::Function=gpu)
    
    function l2_penalty()
        penalty = 0
        for layer in network
            if typeof(layer) != typeof(Flux.flatten)
                # layer = layer
                penalty += sum(abs2, layer.weight)
            end
        end
        return penalty
    end

    loss(input, expected) = Flux.Losses.mse(network(input), expected) + (l2_penalty() * hp.l2_multiplier)

    # Initialising the run's params
    hp, df, pp = generate_params(run_path)

    # Initialising the network and training tools
    network = device(neural_network())
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
        track_expected = Float32[]
        track_obtained = Float32[]

        # Prepare to stream data from disk
        not_done = true
        reader = Datareader(datafile = df)

        while not_done
            not_done, i, e = load_next_batch(reader)
            
            # Preparing input/expected values
            true_i = Array{Float32, 3}(undef, 31, 4, length(i))
            for (j, t) in enumerate(onehot_kmer.(i))
                # println(size(t))
                true_i[:, :, j] = t
            end
            i = true_i
            # println(i)
            if hp.use_log_counts
                e = log10.(e)
            end
            e = reshape(e, 1, length(e))
            e = convert(Matrix{Float32}, e)
            i, e = device(i), device(e)

            # evaluates the network's performances
            obtained = network(i)
            l2_value = l2_penalty() * hp.l2_multiplier
            loss_value = Flux.Losses.mse(obtained, e) + l2_value
            push!(epoch_losses, cpu(loss_value))
            if length(track_expected) < pp.accuracy_subset
                push!(track_expected, cpu(e)...)
                push!(track_obtained, cpu(obtained)...)
            end

            # Calculating gradient, then descend
            gs = gradient(() -> loss(i, e), ps)
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
                accur_graph = plot_accuracy(track_obtained, track_expected, pp.accuracy_subset, epoch)
                if pp.save_plots
                    draw(PNG("$(pp.plot_path)accuracy_at_epoch-$(epoch-1).png"), accur_graph)
                end
                if pp.show_plots
                    draw(PNG(), accur_graph)
                end
                accur_graph = plot_hex_accuracy(track_obtained, track_expected, pp.accuracy_subset, epoch)
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


main(run_path="/u/jacquinn/Aboleth/CNN_4_samples", nohup=true, device=gpu)
