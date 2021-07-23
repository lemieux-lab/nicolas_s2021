using BSON: @save

fasta_files = [
    "/home/golem/scratch/jacquinn/data/13H107-k31_min-5.FASTA",
    "/home/golem/scratch/jacquinn/data/14H171_min-5.FASTA",
    "/home/golem/scratch/jacquinn/data/16H106_min-5.FASTA",
    "/home/golem/scratch/jacquinn/data/17H073_min-5.FASTA"
    # "/home/golem/scratch/jacquinn/data/13H107-k31_min-5_10k.FASTA"
]

Base.@kwdef struct Hyperparams
    training_rate::Float32=0.001
    batchsize::Int64=4096
    use_log_counts::Bool=true
    substract_average::Bool=true
    l2_multiplier::Float32=1
end

Base.@kwdef struct Datafile
    paths::Array{String, 1}=fasta_files
    save_model::Bool=true
    save_every::Int64=5
    model_path::String=""
end

Base.@kwdef struct Plotparams
    plot_loss::Bool=true
    plot_accuracy::Bool=true
    plot_l2::Bool=true
    show_plots::Bool=true
    save_plots::Bool=true
    plot_path::String=""
    plot_every::Int32=5
    accuracy_subset::Int32=100000
end

function generate_params(save_path::String="", save_file::String="params.bson", log_file::String="params.txt")
    
    # Generating params with current definitions
    hp, df, pp = Hyperparams(), Datafile(model_path=save_path*"/"), Plotparams(plot_path=save_path*"/")
    
    # Saving the params if the path isn't empty
    if length(save_path) != 0
        
        # Saving loadable Julia object file
        @save "$(save_path)/$(save_file)" hp df pp
        
        # Parsing params into text and saving that text in a txt file
        param_logs = "Parameters used for this run are written in this file."
        param_logs *= " Please use the .bson file to load the struct in Julia"
        for param_struct in (hp, df, pp)
            param_logs *= "\n\n<---$(typeof(param_struct))--->"
            for field in fieldnames(typeof(param_struct))
                param_logs *= "\n$field = $(getfield(param_struct, field))"
            end
        end

        open("$(save_path)/$(log_file)", "w") do io
            write(io, param_logs)
        end
    end
    return hp, df, pp
end

function save_model_architecture(network, save_path::String="", log_file::String="params.txt")
    open("$(save_path)/$(log_file)", "a") do io
       write(io, "\n\n<---Network Model-->\n$(network)")
    end
end