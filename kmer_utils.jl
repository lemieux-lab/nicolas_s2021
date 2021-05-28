using Flux
using ProgressBars
using HDF5
using DataFrames

# Opens a kmer count dump file from jellyfish and parses it into a dataframe
function parse_kmer_count(path::String; min_count::Int64=0, max_lines::Int64=-1)
    open(path) do file
        kmers = Vector{Bool}[]
        counts = Int32[]
        for (i, line) in ProgressBar(enumerate(eachline(file)))
            if max_lines != -1 && i > max_lines
                break
            end
            if i%2 == 0
                stripped_kmer = String(strip(line))
                # push!(kmers, stripped_kmer)
                # println(onehot_kmer(stripped_kmer))
                # return
                push!(kmers, onehot_kmer(stripped_kmer))
                # kmers = [kmers; onehot_kmer(stripped_kmer)]
                
            else
                count =  parse(Int32, line[2:length(line)])
                if count < min_count
                    continue
                end
                push!(counts, count)
            end
        end
        return @time hcat(kmers...), counts
    end
end

# Encodes a kmer into a onehot array based on the nucleotides
function onehot_kmer(kmer::String)
    return vcat(Flux.onehotbatch(split(kmer, ""), ["A", "T", "G", "C"])...)
end

# Transformer le DF en sets utilisable par le neural net
function create_flux_sets(kmer_df::DataFrame)
    kmers = onehot_kmer.(kmer_df[!, "kmers"])
    counts = kmer_df[!, "counts"]
    return kmers, counts
end

# Splits the DF into 2 based on a percentage
function split_kmer_df(df::DataFrame, split_indice::Int64)
    if split_indice > 100 || split_indice < 1
        return
    end
    split_index = (nrow(df)*split_indice)÷100
    return df[1:split_index, [:kmers, :counts]], df[split_index+1:nrow(df), [:kmers, :counts]]
end

function kmer_to_hdf5(kmer_file::String, output_file::String, dataset_name::String; min_count::Int64=0, max_lines::Int64=-1)
    kmers, counts = parse_kmer_count(kmer_file, min_count=min_count, max_lines=max_lines)
    # println(hcat(kmers...)[1])
    h5open(output_file, "w") do h5_file
        @time create_group(h5_file, "kmers")
        @time create_group(h5_file, "counts")
        @time h5_file["kmers"]["kmers"] = kmers
        @time h5_file["counts"]["counts"] = counts
    end
end

# parsed_df = parse_kmer_count("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_min-5.FASTA", max_lines = 300000)

# ↓ About 8.5 hours. Parses then saves the entire dataset with onehot encoded kmers in a hdf5 file.
# ↓ GC time is over 73%, this needs to be dealt with.
# @time kmer_to_hdf5("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_min-5.FASTA", 
#                    "/home/golem/rpool/scratch/jacquinn/data/encoded_kmer_counts.h5", 
#                    "13H107-k31_min-5")


