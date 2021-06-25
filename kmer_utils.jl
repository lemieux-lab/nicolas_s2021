using Flux
using ProgressBars
using HDF5
using DataFrames

# Opens a kmer count dump file from jellyfish and parses it into a dataframe
function parse_kmer_count(path::String; min_count::Int64=0, max_kmers::Int64=-1)
    file = open(path)
    line_nb = countlines(file)
    close(file)
    open(path) do file
        if max_kmers != -1
            limit = Int32(max_kmers)
        else
            limit = Int32(line_nb/2)
        end
        kmers = Array{Bool, 2}(undef, 124, limit)
        counts = Vector{Int32}(undef, limit)
        for (i, line) in ProgressBar(enumerate(eachline(file)))
            if i%2 == 0
                line = String(strip(line))
                kmers[:, (i-1)÷2+1] = onehot_kmer(line)
                if i == max_kmers*2
                    break
                end
                
            else
                line =  parse(Int32, line[2:length(line)])
                if line < min_count
                    continue
                end
                # push!(counts, count)
                counts[(i-1)÷2+1]=line
            end
        end
        return @time kmers, counts
    end
end

# Encodes a kmer into a onehot array based on the nucleotides
function onehot_kmer_old(kmer::SubString{String})
    kmer = split(kmer, "")
    return vcat(Flux.onehotbatch(kmer, ["A", "T", "G", "C"])...)
end

function onehot_kmer_old(kmer::String)
    kmer = split(kmer, "")
    return vcat(Flux.onehotbatch(kmer, ["A", "T", "G", "C"])...)
end

# New and improved onehot_kmer (about 12 times faster)
function onehot_kmer(kmer::String, kmer_length::Int64=31)
    # hot_kmer = Array{Bool, 2}(undef, 4, kmer_length)
    hot_kmer = fill(false, 4, kmer_length)
    encode_table = Dict(
        'A' => 1,
        'T' => 2,
        'G' => 3,
        'C' => 4
    )
    for (i, nuc) in enumerate(kmer)
        hot_kmer[encode_table[nuc], i] = true
    end
    return vcat(hot_kmer...)
end

function split_kmer_data(kmers::Array{Bool, 2}, counts::Array{Float64, 2}, split_indice::Int64, no_test::Bool=false)
    if split_indice > 100 || split_indice < 1
        return
    end
    split_index = (length(counts)*split_indice)÷100
    if no_test
        return kmers, counts, kmers[:, split_index+1:end], counts[:, split_index+1:end]
    end
    return kmers[:, 1:split_index], counts[:, 1:split_index], kmers[:, split_index+1:end], counts[:, split_index+1:end]
end

function split_kmer_data(kmers::Array{Bool, 2}, counts::Array{Int32, 2}, split_indice::Int64, no_test::Bool=false)
    if split_indice > 100 || split_indice < 1
        return
    end
    split_index = (length(counts)*split_indice)÷100
    if no_test
        return kmers, counts, kmers[:, split_index+1:end], counts[:, split_index+1:end]
    end
    return kmers[:, 1:split_index], counts[:, 1:split_index], kmers[:, split_index+1:end], counts[:, split_index+1:end]
end

function kmer_to_hdf5(kmer_file::String, output_file::String, dataset_name::String; min_count::Int64=0, max_kmers::Int64=-1)
    kmers, counts = parse_kmer_count(kmer_file, min_count=min_count, max_kmers=max_kmers)
    h5open(output_file, "cw") do h5_file
        if !("kmers" in keys(h5_file))
            create_group(h5_file, "kmers")
            create_group(h5_file, "counts")
        end
        h5_file["kmers"][dataset_name] = kmers
        h5_file["counts"][dataset_name] = counts
    end
end

# parsed_df = parse_kmer_count("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_min-5.FASTA", max_kmers = 300000)

# @time kmer_to_hdf5("/home/golem/rpool/scratch/jacquinn/data/17H073_min-5.FASTA", 
#                    "/home/golem/rpool/scratch/jacquinn/data/17H073_min-5.h5", 
#                    "17H073_min-5_ALL")
