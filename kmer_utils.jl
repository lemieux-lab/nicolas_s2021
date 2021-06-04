using Flux
using ProgressBars
using HDF5
using DataFrames

# Opens a kmer count dump file from jellyfish and parses it into a dataframe
function parse_kmer_count(path::String; min_count::Int64=0, max_kmers::Int64=-1)
    # line_nb = 0
    # open(path) do file
        # line_nb = countlines(file)
    #     # Première passe:
    #     for _ in ProgressBar(eachline(file))
    #         line_nb += 1
    #     end
    # end
    file = open(path)
    line_nb = countlines(file)
    close(file)
    open(path) do file
        if max_kmers != -1
            limit = Int32(max_kmers)
        else
            limit = Int32(line_nb/2)
        end
        # kmers = Vector{Vector{Bool}}(undef, limit)
        kmers = Array{Bool, 2}(undef, 124, limit)
        counts = Vector{Int32}(undef, limit)
        for (i, line) in ProgressBar(enumerate(eachline(file)))
            if i%2 == 0
                line = String(strip(line))
                kmers[:, (i-1)÷2+1] = onehot_kmer(line)
                # push!(kmers, stripped_kmer)
                # println(onehot_kmer(stripped_kmer))
                # return
                # push!(kmers, onehot_kmer(line))
                # kmers = [kmers; onehot_kmer(stripped_kmer)]
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
function onehot_kmer(kmer::SubString{String})
    kmer = split(kmer, "")
    return vcat(Flux.onehotbatch(kmer, ["A", "T", "G", "C"])...)
end

function onehot_kmer(kmer::String)
    kmer = split(kmer, "")
    return vcat(Flux.onehotbatch(kmer, ["A", "T", "G", "C"])...)
end

function split_kmer_data(kmers::Array{Bool, 2}, counts::Array{Float64, 1}, split_indice::Int64)
    if split_indice > 100 || split_indice < 1
        return
    end
    split_index = (length(counts)*split_indice)÷100
    return kmers[:, 1:split_index], counts[1:split_index], kmers[:, split_index+1:end], counts[split_index+1:end]
end

function split_kmer_data(kmers::Array{Bool, 2}, counts::Array{Int32, 1}, split_indice::Int64)
    if split_indice > 100 || split_indice < 1
        return
    end
    split_index = (length(counts)*split_indice)÷100
    return kmers[:, 1:split_index], counts[1:split_index], kmers[:, split_index+1:end], counts[split_index+1:end]
end

function kmer_to_hdf5(kmer_file::String, output_file::String, dataset_name::String; min_count::Int64=0, max_kmers::Int64=-1)
    kmers, counts = parse_kmer_count(kmer_file, min_count=min_count, max_kmers=max_kmers)
    # println(hcat(kmers...)[1])
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

# ↓ About 8.5 hours. Parses then saves the entire dataset with onehot encoded kmers in a hdf5 file.
# ↓ GC time is over 73%, this needs to be dealt with.
# @time kmer_to_hdf5("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_min-5.FASTA", 
#                    "/home/golem/rpool/scratch/jacquinn/data/13H107-k31.h5", 
#                    "13H107-k31_min-5_ALL")


