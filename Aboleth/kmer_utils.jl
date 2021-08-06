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

# New and improved onehot_kmer (about 12 times faster)
function onehot_kmer(kmer::String, kmer_length::Int64=31)
    # hot_kmer = Array{Bool, 2}(undef, 4, kmer_length)
    hot_kmer = fill(false, kmer_length, 4)
    encode_table = Dict(
        'A' => 1,
        'T' => 2,
        'G' => 3,
        'C' => 4
    )
    for (i, nuc) in enumerate(kmer)
        hot_kmer[i, encode_table[nuc]] = true
    end
    return hot_kmer
end

function onehot_idxkmer(kmer::String, kmer_length::Int64=31)
    hot_idxkmer = Array{Int8, 1}(undef, kmer_length)
    encode_table = Dict(
        'A' => 1,
        'T' => 2,
        'G' => 3,
        'C' => 4
    )
    for (i, nuc) in enumerate(kmer)
        hot_idxkmer[i] = encode_table[nuc]
    end
    return hot_idxkmer
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

function get_random_kmers(set_length::Int64=100, k::Int64=31)
    nucs = ["A", "T", "G", "C"]
    return [join([nucs[rand(1:length(nucs))] for j in 1:k]) for i in 1:set_length]
end