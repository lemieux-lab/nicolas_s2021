using Flux
using ProgressBars
using JDF
using DataFrames

# Opens a kmer count dump file from jellyfish and parses it into a dataframe
function parse_kmer_count(path::String; min_count::Int64=0, max_lines::Int64=-1)
    open(path) do file
        kmers = String[]
        counts = Int32[]
        buffer = ""
        for (i, line) in ProgressBar(enumerate(eachline(file)))
            if max_lines != -1 && i > max_lines
                break
            end
            if i%2 == 0
                stripped_kmer = String(strip(line))
                push!(kmers, stripped_kmer)
                # push!(kmers, onehot_kmer(stripped_kmer))
                
            else
                count =  parse(Int32, line[2:length(line)])
                if count < min_count
                    continue
                end
                push!(counts, count)
            end
        end
        println(typeof(kmers))
        return DataFrame(kmers=kmers, counts=counts)
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

# parsed_df = parse_kmer_count("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_min-5.FASTA", max_lines = 300000)
# @time JDF.save("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_DataFrame_min-5_300k.JDF", parsed_df)


