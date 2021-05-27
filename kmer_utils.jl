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
                # push!(kmers, matrix_to_string(hcat([col for col in eachcol(onehot_kmer(stripped_kmer))]...)))
                
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

function matrix_to_string(matrix::Array{Bool, 2})
    to_return = ""
    for row in eachrow(matrix)
        to_return*=join(row, " ")*";"
    end
    return to_return
end

# Encodes a kmer into a onehot matrix based on the 4 nucleotides
function onehot_kmer(kmer::String)
    return vcat(Flux.onehotbatch(split(kmer, ""), ["A", "T", "G", "C"])...)
end

function create_flux_sets(kmer_dict::Dict)
    kmers = Flux.OneHotArray[]
    counts = Int64[]
    for (kmer, count) in kmer_dict
        push!(kmers, onehot_kmer(kmer))
        push!(counts, count)
    end
    return kmers, counts
end

function parse_saved_onehot_kmers(encoded_kmers)
    m(x) = p.(encodedkmerstring_to_array(x))
    p(x) = parse.(Bool, x)
    return hcat(m.(encoded_kmers)...)
end

function create_flux_sets(kmer_df::DataFrame)
    # for (kmer, count) in ProgressBar(eachrow(kmer_df))
    #     push!(kmers, hcat([col for col in eachcol(onehot_kmer(kmer))]...))
    #     push!(counts, count)
    # end
    kmers = onehot_kmer.(kmer_df[!, "kmers"])
    counts = kmer_df[!, "counts"]
    return kmers, counts
end

function create_flux_sets_old(kmer_df::DataFrame)
    kmers = []
    counts = []
    for (kmer, count) in ProgressBar(eachrow(kmer_df))
        push!(kmers, hcat([col for col in eachcol(onehot_kmer(kmer))]...))
        push!(counts, count)
    end
    return kmers, counts
end

function split_kmer_df(df::DataFrame, split_indice::Int64)
    if split_indice > 100 || split_indice < 1
        return
    end
    split_index = (nrow(df)*split_indice)÷100
    return df[1:split_index, [:kmers, :counts]], df[split_index+1:nrow(df), [:kmers, :counts]]
end

function encodedkmerstring_to_array(encoded_kmer::String)
    tmp = (split.(split(encoded_kmer, ";"), " "))
    pop!(tmp)
    return tmp
end

# OUTDATED
# This generates a purely random kmer count dict. This is purely for testing input type in the kmer oracle
function generate_fake_kmer_counts(kmer_number::Int64; kmer_length::Int64=31, kmer_count_range::UnitRange=1:5000)
    nucs = ["A", "T", "G", "C"]
    to_return = Dict{String, Int32}()
    for i in 1:kmer_number
        kmer = join([nucs[rand(1:length(nucs))] for i in 1:kmer_length])
        to_return[kmer] = rand(kmer_count_range)
    end
    return to_return
end

# parsed_df = parse_kmer_count("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_min-5.FASTA")
# @time JDF.save("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_DataFrame_min-5_ALL.JDF", parsed_df)
