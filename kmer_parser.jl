using Flux
using CUDA
using ProgressBars
using FileIO

# Opens a kmer count dump file from jellyfish and parses it into a dictionnary
# Around an hour for the entire 13H107 file, around 17min if prefiltering with only count > 5
function parse_kmer_count(path::String; min_count::Int64=0, max_lines::Int64=-1)
    open(path) do file
        data = Dict{String, Int32}()
        buffer = ""
        for (i, line) in ProgressBar(enumerate(eachline(file)))
            if i%2 == 0
                if buffer < min_count
                    continue
                end
                data[strip(line)] = buffer
            else
                if max_lines != -1 && i > max_lines
                    break
                end
                buffer = parse(Int32, line[2:length(line)])
            end
        end
        return data
    end
end

# Encodes a kmer into a onehot matrix based on the 4 nucleotides
function onehot_kmer(kmer::String)
    return Flux.onehotbatch(split(kmer, ""), ["A", "T", "G", "C"])
end


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


# println(onehot_kmer("ATTGTACCGTATGTAAACCGT"))
# println(generate_fake_kmer_counts(10))

# Saving large julia structures on the disk is INSANELY SLOW... It looks like it gets exponentially slower as it goes...
# save("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_JuliaDict_min-5.jld2", parse_kmer_count("/home/golem/rpool/scratch/jacquinn/data/13H107-k31_min-5.FASTA"))
