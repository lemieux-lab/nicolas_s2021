Base.@kwdef mutable struct Datareader
    datafile::Datafile
    open_files::Array{IOStream, 1}=init_files(datafile)
    current_file::Int64=1
    current_line::Int64=1
    load_interval::Int64=8182
end

function load_next_batch(reader::Datareader)
    counter = 0
    kmers = Array{String, 1}(undef, reader.load_interval÷2)
    counts = Array{Any, 1}(undef, reader.load_interval÷2)
    for line in eachline(reader.open_files[reader.current_file])
        if counter % 2 != 0
            kmers[counter÷2+1] = line
        else
            counts[counter÷2+1] = parse(Int32, strip(line, '>'))
        end
        counter += 1
        if counter >= reader.load_interval
            break
        end
    end
    if counter < reader.load_interval
        kmers = [kmers[i] for i in 1:length(kmers) if isassigned(kmers, i)]
        counts = [counts[i] for i in 1:length(counts) if isassigned(counts, i)]

        if reader.current_file < length(reader.open_files)
            reader.current_file += 1
        else
            return (false, kmers, convert(Array{Int32}, counts))
        end
    end
    return (true, kmers, convert(Array{Int32}, counts))
end

function init_files(datafile::Datafile)
    streams = Array{IOStream, 1}(undef, length(datafile.paths))
    for (i, path) in enumerate(datafile.paths)
        streams[i] = open(path)
    end
    return streams
end