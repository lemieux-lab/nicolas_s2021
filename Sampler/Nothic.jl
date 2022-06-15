using ArgParse
using GZip
using ProgressBars
using Dates
import Random

# TODO: Fix GC time on data collection
# TODO: Create mini-fifos network
# TODO: Integrate Salmon automation

arg_parser = ArgParseSettings(allow_ambiguous_opts=true)
@add_arg_table arg_parser begin
    "--fof", "-f"
        help = "file containing paths to r1"
        required = true
    "--temp", "-t"
        help = "location of temporary fifos, defaults to ./temp_fifos"
        default = "./temp_fifos"
    "--out", "-o"
        help = "output directory, defaults to the script's location"
        default = "."
end
parsed_args = parse_args(ARGS, arg_parser)

function quick_concat(a::String, b::String)
    buf = IOBuffer()
    print(buf, a);
    print(buf, b);
    return String(take!(buf))
end

function read_patient(patient_id, user_args)
    samples = readlines(user_args["fof"])
    iter = ProgressBar(enumerate(samples))
    set_description(iter, "Reading and grouping data for patient $patient_id...")
    all = Array{Pair, 1} #(undef, Int(length(samples/4)))

    for (i, sample_r1) in iter
        if !(occursin(patient_id, sample_r1))
            continue
        end
        println("Found data for $patient_id. Decompressing...")
        sample_r2 = replace(sample_r1, "R1" => "R2")
        content_r1 = readlines(GZip.open(sample_r1))
        content_r2 = readlines(GZip.open(sample_r2))

        grouped_r1 = Array{String, 1}(undef, Int(length(content_r1)/4))
        grouped_r2 = Array{String, 1}(undef, Int(length(content_r2)/4))
        cur_r1 = ""; cur_r2 =""

        iter2 = ProgressBar(enumerate(zip(content_r1, content_r2)), leave=false)
        set_description(iter, "Grouping reads...")

        for (j, (line_r1, line_r2)) in iter2
            cur_r1=quick_concat(cur_r1, line_r1)
            cur_r2=quick_concat(cur_r2, line_r2)
            if j%4 == 0
                @time grouped_r1[Int64(j/4)] = cur_r1
                grouped_r2[Int64(j/4)] = cur_r2
                cur_r1 = ""
                cur_r2 = ""
            end
        end

        push!(all, grouped_r1 => grouped_r2)
        # push!(grouped_r1 => grouped_r2
    end
    println("Finished reading and grouping reads for $patient_id.\n\n")
    return all
end

function create_fifo_network(probs, user_args)

    temp = user_args["temp"]
    rm(temp, force=true, recursive=true)
    mkdir(temp)
    samples = readlines(user_args["fof"])
    iter = ProgressBar(enumerate(samples))
    set_multiline_postfix(iter, "Status: Preparing...\nCurrent File: None")
    set_description(iter, "Getting patient ids...")
    fof_network = Dict{String, Array{String, 1}}()
    fifo_network = Dict{String, Dict{Int64, Array{String, 1}}}()

    for (i, sample) in iter
        status="Getting ID";current=sample
        set_multiline_postfix(iter, "Status: $status \nCurrent File: $current")
        patient_id = match(r"[0-9]{2}H[0-9]{3}", sample).match
        if !(haskey(fifo_network, patient_id))
            status="ID not encountered yet"
            set_multiline_postfix(iter, "Status: $status \nCurrent File: $current")
            # push!(patient_ids, patient_id)
            fifo_network[patient_id] = Dict{Int64, Array{String, 1}}()
            fof_network[patient_id] = Array{String, 1}()
        end
        push!(fof_network[patient_id], sample)
    end

    #= Knowing that the max fifo buffer size is 1MB, we will compute the size
    of the data and create a number of fifos so that only ~900kb of data go through each.
    We will unzip a normal file of the RNAseq results, and the last file (which is smaller) 
    to calculate the number of fragments.
    Knowing a fragment is about 258 bytes (round up to 300), we will only pass 3000
    fragments per named pipes before closing it.
    =#
    frag_per_pipe = 3000
    iter = ProgressBar(fof_network)
    set_description(iter, "Building fifo network...")
    set_multiline_postfix(iter, "max number of pipes: /\ncurrent subset: /\nrequired pipes: /\ncreated pipes: /" )
    for (patient_id, cur_samples) in iter
        filesizes = [filesize(f) for f in cur_samples]
        normal = cur_samples[findfirst(x->x==maximum(filesizes), filesizes)]
        last = cur_samples[findfirst(x->x==minimum(filesizes), filesizes)]
        n_frag_number = length(readlines(GZip.open(normal)))
        l_frag_number = length(readlines(GZip.open(last)))
        n_frag = n_frag_number * (length(cur_samples)-1) + l_frag_number
        n_pipes = n_frag / frag_per_pipe
        for p in probs
            np_pipes = n_pipes * (p/1000)
            fifo_network[patient_id][p] = Array{String, 1}()
            for n in range(1, np_pipes, step=1)
                set_multiline_postfix(iter, "max number of pipes: $n_pipes\ncurrent subset: $p%%\nrequired pipes: $np_pipes\ncreated pipes: $n")
                fn = "$temp/$p-$patient_id-$n"
                run(`mkfifo $fn`)
                push!(fifo_network[patient_id][p], fn)
            end
        end
    end

    println("Finished initating fifo network.\n\n")
    return fifo_network, fof_network
end

probs = [1, 2, 5, 10, 20, 100, 250, 750, 1000] # Fragment per thousand.
fifo_network, fof_network = create_fifo_network(probs, parsed_args)
println(fifo_network)