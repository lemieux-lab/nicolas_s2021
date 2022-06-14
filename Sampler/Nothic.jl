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
    # iter = ProgressBar(samples)
    # set_description(iter, "Reading and grouping data for patient $patient_id...")
    all = Array{Pair, 1}()

    for sample_r1 in samples
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

        iter = ProgressBar(enumerate(zip(content_r1, content_r2)), leave=false)
        set_description(iter, "Grouping reads...")

        for (i, (line_r1, line_r2)) in iter
            cur_r1=quick_concat(cur_r1, line_r1)
            cur_r2=quick_concat(cur_r2, line_r2)
            if i%4 == 0
                grouped_r1[Int64(i/4)] = cur_r1
                grouped_r2[Int64(i/4)] = cur_r2
                cur_r1 = ""
                cur_r2 = ""
            end
        end

        push!(all, grouped_r1 => grouped_r2)
    end
    println("Finished reading and grouping reads for $patient_id.\n\n")
    return all
end

function get_patient_ids(user_args)
    samples = readlines(user_args["fof"])
    iter = ProgressBar(enumerate(samples))
    status="Preparing"; current="None"
    set_multiline_postfix(iter, "Status: $status\nCurrent File: $current")
    set_description(iter, "Getting patient ids...")
    patient_ids = Array{String, 1}()

    for (i, sample) in iter
        status="Getting ID";current=sample
        patient_id = match(r"[0-9]{2}H[0-9]{3}", sample).match
        if !(patient_id in patient_ids)
            status="ID not encountered yet"
            push!(patient_ids, patient_id)
        end

    end

    println("Finished extracting patient ids.\n\n")
    return patient_ids
end

function shuffle_and_extract(patient_data, user_args)
    probs = [1, 2, 5, 10, 20, 100, 250, 750, 1000]  # Sample per thousands
end

patient_ids = get_patient_ids(parsed_args)
for patient_id in patient_ids
    patient_data = read_patient(patient_id, parsed_args)
    println(sizeof(patient_data))
end
