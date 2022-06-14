using ArgParse
using GZip
using ProgressBars
using Dates
import Random

# Warning
# Currently set up to write to actual files instead of FIFOS.
# fifo mode is in comments right now.

arg_parser = ArgParseSettings(allow_ambiguous_opts=true)
@add_arg_table arg_parser begin
    "--fof_r1", "-1"
        help = "file containing paths to r1"
        required = true
    "--temp_folder"
        help = "temporary directory path path"
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

function prep_fifos(parsed_args)
    out_rep = parsed_args["out"] * "/samples"
    temp_folder = parsed_args["temp_folder"] * "/samples"
    rm(out_rep, force=true, recursive=true)
    rm(temp_folder, force=true, recursive=true)
    mkdir(out_rep)
    mkdir(temp_folder)

    # probs = [2^i for i in 1:7]
    probs = [2]
    probs_files = Dict(p => [Array{String, 1}(), Array{String, 1}()] for p in probs)
    sample_matrice = Array{Array{String, 1}, 1}() => Array{Array{String, 1}, 1}()
    push!(sample_matrice[1], Array{String, 1}())
    push!(sample_matrice[2], Array{String, 1}())
    patient = ""
    salmon_commands = Array{Cmd, 1}()

    r1 = readlines(parsed_args["fof_r1"])
    iter = ProgressBar(enumerate(r1))
    set_description(iter, "preparing fifos...")
    for (s, sample_r1) in iter
        sample_r2 = replace(sample_r1, "R1" => "R2")
        i = size(sample_matrice[1])[1]
        push!(sample_matrice[1][i], sample_r1)
        push!(sample_matrice[2][i], sample_r2)

        r1_name = replace(sample_r1[findall("/", sample_r1)[end-1][1]+1:findlast(".", sample_r1)[1]-1], "/"=>"-")
        r2_name = replace(sample_r2[findall("/", sample_r2)[end-1][1]+1:findlast(".", sample_r2)[1]-1], "/"=>"-")

        cur_patient = match(r"[0-9]{2}H[0-9]{3}", r1_name).match

        if s == 1
            patient = cur_patient
        end

        # Processes with Salmon the previous patient when switching to another one
        if cur_patient != patient
            mkdir("$out_rep/$patient")

            # Going through all the subsets for that patient
            for (p, paths) in probs_files
                f_path = "$out_rep/$patient/$p"
                mkdir(f_path)
                left, right = join(paths[1], " "), join(paths[2], " ")

                # We're ready to start salmon with all the files from that subset
                # println(left, right, f_path)
                push!(salmon_commands, (`sh start_salmon.sh $left $right $f_path  \&`))
            end

            # Reseting path dict for next patient   
            probs_files = Dict(p => [Array{String, 1}(), Array{String, 1}()] for p in probs)
            patient = cur_patient
            push!(sample_matrice[1], Array{String, 1}())
            push!(sample_matrice[2], Array{String, 1}())
        end

        for p in probs
            # run(`mkfifo $temp_folder/$p\%-$r1_name`)
            run(`touch $temp_folder/$p\%-$r1_name`)
            push!(probs_files[p][1], "$temp_folder/$p%-$r1_name")
            
            # run(`mkfifo $temp_folder/$p\%-$r2_name`)
            run(`touch $temp_folder/$p\%-$r2_name`)
            push!(probs_files[p][2], "$temp_folder/$p%-$r2_name")

        end
    end

    mkdir("$out_rep/$patient")

    # Going through all the subsets for that patient
    for (p, paths) in probs_files
        f_path = "$out_rep/$patient/$p"
        mkdir(f_path)
        left, right = join(paths[1], " "), join(paths[2], " ")
        to_delete = left * right

        # We're ready to start salmon with all the files from that subset
        # println(right)
        push!(salmon_commands, (`sh start_salmon.sh $left $right $f_path $to_delete\&`))
    end
    return sample_matrice, salmon_commands
end

function start_extraction(sample_matrice::Pair{Vector{Vector{String}}, Vector{Vector{String}}}, salmon_commands::Vector{Cmd}, parsed_args)
    probs = [2]
    out_rep = parsed_args["out"] * "/samples"
    temp_folder = parsed_args["temp_folder"] * "/samples"
    for (s, (samples_r1, samples_r2)) in enumerate(zip(sample_matrice[1], sample_matrice[2]))
        to_iter = vcat(samples_r1, samples_r2)
        # println(to_iter)
        # run(salmon_commands[s])
        for (sample) in to_iter

            # sample_r2 = replace(sample_r1, "R1" => "R2")
            name = replace(sample[findall("/", sample)[end-1][1]+1:findlast(".", sample)[1]-1], "/"=>"-")
            # r2_name = replace(sample_r2[findall("/", sample_r2)[end-1][1]+1:findlast(".", sample_r2)[1]-1], "/"=>"-")

            cur = ""
            # probs_total_left = Dict(p => "" for p in probs)
            # probs_total_right = Dict(p => "" for p in probs)
            # probs_total_left = ["" for p in probs]
            # probs_total_right = ["" for p in probs]

            # set_description(iter, "Going through $r1_name and $r2_name")
            content = readlines(GZip.open(sample))
            # println(r1_content[12:-1])
            # print(length(r1_content))
            # sleep(20)
            grouped = Array{String, 1}(undef, Int(length(content)/4))
            # right = Array{String, 1}(undef, Int(length(r2_content)/4))
            for (i, line) in ProgressBar(enumerate(content))
                cur=quick_concat(cur, line)
                if i%4 == 0
                    grouped[Int64(i/4)] = cur
                    cur = ""
                end
            end
            # println(length(tmp))
            # println(tmp)
            # sleep(20)
            # println("[RNA_Sampler_2.0]: Done with $r1_name. Starting to write to named pipes...")
            # for (j, p) in enumerate(probs)
            #     open("$out_rep/$p%-$r1_name", "w+") do io
            #         write(io, probs_total_left[j])
            #     end
            #     open("$out_rep/$p%-$r2_name", "w+") do io
            #         write(io, probs_total_right[j])
            #     end
            # end
            for p in probs
                Random.shuffle!(grouped)
                tmp = join(grouped[1:Int64(round(length(grouped)*(p/100)))], "\n")
                io = open("$temp_folder/$p%-$name", "w+")
                println("opened")
                write(io, tmp)
                println("written")
                close(io)
                println("closed")
                
                # open("$out_rep/$p%-$name", "w+") do io
                    # write(io, tmp)
                # end
            end
        end
        run(salmon_commands[s])
    end
end

start_extraction(prep_fifos(parsed_args)..., parsed_args)
# prep_fifos(parsed_args)