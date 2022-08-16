using ArgParse
using GZip
using ProgressMeter
using Dates
import Random

# Probs in %% for a fragment to be in each subset
# probs = [64, 128, 216, 512]
# probs = [1, 2, 4, 8, 16, 32, 64, 128, 216, 512, 1000]
# probs = [100]
# probs = [1, 2, 5, 10, 50]
probs = [1, 2, 4, 8, 16, 32, 1000]

arg_parser = ArgParseSettings()
@add_arg_table arg_parser begin
    "--fof", "-f"
        help = "file containing paths to r1."
        required = true
    "--force_id", "-i"
        help = "force a specific run id for file generation purposes."
        default = nothing
    "--record_perfs", "-p"
        help = "records performances of code."
        default = false
end
parsed_args = parse_args(ARGS, arg_parser)

struct Environnement
    run_id::String
    temp_files_dir::String
    salmon_dir::String
end

function prepare_environnement()
    if parsed_args["force_id"] != nothing
        run_id = parsed_args["force_id"]
    else
        run_id = string(rand(1_000_000:999_999_999))
    end
    println("Preparing environnement with run id: $run_id...")
    mkdir("./RNA_Sampler_$run_id")
    mkdir("./RNA_Sampler_$run_id/temp_files")
    mkdir("./RNA_Sampler_$run_id/salmon")
    return Environnement(run_id,"./RNA_Sampler_$run_id/temp_files", "./RNA_Sampler_$run_id/salmon")
end

function make_io_streams(env)
    println("Preparing IO streams...")
    fof = open(parsed_args["fof"], "r")
    # samples = Dict{String, [Array{String, 1}, Array{String, 1}]}()
    samples = Dict{String, Any}()
    # salmon_commands = Array{Cmd, 1}() # Use for make_temp_files instead
    patient = nothing
    
    for (i, sample_r1) in enumerate(eachline(fof))
        sample_r2 = replace(sample_r1, "R1" => "R2")
        r1_name = replace(sample_r1[findall("/", sample_r1)[end-1][1]+1:findlast(".", sample_r1)[1]-1], "/"=>"-")
        r2_name = replace(sample_r2[findall("/", sample_r2)[end-1][1]+1:findlast(".", sample_r2)[1]-1], "/"=>"-")
        cur_patient = match(r"[0-9]{2}H[0-9]{3}", r1_name).match
        # Testing if both files exists
        if !isfile(sample_r1) || !isfile(sample_r2)
            printstyled("WARNING: $r1_name or $r2_name (or both) are missing. Ignoring those...\n"; color = :red)
            continue
        end
        println("Processing environnement for $r1_name and its pair...")

        if patient != cur_patient
            salmon_out_path = env.salmon_dir
            mkdir("$salmon_out_path/$cur_patient")
            for p in probs
                mkdir("$salmon_out_path/$cur_patient/$p")
            end
            if !haskey(samples, cur_patient)
                # samples[cur_patient] = Array{GZipStream, 1}() => Array{GZipStream, 1}()
                samples[cur_patient] = Array{String, 1}() => Array{String, 1}()
            end
            patient = cur_patient
        end

        # println(samples[patient])
        push!(samples[patient][1], sample_r1)
        push!(samples[patient][2], sample_r2)
    end
    return samples
end

function process_patient(patient, rna, env)
    out_dir = env.salmon_dir*"/$patient"
    # mkdir(out_dir)
    left_temp_files = Array{String, 2}(undef, length(probs), length(rna[1]))
    right_temp_files = Array{String, 2}(undef, length(probs), length(rna[2]))

    # Generate temp_files
    for (i, p) in enumerate(probs)
        for (j, (left, right)) in enumerate(zip(rna[1], rna[2]))
            path_l = env.temp_files_dir*"/temp_file_$(patient)_$(j)_$(p)_left.fastq"
            path_r = env.temp_files_dir*"/temp_file_$(patient)_$(j)_$(p)_right.fastq"
            # run(`mktemp_file $path_l`)
            # run(`mktemp_file $path_r`)
            run(`touch $path_l`)
            run(`touch $path_r`)
            left_temp_files[i,j] = path_l
            right_temp_files[i,j] = path_r
            # push!(left_temp_files, path_l)
            # push!(right_temp_files, path_r)
        end 
    end

    # # Start salmon
    # for (p, (l_row, r_row)) in zip(probs, zip(eachrow(left_temp_files), eachrow(right_temp_files)))
    #     p_dir = out_dir*"/$p"
    #     # println(l_row)
    #     # return
    #     left, right = join(l_row, " "), join(r_row, " ")
    #     run(`sh ../start_salmon.sh $left $right $p_dir \&`)
    # end

    # Opens all IO streams
    left_io_temp_files = [open(f, "w+") for f in left_temp_files]
    right_io_temp_files = [open(f, "w+") for f in right_temp_files]

    # Start drip sampling
    iter_nb = 0
    prog = ProgressUnknown("Drip Sampling...", spinner=true; showspeed=true)
    while length(rna[1]) != 0
        iter_nb+=1
        # println(iter_nb)
        ProgressMeter.next!(prog, spinner="🌑🌒🌓🌔🌕🌖🌗🌘"; showvalues=[(:iter_nb, iter_nb)])

        pass_list = Array{Bool, 2}(undef, size(left_io_temp_files, 1), length(rna[1]))
        # for each file i and each subset j of that file, process reads
        # while maintaining pairship in left and right files.
        for (i, io_stream) in enumerate(rna[1])
            if eof(io_stream)
                deleteat!(rna[1], i)
                deleteat!(rna[2], i)
                continue
            end
            fragment = join([readline(io_stream) for i in 1:4], "")

            for (j, (p, row)) in enumerate(zip(probs, eachrow(left_io_temp_files)))
                # println(j, "\n", p, "\n", row)
                if rand(1:1000) > p  # Skipping based on RNG
                    # println("PASSED")
                    pass_list[j, i] = false
                    # println("DIDN'T PASS")
                    continue
                end
                pass_list[j, i] = true
                write(row[i], fragment)
                flush(row[i])
                # println(left_io_temp_files[j][i])
            end
        end
        # println(pass_list)
        for (i, io_stream) in enumerate(rna[2])
            fragment = join([readline(io_stream) for i in 1:4], "")
            for (b, row) in zip(eachrow(pass_list), eachrow(right_io_temp_files))
                # println(b)
                if !b[i]
                    continue
                end
                write(row[i], fragment)
                flush(row[i])
                # println("written right")
            end
        end
    end

    # Closing temp files
    for (l, r) in zip(left_io_temp_files, right_io_temp_files)
        close(l); close(r)
    end

    # Start salmon
    for (p, (l_row, r_row)) in zip(probs, zip(eachrow(left_temp_files), eachrow(right_temp_files)))
        p_dir = out_dir*"/$p"
        left, right = join(l_row, " "), join(r_row, " ")
        try
            run(`sh ../start_salmon.sh $left $right $p_dir \&`)  # COMMENT FOR TESTING
        catch e
            println("WARNING: Salmon run for $out_dir failed with $e")
        end
    end

end

function main(parsed_args)
    env = prepare_environnement()
    patients = make_io_streams(env)

    printstyled("\nFinished building working environnement. Press Enter when ready to begin...\n"; color = :green)
    readline();

    for (patient, rna) in patients
        printstyled("→ Now processing: $patient\n"; color = :cyan)
        opened_rna = [GZip.open(r1) for r1 in rna[1]] => [GZip.open(r2) for r2 in rna[2]]
        process_patient(patient, opened_rna, env)

        # Closing opened files
        for (r1, r2) in zip(opened_rna[1], opened_rna[2])
            GZip.close(r1); GZip.close(r2)
        end
    end
end

@time main(parse_args)
