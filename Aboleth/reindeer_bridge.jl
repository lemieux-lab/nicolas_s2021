
function load_reindeer_index(reindeer_dir::String)
    ccall((:load_index, "/u/jacquinn/reindeer/REINDEER/reindeer.so"), 
                 Cvoid, (Cstring, ), reindeer_dir)
end

function query_on_loaded_index(kmer::String, query_receptacle::Array{Int32, 1})
    ccall((:query_on_loaded_index, "/u/jacquinn/reindeer/REINDEER/reindeer.so"), 
           Cvoid, (Cstring, Ptr{Cint}), kmer, query_receptacle)
end


# load_reindeer_index("/home/golem/scratch/jacquinn/data/reindeer_files/3_index/output_reindeer")
# to_build = Array{Int32, 1}(undef, 3)
# query_on_loaded_index("CAGGACTCCAATATAGAGATAAGTTAATGTC", to_build)
# println(to_build)

# function test_table(to_add::Int64, table::Array{Int32, 1})
#     ccall((:test_table, "/u/jacquinn/reindeer/REINDEER/reindeer.so"), 
#            Cvoid, (Cint, Ptr{Cint}), to_add, table)
# end

# table = Array{Int32, 1}(undef, 10)
# test_table(42, table)
# println(table)