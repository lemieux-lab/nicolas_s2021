
function load_reindeer_index(reindeer_dir::String)
    ccall((:load_index, "/u/jacquinn/reindeer/REINDEER/reindeer.so"), 
                 Cvoid, (Cstring, ), reindeer_dir)
end

function query_on_loaded_index(query_path::String)
    ccall((:query_on_loaded_index, "/u/jacquinn/reindeer/REINDEER/reindeer.so"), 
           Cvoid, (Cstring, ), query_path)
end