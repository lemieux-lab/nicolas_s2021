using Gadfly
using Cairo

function plot_loss(train_losses::Vector{Float32}, test_losses::Vector{Float32})
    df = DataFrame(train = train_losses, test = test_losses, epoch=0:length(train_losses)-1)
    graph = plot(stack(df, [:train, :test]), x=:epoch, y=:value, 
            color=:variable, Guide.xlabel("Epoch"), Guide.ylabel("Loss"), 
            Guide.title("Loss per epoch"), Theme(panel_fill="white"), Geom.line)
    return graph
end

function plot_loss_with_l2(train_losses::Vector{Float32}, test_losses::Vector{Float32}, l2_values::Vector{Float32})
    df = DataFrame(train = train_losses, test = test_losses, 
                   l2 = l2_values, epoch=0:length(train_losses)-1)
    # println(df)
    df = stack(df, [:train, :test])
    rename!(df, Dict(:variable => "set", :value => "loss"))
    df = stack(df, [:l2, :loss])
    df[df.variable.== "l2", "set"].="l2"
    # df now has a "variable" column for l2 & loss, and a "set" color for train, loss & L2
    graph = plot(df, ygroup = "variable", x="epoch", y="value", color="set", 
                 Geom.subplot_grid(Geom.line, free_y_axis=true),
                 Guide.title("Loss & L2 per epoch"), Theme(panel_fill="white"))
    return graph
end

function plot_accuracy(obtained, expected, epoch)
    sample_names = ["s$i" for i in 1:(size(obtained)[2])]
    df_obtained = stack(DataFrame(obtained, sample_names), sample_names)
    df_expected = stack(DataFrame(expected, sample_names), sample_names)
    rename!(df_obtained, Dict(:variable => "sample", :value => "obtained"))
    rename!(df_expected, Dict(:variable => "sample", :value => "expected"))
    to_plot = df_obtained
    to_plot[!, "expected"] = df_expected[!, "expected"]
    graph = plot(to_plot, x=:expected, y=:obtained, 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"),
            Guide.title("Accuracy at epoch $(epoch-1)"), Theme(panel_fill="white"),
            color=:sample, Geom.point)
    return graph
end

function plot_hex_accuracy(obtained, expected, epoch)
    sample_names = ["s$i" for i in 1:(size(obtained)[2])]
    df_obtained = stack(DataFrame(obtained, sample_names), sample_names)
    df_expected = stack(DataFrame(expected, sample_names), sample_names)
    rename!(df_obtained, Dict(:variable => "sample", :value => "obtained"))
    rename!(df_expected, Dict(:variable => "sample", :value => "expected"))
    to_plot = df_obtained
    to_plot[!, "expected"] = df_expected[!, "expected"]
    graph = plot(to_plot, x=:expected, y=:obtained, 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"),
            Guide.title("Accuracy at epoch $(epoch-1)"), Theme(panel_fill="white"),
            color=:sample, Geom.hexbin, Geom.abline)
    return graph
end