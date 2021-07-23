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

function plot_accuracy(obtained, expected, subset, epoch)
    subset_indexes = rand(1:length(obtained), subset)
    sub_obtained = obtained[subset_indexes]
    sub_expected = expected[subset_indexes]
    graph = plot(x=sub_expected, y=sub_obtained, 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"),
            Guide.title("Accuracy at epoch $(epoch-1)"), Theme(panel_fill="white"),
            Geom.point)
    return graph
end

function plot_hex_accuracy(obtained, expected, subset, epoch)
    subset_indexes = rand(1:length(obtained), subset)
    sub_obtained = obtained[subset_indexes]
    sub_expected = expected[subset_indexes]
    graph = plot(x=sub_expected, y=sub_obtained, 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"),
            Guide.title("Accuracy at epoch $(epoch-1)"), Theme(panel_fill="white"),
            Geom.hexbin, Geom.abline)
    return graph
end