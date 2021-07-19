using Flux
using CUDA
using ProgressBars
using Gadfly
using Cairo
using Statistics
using DataFrames

CUDA.allowscalar(false)

# Generates dummy data for testing
function generate_fake_data(size::Int64=10000)
    kmers = Array{Float32, 3}(undef, 31, 4, size)
    values = Array{Float64, 1}(undef, size)
    iter = ProgressBar(1:size)
    set_description(iter, "Generating data...")

    for i in iter
        to_add = convert(Array{Float32, 2}, rand(0:1, 31, 4))
        f(x) = x==1.0
        value = count(f, to_add[:, 1]) - count(f, to_add[:, 2]) + count(f, to_add[:, 3]) - + count(f, to_add[:, 4])
        kmers[:,:,i] = to_add
        values[i] = value
    end
    return (kmers, values)
end

function evaluate_network(training_set, testing_set; device::Function=cpu)
    training_accuracy = Float32[]
    training_loss = Float32[]
    iter = ProgressBar(training_set)
    set_description(iter, "Evaluating network (training set)...")
    for (i, e) in iter
        # println(i)
        i, e = device(i), device(e)
        push!(training_accuracy, cpu(network(i))...)
        push!(training_loss, cpu(loss(i, e))...)
    end

    testing_accuracy = Float32[]
    testing_loss = Float32[]
    iter = ProgressBar(training_set)
    set_description(iter, "Evaluating network (testing set)...")
    for (i, e) in iter
        i, e = device(i), device(e)
        push!(testing_accuracy, cpu(network(i))...)
        push!(testing_loss, cpu(loss(i, e))...)
    end
    push!(training_losses, mean(training_loss))
    push!(testing_losses, mean(testing_loss))
    push!(l2_values, cpu(l2_penalty()))

    return training_accuracy, testing_accuracy
end

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

function simplify(tensor, batchsize=1024)
    # println("Entering simplification")
    return reshape(tensor, (100, batchsize))
    # println("Exiting simplification")
    # return tmp
end

function neural_network(k::Int64=31)
    return Chain(
        Conv((31,), 4=>100, relu),
        simplify,
        # MaxPool((2, 2)),
        Dense(100, 1, identity)
    )
end

function l2_penalty()
    penalty = 0
    for layer in network
        layer = cpu(layer)
        if typeof(layer) != typeof(simplify)
            penalty += sum(abs2, layer.weight)
        end
    end
    return penalty
end

# loss(input, expected) = Flux.Losses.mse(convert(Array{Float64}, reshape(network(input), 1024)), expected) + (l2_penalty() * 1e-5)
loss(input, expected) = Flux.Losses.mse(network(input), expected) + (l2_penalty() * 1e-5)

tests = generate_fake_data(1024*5)

training_set = Flux.DataLoader(generate_fake_data(1024*30), batchsize=1024, shuffle=true)
testing_set = Flux.DataLoader(tests, batchsize=1024)

training_losses = Float32[]
testing_losses = Float32[]
l2_values = Float32[]
training_accuracy = Float32[]
testing_accuracy = Float32[]
device = cpu

network = neural_network() |> device
opt = Flux.ADAM()
ps = params(network)

epoch = 2
for t in ProgressBar(1:epoch)
    accurs =  evaluate_network(training_set, testing_set, device=device)
    global training_accuracy = accurs[1]
    global testing_accuracy = accurs[2]

    for (i, e) in training_set
        # println(typeof(e))
        # e = reshape(e, 1, 1024)
        i, e = device(i), device(e)
        gs = gradient(ps) do
            loss(i, e)
        end
        # println("Done with gradient")
        Flux.update!(opt, ps, gs)
    end
end
        
# loss_graph = plot_loss(training_losses, testing_losses)
loss_graph = plot_loss_with_l2(training_losses, testing_losses, l2_values)
# println(loss_graph)
draw(SVG("Flux Convolution Testing Loss.svg"), loss_graph)
println(length(testing_accuracy))
println(length(tests[2]))
accur_graph = plot_hex_accuracy(testing_accuracy, tests[2], 2048*2, 200)
draw(SVG("Flux Convolution Testing Accuracy.svg"), accur_graph)
