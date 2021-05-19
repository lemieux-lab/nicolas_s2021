#=
This is a small and **very** simple model of a neural net built with Flux
I made it to get familiar with Flux, so I can avoid basic mistakes when
implementing the actual thing for the internship.
This neural net is trained to take 3 numbers transformed by a log, an exp and a sin
and return the original number (yes it's useless, but I still love it <3)

Stuff I should remember when implementing the real thing:
    - ADAM is much yes
    - The Flux.train! thing is broken. Make the loop by hand (I kinda have to anyway for data recording)
    - DataLoader's batchsize means how many elements from each list it takes at each iteration
    - Use ProgressBar on the main training loop
    - You can apply a loss function to all the data with loss.()
    - While Flux.update! only happens on the training data, both training and testing sets get their loss recorded during training
    - When applying the loss function in the gradient part, make sure the data structure your giving loss is what it expects
=#    


using Flux
using Gadfly
using ProgressBars
# using CUDA

# Generates sets of random data for training
function generate_set(set_length = 1000)
    set_in = []
    set_expected_out = []
    for i in 1:set_length
        gen = rand()
        push!(set_in, [log(gen), exp(gen), sin(gen)])
        push!(set_expected_out, gen)
    end
    return set_in, set_expected_out
end

# A function that returns a neural net that takes 3 in, 1 out
function neural_network()
    return Chain(
            Dense(3, 25, relu),
            Dense(25, 25, relu),
            Dense(25, 1, x->σ.(x))
            )
end

# Generating training & testing sets
training_in, training_expected_out = generate_set()
testing_in, testing_expected_out = generate_set()

# Initialising neural net, and everything needed to train it
goblin = neural_network()
loss(results, expected) = Flux.mse(goblin(results), expected)
data = Flux.Data.DataLoader((training_in, training_expected_out), batchsize=1, shuffle=true)
opt = Flux.ADAM()

# Empty lists that will contain loss values over iterations (for plotting)
losses = []
testing_losses = []

# Main training loop that
epochs = 200
for i in ProgressBar(1:epochs)
    for d in data
        gs = gradient(params(goblin)) do
            loss(d[1][1], d[2][1])
        end 
        Flux.update!(opt, params(goblin), gs)
    end
    push!(losses, sum(loss.(training_in, training_expected_out)))
    push!(testing_losses, sum(loss.(testing_in, testing_expected_out)))
end

# Running the whole testing set through the final neural net 
# To get results to compare to the expected ones
testing_out = [goblin(input) for input in testing_in]

# Making a plot to visualise results
set_default_plot_size(30cm, 30cm)
vstack(
plot(layer(x=1:length(losses), y=losses, Geom.line, Guide.xlabel("Iter"), Guide.ylabel("loss")),
     layer(x=1:length(testing_losses), y=testing_losses, Geom.line, Theme(default_color=color("orange")))),
plot(x=hcat(testing_expected_out...), y=hcat(testing_out...), Geom.point, Guide.xlabel("expected"), Guide.ylabel("obtained")))
