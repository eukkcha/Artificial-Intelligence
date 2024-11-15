[net, classes] = imagePretrainedNetwork("alexnet");

% View the layers of AlexNet and save the layers to a variable
layers = net.Layers

% Save the first and last layers of AlexNet
layers_input = layers(1)
layers_output = layers(end)
