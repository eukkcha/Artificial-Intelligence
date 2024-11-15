%% Overview of Pretrained Convolutional Neural Networks
% Load AlexNet into a network variable in MATLAB

[net, classes] = imagePretrainedNetwork("alexnet"); % Error? Then, install 'Deep Learning Toolbox Model for AlexNet Network' in Add-Ons Explorer

% View the layers of AlexNet and save the layers to a variable
layers = net.Layers

% Save the first and last layers of AlexNet
layers_input = layers(1)
layers_output = layers(end)

%% Preprocessing an Image
% Import an image and resize it to be compatible with AlexNet

% Display an image
image_original = imread("harper.jpg");
imshow(image_original)

% View the input size of AlexNet
layers_input.InputSize

% Resize the image so that it can be applied to AlexNet
image_resized = imresize(image_original, [227, 227])
imshow(image_resized)

% Classify the image and see its score
im = single(image_resized)
scores = minibatchpredict(net, im)
[testPred, score] = scores2label(scores, classes)
