%%     1. Transfer Learning
%% Overview of Pretrained Convolutional Neural Networks
% Load AlexNet into a network variable in MATLAB

[net, classes] = imagePretrainedNetwork("alexnet")    % Error? Then, install 'Deep Learning Toolbox Model for AlexNet Network' in Add-Ons Explorer
%% 
% Peek into AlexNet

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
%% Image Datastore
% When you import large data, you don't have to save them as variables because 
% it costs too much time and memory. 
% 
% Instead, you can read image files by creating a datastore, which references 
% a data source such as a folder of image files. When you create a datastore, 
% basic information such as the file name and formats is stored.

% Create an image datastore referring to 'Lucy'
imageDS = imageDatastore("/Users/eukkcha/Desktop/Sejong/AI/petImages/Lucy",...
    "IncludeSubfolders",true,"LabelSource","foldernames")
% Read and display 'Lucy' images
montage(imageDS)
%% 
% For the images in an image datastore, basic preprocessing can be done by |augmentedImageDatastore| 
% function

augImageDS = augmentedImageDatastore([227, 227],imageDS)
%% Modifying a Pretrained Network
% An existing pretrained network does not ouput the classes you want. When performing 
% transfer learning, you will typically change the fully connected layer and the 
% classification layer to suit your specific application. 

clear    % clear
% Load AlexNet and view the layers
net = imagePretrainedNetwork("alexnet")
layers = net.Layers
%% 
% Change the number of output classes by modifying the fully conneceted layer(23).

layers(23) = fullyConnectedLayer(14);
%% Preparing Training Data

% Create an image datastore with the images labeled by their folder name
imageDS = imageDatastore("/Users/eukkcha/Desktop/Sejong/AI/petImages",...
    "IncludeSubfolders",true,"LabelSource","foldernames") % IncludeSubfolders=true,LabelSource="foldernames")
classNames = categories(imageDS.Labels)

% Split the data into training and test data sets
[trainImages, testImages] = splitEachLabel(imageDS, 0.8)

% Create augmented datastores to resize the images so that they can be applied to AlexNet
trainData = augmentedImageDatastore([227, 227], trainImages)
testData = augmentedImageDatastore([227, 227], testImages)
%% Training Options
% Use Stochastic Gradient Descent with Momentum (SGDM) and decrease the initial 
% learning rate to 0.0001.
% 
% The default initial learning rate is 0.01 but generally it should be decreased 
% for better performances in a transfer learning.

opts = trainingOptions("sgdm","InitialLearnRate",0.0001,...
    "Plots","training-progress","Metrics","accuracy")
%% Training and Evaluating the Network
% Train a new network. In order to use a GPU instead of a CPU, install 'Parallel 
% Computing Toolbox' in Add-Ons explorer. A GPU is much faster than a CPU in training 
% a neural network.

newnet = trainnet(trainData,layers,"crossentropy",opts)
%% 
% Evaluate the network by classifying the test data

scores = minibatchpredict(newnet,testData)    % Prediction by the new network
[testPred,score] = scores2label(scores,classNames)

testGT = testImages.Labels    % Ground truth of the test data
testAcc = sum(testGT == testPred) / numel(testGT)    % Prediction Accuracy 
% 또는 nnz(testGT == testPred) / numel(testGT)
confusionchart(testGT,testPred)    % Visualization in a confusion matrix
