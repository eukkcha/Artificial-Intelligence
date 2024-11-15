%% 3. Creating Networks

load satData.mat % load the image and class labels
% Peek into the data
% Note the dimensions of the data.

classes = categories(YTrain) % View the class labels
size(XTrain) % View the dimensions of the training set of image data
%% 
% View the infrared channel of the 507th land cover image in grayscale

image507 = XTrain(:,:,4,507)
imshow(image507)
label507 = YTrain(507)
%% Create a Network from Layers
% Create the CNN Layers
% Create a column vector of 6 layers named |layers|, with the layers in the 
% order shown below.
% 
% |imageInputLayer(_inputSize_)|
% 
% |convolution2dLayer(_filterSize_,_numFilters_)|
% 
% |reluLayer()|
% 
% |maxPooling2dLayer(_poolSize_)|
% 
% |fullyConnectedLayer(_numClasses_)|
% 
% |softmaxLayer()|
% 
% Use the following information to choose the inputs for the layers:
%% 
% * Each image is size 28-by-28-by-4.
% * The convolution layer should have a 20 filters of size 3-by-3.
% * The pooling layer should have a pool size of 3-by-3.
% * There are six classes.

layers = [imageInputLayer([28,28,4]);...
    convolution2dLayer([3,3], 20);...
    reluLayer(); maxPooling2dLayer([3,3]);...
    fullyConnectedLayer(6); softmaxLayer();];
%% 
% View the network architecture

analyzeNetwork(layers)
% Training Options
% Create training options named |options|. Use the algorithm |"sgdm"|, set the 
% maximum number of epochs to |5|, and set the initial learning rate to |0.0001|.

options = trainingOptions("sgdm","MaxEpochs",100,...
    "InitialLearnRate",0.0001,"Plots","training-progress",...
    "Metrics","accuracy","ExecutionEnvironment","auto","Shuffle","once");
% Train the Network

landnet = trainnet(XTrain,YTrain,layers,"crossentropy",options)
% Classify and Evaluate the Network

scores = minibatchpredict(landnet,XTest)  % Prediction by the network
[testPred, score] = scores2label(scores,classes)
testGT = YTest;
testAcc = nnz(testPred == testGT) / numel(testPred)
confusionchart(YTest,testPred) % Visualization in a confusion matrix
%% Convolutional Layer, Performing Convolutions

im = imread("sunflower.jpg"); % Load a sample image
redIm = im(:,:,1); % Extract the red channel
imshow(redIm)
%% 
% 1. Create a 3-by-3 filter(often called 'kernel') that blurs an image and use 
% the filter with the |conv2| function.

blurKernel = 1/9*ones(3)
blurConv = conv2(blurKernel,redIm)
%% 
% View the result using |imshow| function. Note that it is often useful to use 
% a second argument to imshow when displaying grayscale images. Using empty brackets 
% as the second input will scale the display based on the minimum and maximum 
% values present in the image.

imshow(blurConv,[]) % imshow(rescale(blurConv))
%% 
% 2. Create a 3-by-3 filter that detects edges and use the filter.

edgeKernel = [0 1 0; 1 -4 1; 0 1 0]
edgeConv = conv2(edgeKernel,redIm);
imshow(edgeKernel,[])
%% Viewing Learned Filters
% Load and display an image with different colored flowers.

im = imread("roses.JPG");
imshow(im)
%% 
% Load GoogLeNet and saves the layers in a variable

net = imagePretrainedNetwork("googlenet");
layers = net.Layers
analyzeNetwork(net)
% analyzeNetwork(layers)
%% 
% The |Weights| property of a layer contains that layer's weights. Weights are 
% learned during training. Since GoogLeNet is a pretrained network, it has already 
% learned weights that will find useful features. In this activity, you will inspect 
% these weights in more detail.
% 
% Save the weights of the second layer to a variable and view the array of weights

layer2 = layers(2);
weightLayer2 = layer2.Weights
montage(rescale(weightLayer2))
%% 
% You can view a specific filter by indexing into the weights. Extract the eleventh 
% filter from the array of weights, then display it.

filter11 = weightLayer2(:,:,:,11)
imshow(rescale(filter11))
%% 
% The eleventh filter seems to be looking for the color pink. The image roses.jpg 
% contains a few plants, but only the roses are red. Hence, only the roses should 
% be positively activated from the eleventh filter.
%% 
% Let's see the eleventh filter actually activate the pink color from the image.

actvn = minibatchpredict(net,im,Outputs="conv1-7x7_s2");
imshow(rescale(actvn(:,:,11)))