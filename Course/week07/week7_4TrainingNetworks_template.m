%% 4. Training Networks

load satData.mat % load the image and class labels
% Peek into the data
% Note the dimensions of the data.

categories(YTrain) % View the class labels
size(XTrain) % View the dimensions of the training set of image data
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

layers = [imageInputLayer([28,28,4]); convolution2dLayer([3,3], 20);...
    reluLayer(); maxPooling2dLayer([3,3]);...
    fullyConnectedLayer(6); softmaxLayer()];
%% Train the Network with Different Training Options
% Training Options
% Create training options named |options|. 
%% 
% * Use the algorithm |"sgdm"|,|"adam"| and |"rmsprop"|
% * Set the maximum number of epochs to 5 to 100
% * Set the minibatch size to 200
% * Set the initial learning rate to 0.0001 and 0.001 (If you set 0.001, you 
% will see |NaN| of Mini-batch Loss in the training-progress plot. How can you 
% fix this without decreasing the initial learning rate? Use Gradient Clipping 
% with |GradientThreshold| option.)
% * Plot the training progress

options = trainingOptions("adam","MaxEpochs",50,...
    "InitialLearnRate",0.0001,"MiniBatchSize",128,...
    "Plots","training-progress","Metrics","accuracy");
% Train the Network

landnet = trainnet(XTrain,YTrain,layers,"crossentropy",options);
% Classify and Evaluate the Network

classes = categories(YTest);
scores = minibatchpredict(landnet,XTest); % Prediction by the network
[testPred, score] = scores2label(scores,classes);
testAcc = nnz(testPred == YTest) / numel(testPred)
confusionchart(YTest,testPred) % Visualization in a confusion matrix
%% Validation Data Set
% To prevent overfitting, you can use a validation data set. If the validation 
% accuracy is consistently lower than the training accuracy, you can stop the 
% training.

valData = {XVal YVal}
options = trainingOptions("rmsprop", "MaxEpochs",50,...
    "MiniBatchSize",128,"InitialLearnRate",0.0001,...
    "Plots","training-progress","Metrics","accuracy",...
    "ValidationData",valData,"ValidationFrequency",10,...
    "OutputNetwork","best-validation");


landnet = trainnet(XTrain,YTrain,layers,"crossentropy",options);
classes = categories(YTest);
scores = minibatchpredict(landnet,XTest); % Prediction by the network
[testPred, score] = scores2label(scores,classes);
testAcc = nnz(testPred == YTest) / numel(testPred)
confusionchart(YTest,testPred) % Visualization in a confusion matrix