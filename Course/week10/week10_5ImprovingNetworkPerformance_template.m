%% 5. Improving Network Performance
%%   5.2. Training Option
%% Load Dataset

load ("Dataset_reshaped.mat")
%% Spliting Data
% Spliting training data into training and validation sets. (8:2)

pt = cvpartition(train_y,"HoldOut",0.2)

training_x = train_x(:,:,:,pt.training);
training_y = train_y(pt.training);

val_x = train_x(:,:,:,pt.test);
val_y = train_y(pt.test);


%% Creating Network
% Create a Convolutional Neural Network.

layers = [imageInputLayer([28,28,1]); convolution2dLayer([3,3],20);,...
    reluLayer(); maxPooling2dLayer([3,3]);...
    fullyConnectedLayer(10); softmaxLayer();]
%% Training Network
% Create appropriate training options and train the network

opts = trainingOptions("sgdm","MaxEpochs",200,"InitialLearnRate",0.1,...
    "Plots","training-progress","Metrics","accuracy",...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropPeriod",10,"LearnRateDropFactor",0.1,...
    "L2Regularization",0.1,...
    "ValidationData",{val_x,val_y})

landnet = trainnet(training_x,training_y,layers,"crossentropy",opts);
%% Evaluating Network
% Calculate the prediction accuracy on the test data.

scores = minibatchpredict(landnet,test_x)
[testPred,score] = scores2label(scores,categories(training_y))
testGT = test_y
testAcc = nnz(testPred == testGT) / numel(testPred)
confusionchart(test_y,testPred)