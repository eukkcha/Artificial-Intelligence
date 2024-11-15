%% Q2. Transfer Learning - Image Classification _(40 points)_
%% 1. Pretrained network (3)
%     1.1. Import GoogLeNet (1)

net = imagePretrainedNetwork("googlenet")
%     1.2. What is the input image size of GoogLeNet? (2)
% Enter your code to ouput the input image size of GoogLeNet.

layers = net.Layers
layer1 = layers(1)
%% 2. Importing dataset (5)
%     2.1. Create an image datastore (3)
% Create an image datastore which references the folder of the dataset.

imageDS = imageDatastore("/Users/eukkcha/Desktop/Sejong/AI/week8_midterm/q2_data",...
    "IncludeSubfolders",true,"LabelSource","foldernames")
%     2.2 What ouput classes does the dataset contain? (2)
% Enter your code to output the classes of the dataset.

classes = categories(imageDS.Labels)
%% 
% _View a sample image of each class_

    %-------------- DO NOT modify this code--------------
    selectedClass="squirrel";
    classIdx = contains(imageDS.Files,selectedClass);
    countClass = sum(classIdx);
    randNum = randi([1,countClass],1);
    classRowNum = find(classIdx);
    classRandIm = readimage(imageDS, classRowNum(1)+randNum-1);
    imshow(classRandIm)
    %----------------------------------------------------
%% 3. Preparing Data (5)
%     3.1. Split dataset (2)
% Split the dataset into train(0.8) and test data(0.2). 
%% 
% * *For an optional argument, input "|randomized|" option.*

[trainImg,testImg] = splitEachLabel(imageDS,0.8,"randomized")
%     3.2. Augment data (3)
% Resize the dataset for transfer learning.
%% 
% * *Resize the images to 110 x 110.*
% * *For an optional input argument, input "|gray2rgb|" for "|ColorPreprocessing|" 
% option.*

trainData = augmentedImageDatastore([110,110],trainImg,"ColorPreprocessing","gray2rgb")
testData = augmentedImageDatastore([110,110],testImg,"ColorPreprocessing","gray2rgb")
%% 4. Modifying Network (10)
% Modify appropriate layers of GoogLeNet so that the network works for your 
% dataset and then, export the network into your workspace. 
%% 
% * *Note that the input image size should be 110 x 110.*
% * *Be sure to name your network 'net_1' and include it into the zip file, 
% when you submit your answer.*
%% 5. Training Network (7)
% Create appropriate training options to obtain good prediction accuracy for 
% the test data.
%     5.1. Create training options (5)
% The training options you *SHOULD KEEP* are as follows. The rest of training 
% options are tunerable.
%% 
% * Max Epochs: 10
% * Display training progress and accuracy for the training data

opts = trainingOptions("sgdm","MaxEpochs",10,"InitialLearnRate",0.001,...
    "Plots","training-progress","Metrics","accuracy","ExecutionEnvironment","auto","Shuffle","once")
%     5.2. Train Network (2)
% Train network.

landnet = trainnet(trainData,net_1,"crossentropy",opts)
%% 6. Evaluating network (10)
% Evaluate the network by classifying the test data, calculating the classification 
% accuracy and displaying a confusion chart.
%     6.1. Predict the class of the test data (4)

scores = minibatchpredict(landnet,testData)
[testPred,score] = scores2label(scores,classes)
%     6.2. Calculate the prediction accuracy (4)

testGT = testImg.Labels
testAcc = nnz(testGT == testPred) / numel(testPred)
%     6.3. Display a confusion chart (2)

confusionchart(testGT,testPred)