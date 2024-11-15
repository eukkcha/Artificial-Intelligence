%% 6. Image Regression

load imageRegressionData.mat
%% What is Regression?
% Regression is another task that can be accomplished with deep learning. _Regression_ 
% refers to assigning continuous response values to data, instead of discrete 
% classes.
% 
% One example of image regression is correcting rotated images. The input data 
% is a rotated image, and the known response is the angle of rotation.
% 
% 
% 
% You will build a network that corrects the color in images. 
% 
% A set of images have been modified by changing their red, green, and blue 
% channels. The response for each modified image is three numeric values that 
% correspond to the intensity increase or decrease of the corresponding channel.
% Peek into the data

imshow(imread(trainingData.File{1}))

%% Transfer Learning for Image Regression
%     Modify the GoogLeNet for regression using Deep Network Designer App
%% 
% # Replace the |fullyConnectedLayer| 
% # Delete the |softmaxLayer|
%     Prepare the data

trainds = augmentedImageDatastore([224,224],trainingData);
testds = augmentedImageDatastore([224,224],testData);
%% Train the Network 

opts = trainingOptions("adam","InitialLearnRate",0.0001,...
    "MaxEpochs",30,"Plots","training-progress","Metrics","rmse");
newNet = trainnet(trainds,net_1,"mse",opts)
%% Evaluate the Network
% Predict the response for all images in the test data and calculate the root 
% mean squre error (RMSE) for the test data set.

testPred = minibatchpredict(newNet, testds)
rgbGT = testData.Color
err = rgbGT - testPred
rootMeanSquaredError = sqrt(mean(err.^2))
% rootMeanSquaredError = rmse(rgbGT, testPred)
%% Correct the color
% Use the network to correct the rgb value of the first test image and display 
% the corrected image.

testImage = imread(testData.File{10})
imshow(testImage)
rgb = testPred(10,:)
correctedImage = correctColor(testImage,rgb)
imshow(correctedImage)
%%
function fixedim = correctColor(im, rgb)
    fixedim = uint8(single(im) - reshape(rgb,[1 1 3]));
end