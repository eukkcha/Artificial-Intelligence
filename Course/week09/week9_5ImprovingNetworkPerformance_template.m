%% 5. Improving Network Performance
%%   5.1. Data Preprocessing
%% Load Dataset
% xData - Each row of the array stores a 32 x 32 color image. The first 1024 
% entries contain the red channel values, the next 1024 the green, the final 1024 
% the blue.
% 
% yLabel - a list of 10000 numbers in the range of 0-9.

load ("data_batch_1.mat")
data
labels
categories(categorical(labels)) % 10개의 데이터가 있구나
summary(categorical(labels))
%% 1. Preprocessing xData (permute(transpose) -> reshape -> permute(imrotate))
% xData is a 10000 x 3072 array, which contains 10000 observations and  32 x 
% 32 x 3 sized images. Your goal is to reshape xData to 32 x 32 x 3 x 10000.
% 1. Reshape xData to 3072 x 10000
% There are two ways to create 3072 x 10000, by using |permute| or |transpose|.
% 
% a. By |permute|

dataPermute1 = permute(data,[2,1])
%% 
% b. By |transpose|

dataTransposed = data' % 사용하지 말 것.
%% 
% The size of |dataPermute1| variable is 3072 x 1000.
% 2. Reshape it to 32 x 32 x 3 x 10000 using |reshape| and view the 10000th image of |dataReshape| variable

dataReshape = reshape(dataPermute1,[32, 32, 3, 10000])
imshow(dataReshape(:,:,:,10000))
%% 
% You recognize the image is a dog but it is rotated.
% 3. Rotate the images
% a. By |permute|

dataPermute2 = permute(dataReshape,[2 1 3 4])
imshow(dataPermute2(:,:,:,10000))
%% 
% b. By |imrotate|



%%
montage(dataPermute2)
xData = dataPermute2
%% 2. Preprocessing yLabel
% |labels| variable is 10000 x 1 unit8 vector. Convert it to a categorical variable 
% and display the type and number of categories.

yLabel = categorical(labels);
categories(yLabel) % the type of categories
summary(yLabel) % the number of categories
%% 3. Spliting Dataset
% Partition the data into train and test sets. As in Machine Learning, |cvpartition| 
% is used for partition.

% 1. 이 방법 사용하기
pt = cvpartition(yLabel,"HoldOut",0.2)
xTrain = xData(:,:,:,pt.training);
yTrain = yLabel(pt.training);

xTest = xData(:,:,:,pt.test);
yTest = yLabel(pt.test);

summary(yTrain)
summary(yTest)
%%
 % 2. 이렇게 하진 마셈 (이런 방법도 있다)
pt = cvpartition(yLabel,"HoldOut",0.2)
tridx = training(pt)

xTrain = xData(:,:,:,tridx);
yTrain = yLabel(tridx);

xTest = xData(:,:,:, ~tridx);
yTest = yLabel(~tridx);

summary(yTrain)
summary(yTest)