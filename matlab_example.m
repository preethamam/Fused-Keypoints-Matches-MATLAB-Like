clc; close all; clear;

% I1 = imread("cameraman.tif");
% I2 = imresize(imrotate(I1,-20), 1.2);

% I1 = imread("images/cameraman.png");
% I2 = imread("images/cameraman_rotsc.png");

I1 = imread("peppers.png");
I2 = imresize(imrotate(I1,-20), 1.2);

% Detect SURF features.
points1 = detectSURFFeatures(rgb2gray(I1));
points2 = detectSURFFeatures(rgb2gray(I2));

% Extract features.
[f1,vpts1] = extractFeatures(rgb2gray(I1),points1);
[f2,vpts2] = extractFeatures(rgb2gray(I2),points2);

% Match features.
indexPairs = matchFeatures(f1,f2);
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));

% Visualize candidate matches.
figure;
showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
title("Putative point matches");
legend("Matched points 1","Matched points 2");