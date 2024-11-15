%% Spectrogram for Image Classification
%% Read Audio Data

[y, Fs] = audioread("pianoSound.m4a");
sound(y(:,1), Fs) % play the audio
%% Comparison Between Plot and Spectrogram

plot(y(:,1))
pspectrum(y(:,1),Fs,"spectrogram")
ylim([0,5])
%% Remove the Unnecessary
% Remove axis, colorbar and title for image classification

hold on
axis off
colorbar off
title("")
hold off