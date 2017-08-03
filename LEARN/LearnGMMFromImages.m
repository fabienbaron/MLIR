%% initialize model
pkg load image
pkg load statistics
pkg load fits

PatchSize = 8;
nmodels = 15;
MiniBatchSize = 5000;
output_dict = sprintf('GMM_%dx%d_%d_%d.mat',PatchSize,PatchSize,nmodels,MiniBatchSize);

% load images into cell
Images = {};

% Note: FITS images need to be rescaled to range 0-1
% Also better crop to non-zero flux than to use the true flag in OnlineGMMEM
Images{1} = (double(read_fits_image('2004true137.fits')))';
Images{2} = (double(read_fits_image('2004true137.fits')))';
Images{3} = (double(read_fits_image('2004true137.fits')))';

GMM0=nmodels;
DataSource=@(N) removeDC(RandPatchesFromImagesCell(N,PatchSize,Images));
NumIterations=1000;
OutputFile=output_dict;
T0=500;
alpha=0.6;
FirstBatchSize=MiniBatchSize*10;
removeFlatPatches=false;
%% learn model from training data
NewGMM = OnlineGMMEM(nmodels,@(N) removeDC(RandPatchesFromImagesCell(N,PatchSize,Images)),50,MiniBatchSize,output_dict, 500, 0.6, MiniBatchSize*10, false);

% sort output
%[NewGMM.mixweights,inds] = sort(NewGMM.mixweights,'descend');
%NewGMM.covs = NewGMM.covs(:,:,inds);
