using FITSIO
include("removeDC.jl")
include("RandPatchesFromImagesCell.jl")
include("OnlineGMMEM.jl")
PatchSize = 8;
nmodels = 15;
MiniBatchSize = 5000;
output_dict = "GMM_$(PatchSize)x$(PatchSize)_$(nmodels)_$(MiniBatchSize).mat";
Images = Array(Array{Float64, 2}, 3);
#% Note: FITS images need to be rescaled to range 0-1
# Also better crop to non-zero flux than to use the true flag in OnlineGMMEM
Images[1] = read((FITS("2004true137.fits"))[1]);
Images[2] = read((FITS("2004true137.fits"))[1]);
Images[3] = read((FITS("2004true137.fits"))[1]);

DataSource = N->removeDC(RandPatchesFromImagesCell(N,PatchSize,Images))
GMM0 = nmodels
NumIterations = 1000
OutputFile = output_dict
T0 = 500
alpha = 0.6
FirstBatchSize = MiniBatchSize*10
removeFlatPatches = false

# learn model from training data
#NewGMM = OnlineGMMEM(nmodels,N->removeDC(RandPatchesFromImagesCell(N,PatchSize,Images)),1000,MiniBatchSize,output_dict, 500, 0.6, MiniBatchSize*10, false);
