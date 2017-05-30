using FITSIO;
using PyPlot
PyPlot.show()
include("MAPGMM.jl")


# read in true image (for reference) and noisy image (= data from which we reconstruct)
x = rotl90(read((FITS("2004true141.fits"))[1])); #rotl90 to get same orientation as IDL
n = 80/255;
xn = x + n*randn(size(x));
GDict = importGMM("GMM_YSO.mat");
tic();
g = zeros(Float64, size(x));
f = EPLL_fg(x, g, GDict);
toc();
imshow(g, interpolation="none");


readline();
