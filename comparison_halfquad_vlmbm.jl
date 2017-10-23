using FITSIO;
using PyPlot;
#using OptimPack
PyPlot.show();
include("EPLL.jl");
include("oiplot.jl")

# read in true image (for reference) and noisy image (= data from which we reconstruct)
true_image = rotl90(read((FITS("2004true137.fits"))[1])); #rotl90 to get same orientation as IDL
noiseSD = 80/255;
noisy_image = true_image + noiseSD*randn(size(true_image));
# load the machine-learned dictionary and setup corresponding patch size
GDict = importGMM("GMM_YSO.mat");
patchSize = Int(sqrt(GDict.dim));
#setup the regularization step for ADMM
MAPGMM = (Z,patchSize,noiseSD,imsize)->aprxMAPGMM(Z,patchSize,noiseSD,imsize,GDict);
figure(1);imshow(true_image, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
figure(2);imshow(noisy_image, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
figure(3);
println("Reconstruction with patchsize= ", patchSize, " and nmodels= ", GDict.nmodels);
reconstructed_image = EPLLhalfQuadraticSplit(noisy_image,patchSize^2/noiseSD^2,patchSize, (1/noiseSD^2.0)*[1 4 8 16 32 64 128 256 512 1024 2048 4000 8000 20000 40000 80000 160000 320000], MAPGMM, true_image, Gdict);
println("PSNR: ",20*log10(1/std(reconstructed_image-true_image)), "\n");
println("EPLL: ", EPLL(reconstructed_image, GDict), "\n");
println("l1dist: ", sum(abs.(reconstructed_image-true_image))/length(true_image), "\n");
imshow(reconstructed_image, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
readline();

# ADMM method

using StatsBase
rho = .5
maxRho = 5
sigma = noiseSD
nx = size(true_image,1);
Y = copy(noisy_image);
X = copy(noisy_image);
U = zeros(size(im2col(X,(patchSize,patchSize))));
temp = vec(im2col( reshape(1:(nx*nx),nx,nx),(patchSize,patchSize)));
for i=1:30
# step 1
# argmin { f_prior(Z)+ rho/2mu || Z - Ztilde ||^2 }
# Ztilde = PX - U/rho
Ztilde = im2col(X,(patchSize,patchSize)) - U / rho
# z = proximal_GMM(ztilde, noiseSD)
Z = MAPGMM(Ztilde, patchSize, noiseSD, (nx,nx));

# step 2
X = 1.0/noiseSD^2.*noisy_image+patchSize^2*reshape( ( counts(temp,WeightVec(vec(U+rho*Z)))./counts(temp) )', (nx, nx))
X /= (1.0/noiseSD^2.+rho*patchSize^2)
imdisp(X)
# step 3
U = U + rho * (Z - im2col(X,(patchSize,patchSize)))
println("Iteration $i \n");
#readline()
rho = minimum([maxRho, rho*1.1]);
end



using OptimPack
# gradient method
function denoise_fg(x, g, data, noiseSD, GDict)
mu=0.;
chi2_f = sum((x-data).^2)/(noiseSD^2 * length(x));
chi2_g = (x-data)/(0.5*noiseSD^2 * length(x));
nx = Int(sqrt(length(x)));
reg_g = zeros(Float64, (nx, nx));
reg_f = EPLL_fg(reshape(x, (nx,nx)), reg_g, GDict);
g[:] = chi2_g + mu*vec(reg_g);
println("Chi2 = ", chi2_f, " EPLL =", reg_f);
imshow(reshape(x, (nx, nx)), ColorMap("gray"));PyPlot.draw();
return chi2_f + mu*reg_f
end


xstart= vec(noisy_image);
data  = vec(noisy_image);
g = zeros(size(xstart));
crit = (x,g)->denoise_fg(x, g, data, noiseSD, GDict);figure(4);
figure(4);
x = OptimPack.vmlmb(crit, xstart, verb=true, lower=0, maxiter=80, blmvm=false);
reconstructed_image = reshape(x, size(noisy_image));
println("PSNR: ",20*log10(1/std(reconstructed_image-true_image)), "\n");
println("EPLL: ", EPLL(reconstructed_image, GDict), "\n");
println("l1dist: ", sum(abs.(reconstructed_image-true_image))/length(true_image), "\n");
imshow(reconstructed_image, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
readline();
