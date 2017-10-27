using FITSIO;
using PyPlot;
#using OptimPack
PyPlot.show();
include("EPLL.jl");
include("oiplot.jl")

# read in true image (for reference) and noisy image (= data from which we reconstruct)
x_true = rotl90(read((FITS("2004true137.fits"))[1])); #rotl90 to get same orientation as IDL
sigma = 100/255;
x_noisy = x_true + sigma*randn(size(x_true));
# load the machine-learned dictionary and setup corresponding patch size
dict = importGMM("GMM_YSO.mat");
np = Int(sqrt(dict.dim));
#setup the regularization step for ADMM
MAPGMM = (Z,np,sigma,imsize)->aprxMAPGMM(Z,np,sigma,imsize,dict);
figure(1);imshow(x_true, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
figure(2);imshow(x_noisy, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
figure(3);
println("Reconstruction with np= ", np, " and nmodels= ", dict.nmodels);
#reconstructed_image = EPLLhalfQuadraticSplit(x_noisy,np^2/sigma^2,np, (1/sigma^2.0)*[1 4 8 16 32 64 128 256 512 1024 2048 4000 8000 20000 40000 80000 160000 320000], 1, MAPGMM, x_true);
x_HQ = EPLLhalfQuadraticSplitNew(x_noisy,sigma,np, 1.0*[1,4,8,16,32,64,128,256,512,1024,2048,4000,8000,20000,40000,80000,160000,320000], x_true, dict);
println("PSNR: ",20*log10(1/std(x_HQ-x_true)), "\n");
println("EPLL: ", EPLL(x_HQ, dict), "\t F: ",  1/sigma^2*norm(x_HQ-x_noisy)^2+EPLL(x_HQ, dict));
println("l1dist: ", sum(abs.(x_HQ-x_true))/length(x_true), "\n");
imshow(x_HQ, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
#readline();

# ADMM method

using StatsBase
rho = 1
maxRho = 1
nx = size(x_true,1);
y = copy(x_noisy);
x = copy(x_noisy);
u = zeros(np*np, (nx-np+1)*(nx-np+1));
precalc1 = vec(im2col( reshape(1:(nx*nx),nx,nx),(np,np)));
precalc2 = counts(precalc1);
P=a->im2col(a,(np,np)); # decomposition into patches
Pt=a->reshape( ( counts(precalc1,fweights(vec(a)))./precalc2 )', (nx, nx)); # transpose
λ = 1/sigma^2;
for i=1:100
# step 1
# argmin { f_prior(Z)+ rho/2mu || Z - Ztilde ||^2 }
z = prox_GMM(P(x) + u/rho, 1./rho, dict);
# step 2
x = (λ*y + Pt(rho*z-u) )/(λ + rho);
figure(3); imshow(x, ColorMap("gist_heat"));PyPlot.draw();PyPlot.pause(0.05);
# step 3
u = u + rho * (P(x) - z)
println("Iteration $i \n");
#readline()
rho = minimum([maxRho, rho*1.1]);
println("PSNR: ",20*log10(1/std(x-x_true)));
println("l1dist: ", sum(abs.(x-x_true))/length(x_true));
println("EPLL: ", EPLL(x, dict), "\t F: ",  1/sigma^2*norm(x-x_noisy)^2+EPLL(x, dict));
end
x_ADMM = copy(x);
imshow(x_ADMM, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);


using OptimPack
# gradient method
function denoise_fg(x, g, data, sigma, dict)
mu=10.;
chi2_f = sum((x-data).^2)/(2*sigma^2);
chi2_g = (x-data)/sigma^2;
nx = Int(sqrt(length(x)));
reg_g = zeros(Float64, (nx, nx));
reg_f = EPLL_fg(reshape(x, (nx,nx)), reg_g, dict);
g[:] = chi2_g + mu*vec(reg_g);
println("Chi2r: ", chi2_f/length(x), " EPLL: ", reg_f, " F: ", chi2_f + mu*reg_f );
imshow(reshape(x, (nx, nx)), ColorMap("gray"));PyPlot.draw();
return chi2_f + mu*reg_f
end

x_start= vec(x_noisy);
data  = vec(x_noisy);
g = zeros(size(xstart));
crit = (x,g)->denoise_fg(x, g, data, sigma, dict);figure(4);
figure(4);
x = OptimPack.vmlmb(crit, x_start, verb=true, lower=0, maxiter=80, blmvm=false);
reconstructed_image = reshape(x, size(x_noisy));
println("PSNR: ",20*log10(1/std(reconstructed_image-x_true)), "\n");
println("EPLL: ", EPLL(reconstructed_image, dict), "\n");
println("l1dist: ", sum(abs.(reconstructed_image-x_true))/length(x_true), "\n");
imshow(reconstructed_image, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
readline();
