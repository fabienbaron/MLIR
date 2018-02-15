using FITSIO;
using PyPlot;
using JLD;
#using OptimPack
PyPlot.show();
include("EPLL.jl");
include("oiplot.jl")

# read in true image (for reference) and noisy image (= data from which we reconstruct)
x_true = rotl90(read((FITS("2004true137.fits"))[1])); #rotl90 to get same orientation as IDL
sigma = 100/255;
x_noisy = x_true + sigma*randn(size(x_true));
# load the machine-learned dictionary and setup corresponding patch size
dict = load("GMM_YSO.jld", "GMM");
#setup the regularization step for ADMM
figure(1);imshow(x_true, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
figure(2);imshow(x_noisy, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
figure(3);
x_HQ = HQ_EPLL(dict, x_noisy,sigma, x_true);
figure(3); imshow(x_HQ, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);

# ADMM method

using StatsBase
t = 1.0
maxt = 1.0
np = Int(sqrt(dict.dim));
nx = size(x_noisy,1);
y = copy(x_noisy);
x = copy(x_noisy);
u = zeros(np*np, (nx-np+1)*(nx-np+1));
precalc1 = vec(im2col( reshape(1:(nx*nx),nx,nx),(np,np)));
precalc2 = counts(precalc1);
P=a->im2col(a,(np,np)); # decomposition into patches
Pt=a->reshape( ( counts(precalc1,fweights(vec(a)))./precalc2 )', (nx, nx)); # transpose
λ = 1/sigma^2;
for i=1:300
# step 1
# argmin { f_prior(Z)+ rho/2mu || Z - Ztilde ||^2 }
z = prox_GMM(P(x) + u/t, 1./t, dict);
# step 2
x = (λ*y + Pt(t*z-u) )/(λ + t);
figure(3); imshow(x, ColorMap("gist_heat"));PyPlot.draw();PyPlot.pause(0.05);
# step 3
u = u + t * (P(x) - z)
println("Iteration $i \n");
#readline()
#rho = minimum([maxRho, rho*1.1]);
println("PSNR: ",20*log10(1/std(x-x_true)), " l1dist: ", norm(x-x_true,1), " MSE: ", norm(x-x_true,2), "\n");
println("EPLL: ", EPLL(x, dict), "\t F: ",  1/sigma^2*norm(x-x_noisy)^2+EPLL(x, dict));
end
x_ADMM = copy(x);
imshow(x_ADMM, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);

#
# Pure gradient method -- Don't run
#
using OptimPack
# gradient method
function denoise_fg(x, g, data, sigma, dict)
mu=1.;
chi2_f = 0.5*norm((x-data)./sigma)^2;
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
crit = (x,g)->denoise_fg(x, g, data, sigma, dict);
figure(4);
x = OptimPack.vmlmb(crit, x_start, verb=true, lower=0, maxiter=80, blmvm=false);
reconstructed_image = reshape(x, size(x_noisy));
println("PSNR: ",20*log10(1/std(x-x_true)), " l1dist: ", norm(x-x_true,1), " MSE: ", norm(x-x_true,2), "\n");
println("EPLL: ", EPLL(reconstructed_image, dict), "\n");
imshow(reconstructed_image, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
readline();
