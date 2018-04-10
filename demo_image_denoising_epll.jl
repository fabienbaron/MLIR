include("EPLL.jl");
include("aradmm_image_denoising.jl")
include("aradmm_image_epll.jl")
## minimize  mu/2 ||x-f||^2 + lam1 |\grad x|
## load image
using FITSIO
using PyPlot
imgfile="2004true137.fits";
x_true = Float64.(read(FITS(imgfile)[1]));#load image
@printf("loaded image size: %d*%d*%d\n", size(x_true, 1), size(x_true, 2), size(x_true, 3));
# 2004true image
sig = maximum(x_true)/10.; # noise
mu = 1.0/sig^2;  #constraint

x_noisy = x_true + sig*randn(size(x_true));
## model parameter
opts = optsinfo(1e-3,200,0.1,3,0,2,2,1000,0.2,1.01,0.5,1.0); #relaxation parameter
#opts.verbose = 3
#
# opts.adp_flag = 3;
# opts.γ = 1.0;
#lam1 = 12.0; # l1 regularizer of gradient
# (xsol7,outs7) =  aradmm_image_denoising(x_noisy,x_noisy, mu, lam1, opts);
# println("lambda = $lam1 PSNR: ",20*log10(1/std(xsol7[:]-x_true[:])), "  l1diff: ", norm(xsol7[:]-x_true[:],1), " MSE: ", norm(xsol7[:]-x_true[:],2), "\n");
# @printf("ARADMM complete after %d iterations!\n", outs7.iter);
# figure(3)
# imshow(xsol7);
#
# #EPLL
lam1 = 1.0;
Gdict = load("GMM_YSO.jld","GMM");
#println("EPLL criterion for TV image:", 0.5*mu*norm(xsol7[:]-x_noisy[:])^2+lam1*EPLL(xsol7, Gdict));
#println("EPLL criterion for true image:", 0.5*mu*norm(x_true[:]-x_noisy[:])^2+lam1*EPLL(x_true, Gdict));
#
#
# opts.adp_flag = 0;
# opts.γ = 1;
# (xsol1, outs1) = aradmm_image_epll(x_noisy, mu, lam1, opts, Gdict);
# @printf("vanilla ADMM complete after %d iterations!\n", outs1.iter);
# figure(3)
# imshow(xsol1, interpolation="none");

# relaxed ADMM
opts.adp_flag = 0;
opts.γ = 1.3;
(xsol2,outs2) =  aradmm_image_epll(x_noisy, mu, lam1, opts, Gdict);
#t2 = outs2.runtime;
@printf("relaxed ADMM complete after %d iterations!\n", outs2.iter);
figure(3)
imshow(xsol2, interpolation="none");

# residual balancing
opts.adp_flag = 3; #residual balance
opts.γ = 1.0;
opts.τ = 1e-2;
opts.maxiter = 200;
(xsol3,outs3) =  aradmm_image_epll(x_noisy, mu, lam1, opts, Gdict);
#t3 = outs2.runtime;
@printf("RB ADMM complete after %d iterations!\n", outs3.iter);
figure(3)
imshow(xsol3, interpolation="none");

# Adaptive ADMM, AISTATS 2017
opts.adp_flag = 1;
opts.γ = 1.0;
(xsol4,outs4) =  aradmm_image_epll(x_noisy, mu, lam1, opts, Gdict);
#t4 = outs4.runtime;
@printf("adaptive ADMM complete after %d iterations!\n", outs4.iter);
figure(3)
imshow(xsol4, interpolation="none");

# ARADMM
#tic();
opts.adp_flag = 5;
opts.γ = 1.0;
(xsol6,outs6) =  aradmm_image_epll(x_noisy, mu, lam1, opts, Gdict);
#t6 = outs6.runtime;
@printf("ARADMM complete after %d iterations!\n", outs6.iter);
figure(3)
imshow(xsol6, interpolation="none");







#(sol6,outs6) =  aradmm_image_epll(x_noisy, mu, lam1, opts, Gdict);
#println("lambda = $lam1 PSNR: ",20*log10(1/std(sol6-x_true)), "  l1diff: ", norm(sol6-x_true,1), " MSE: ", norm(sol6-x_true,2), "\n");
#@printf("ARADMM complete after %d iterations!\n", outs6.iter);
