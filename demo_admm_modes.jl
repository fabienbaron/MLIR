include("aradmm_image_denoising.jl")
## minimize  mu/2 ||x-f||^2 + lam1 |\grad x|
## load image
using FITSIO
using PyPlot
imgfile="2004true137.fits";
x_true = Float64.(read(FITS(imgfile)[1]));#load image
@printf("loaded image size: %d*%d*%d\n", size(x_true, 1), size(x_true, 2), size(x_true, 3));
# 2004true image
sig = maximum(x_true)/20.; # noise
mu = 1.0/sig^2;  #constraint
lam1 = 12.0; # l1 regularizer of gradient
x_given = x_true + sig*randn(size(x_true));

## model parameter
opts = optsinfo(1e-3,2000,0.1,1,5,2,2,1000,0.2,2,0.1,1); #relaxation parameter
opts.verbose = 3

@printf("ADMM start...\n");
##
# vanilla ADMM
opts.adp_flag = 0;
opts.gamma = 1;
(sol1, outs1) = aradmm_image_denoising(x_given, mu, lam1, opts);
@printf("vanilla ADMM complete after %d iterations!\n", outs1.iter);
figure(1)
imshow(sol1)

# relaxed ADMM
opts.adp_flag = 0;
opts.gamma = 1.5;
(sol2,outs2) =  aradmm_image_denoising(x_given, mu, lam1, opts);
#t2 = outs2.runtime;
@printf("relaxed ADMM complete after %d iterations!\n", outs2.iter);

figure(2)
imshow(sol2)

# residual balancing
opts.adp_flag = 3; #residual balance
opts.gamma = 1.0;
(sol3,outs3) =  aradmm_image_denoising(x_given, mu, lam1, opts);
#t3 = outs2.runtime;
@printf("RB ADMM complete after %d iterations!\n", outs3.iter);
figure(3)
imshow(sol3)

# Adaptive ADMM, AISTATS 2017
opts.adp_flag = 1;
opts.gamma = 1.0;
(sol4,outs4) =  aradmm_image_denoising(x_given, mu, lam1, opts);
#t4 = outs4.runtime;
@printf("adaptive ADMM complete after %d iterations!\n", outs4.iter);

figure(4)
imshow(sol4)

# ARADMM
#tic();
opts.adp_flag = 5;
opts.gamma = 1.0;
(sol6,outs6) =  aradmm_image_denoising(x_given, mu, lam1, opts);
#t6 = outs6.runtime;
@printf("ARADMM complete after %d iterations!\n", outs6.iter);

figure(5)
imshow(sol6)
##

# legends = {'Vanilla ADMM', 'Relaxed ADMM', 'Residual balance', 'Adaptive ADMM', 'ARADMM'};
# figure,
# semilogy(outs1.tols, '-.g'),
# hold,
# semilogy(outs2.tols, '-.r');
# semilogy(outs3.tols, '--m');
# semilogy(outs4.tols, '--', 'Color',[0.7 0.2 0.2]);
# semilogy(outs6.tols, 'b');
# ylabel('Relative residual', 'FontName','Times New Roman');
# xlabel('Iteration', 'FontName','Times New Roman');
# legend(legends, 'FontName','Times New Roman');
# hold off
# figure(1);
# imshow(outs1.);
