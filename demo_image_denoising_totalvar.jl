include("aradmm_image_denoising.jl")
## minimize  mu/2 ||x-f||^2 + lam1 |\grad x|
## load image
using FITSIO
using PyPlot
#imgfile="cameraman256.fits";
imgfile="2004true137.fits";
x_true = Float64.(read(FITS(imgfile)[1]));#load image
@printf("loaded image size: %d*%d*%d\n", size(x_true, 1), size(x_true, 2), size(x_true, 3));

sig = maximum(x_true)/20; # noise
x_given = x_true + sig*randn(size(x_true));
x_given[x_given.>255.0] = 255.0;
x_given[x_given.<0.0] = 0.0;

opts = optsinfo(1e-3,2000,0.1,1,5,2,2,1000,0.2,2,0.1,1); #relaxation parameter
opts.adp_flag = 5;
opts.γ = 1.;
opts.verbose = 0

##
#L-CURVE
 mu = 1.0/sig^2;  #constraint
 lcurve=collect(linspace(11., 20., 10))
 for lam1 in lcurve
 (sol6,outs6) =  aradmm_image_denoising(x_given,x_given,  mu, lam1, opts);
 println("lambda = $lam1 PSNR: ",20*log10(1/std(sol6-x_true)), "  l1diff: ", norm(sol6-x_true,1), " MSE: ", norm(sol6-x_true,2), "\n");
 @printf("ARADMM complete after %d iterations!\n", outs6.iter);
 figure(5)
 imshow(rotl90(sol6),ColorMap("gray"), interpolation="none");
 tight_layout();
 end
