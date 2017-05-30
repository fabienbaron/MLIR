using FITSIO;
using Base.Cartesian #for nloops in blockproc
using StatsBase
using MAT
#using Debug
using PyPlot
PyPlot.show()
include("EPLLhalfQuadraticSplit.jl")

# read in true image (for reference) and noisy image (= data from which we reconstruct)
true_image = rotl90(read((FITS("2004true.fits"))[1])); #rotl90 to get same orientation as IDL
noiseSD = 80/255;
noisy_image = true_image + noiseSD*randn(size(true_image));
#f = FITS("2004noisy.fits", "w");
#write(f, noisy_image);close(f);
#noisy_image = read((FITS("2004noisy.fits"))[1]);


#Array{Float64,2}(read((FITS("160068_noisy.fits"))[1])');
# load the machine-learned dictionary and setup corresponding patch size
vars = get(matread("GMM_YSO.mat"), "GMM", 0);
#vars = get(matread("GSModel_8x8_200_2M_noDC_zeromean.mat"), "GS", 0);
type GMM
  nmodels::Float64
  dim::Float64
  covs::Array{Float64,3}
  invcovs::Array{Float64,3}
  mixweights::Array{Float64,2}
  means::Array{Float64,2}
end

#clumsy
covs = get(vars, "covs", 0);
invcovs = covs*.0;
for i=1:size(covs,3)
  invcovs[:,:,i]=inv(covs[:,:,i]);
end

GDict = GMM(get(vars, "nmodels",0), get(vars, "dim", 0), covs, invcovs, get(vars,"mixweights", 0), get(vars,"means", 0));

#setup the regularization step for ADMM
#MAPGMM = (Z,patchSize,noiseSD,imsize)->aprxMAPGMM(Z,patchSize,noiseSD,imsize,GDict);
# note: npatches = (imageWith - patchSize + 1)^2
#figure(1);imshow(true_image, ColorMap("gray"),interpolation="none");PyPlot.draw();PyPlot.pause(0.05);
#figure(2);imshow(delta_image, ColorMap("gray"),interpolation="none");PyPlot.draw();PyPlot.pause(0.05);
#imshow(reshape(patches[:,6786],(8,8)), interpolation="none")
#imshow(reshape(sum((R'\X).^2,1), 134, 134), interpolation="none")

#true_image = randn(size(true_image));

grad_num = copy(true_image)*0.;
delta = 1e-6;
ref = EPLL(GDict, true_image);
delta_image = copy(true_image);
for x=1:141
  println("x= ",x,"\n");
  for y=1:141
    orig = delta_image[x,y];
    delta_image[x,y]= orig + delta;
    left = EPLL(GDict, delta_image);
    delta_image[x,y]= orig - delta;
    right = EPLL(GDict, delta_image);
    delta_image[x,y]= orig;
    grad_num[x,y]=(right - left)/(2.*delta);
  end
end

f = FITS("grad_num_cent.fits", "w");
write(f, grad_num);
close(f);

imshow(grad_num, interpolation="none");

patchSize = Int(sqrt(GDict.dim))
patches = im2col(true_image,(patchSize,patchSize));
npatches = size(patches,2)
nmodels = Int(GDict.nmodels)
#completely remove the mean
mean_patches = mean(patches,1);
for i=1:npatches
  @inbounds patches[:,i] -= mean_patches[i];
end

Pmodels = zeros(round(Int, GDict.nmodels),size(patches,2));
for k=1:nmodels
  Pmodels[k,:] = log(GDict.mixweights[k]) + loggausspdf2(patches,GDict.covs[:,:,k]);
end
P = logsumexp(Pmodels); # log p(PiX), hence vector of size=npatches
epll = sum(P)#/npatches
GPmodels = zeros(round(Int, GDict.nmodels),(patchSize*patchSize), size(patches,2));
for i=1:npatches
  for k=1:nmodels
      GPmodels[k,:,i] = -exp(Pmodels[k,i]-P[i])*(GDict.invcovs[:,:,k]*patches[:,i])
  end
end
GP = squeeze(sum(GPmodels,1),1);
epll_grad=col2im(GP, (patchSize, patchSize), size(true_image), "sliding", "sum");

imshow(epll_grad, interpolation="none");


#  end
#P = logsumexp(Pmodels,1);
#reg = sum(P)/size(patches,2)
#println("EPLL delta new calc = ", reg)


#tic();
#reconstructed_image = EPLLhalfQuadraticSplit(noisy_image,patchSize^2/noiseSD^2,patchSize, (1/noiseSD^2.0)*[1 4 8 16 32 64 128 256 512 1024 2048 4000 8000 20000 40000 80000],1, MAPGMM, true_image);
#toc();


#println("PSNR is: ",20*log10(1/std(reconstructed_image-true_image)), "\n");
#println("l1 distance is: ", sum(abs(reconstructed_image-true_image))/length(true_image), "\n");
#figure(3);imshow(reconstructed_image, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);

readline();
