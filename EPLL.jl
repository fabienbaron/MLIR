mutable struct GMM
  nmodels::Float64
  dim::Float64
  covs::Array{Float64,3}
  invcovs::Array{Float64,3}
  chols::Array{Array{Float64,2},1}
  mixweights::Array{Float64,1}
  means::Array{Float64,2}
end

include("blockproc.jl")
function loggausspdf2(X, sigma)
  # log pdf of Gaussian with zero mean
  d = size(X,1);
  R = chol(sigma);
  q = sum((R'\X).^2, 1);  # quadratic term (M distance)
  c = d*log(2.0*pi)+2.0*sum(log.(diag(R)));   # normalization constant
  y = -(c+q)/2;
end

function loggausspdf3(X, R)
  # log pdf of Gaussian with zero mean
  d = size(X,1);
  q = sum((R'\X).^2, 1);  # quadratic term (M distance)
  c = d*log(2.0*pi)+2.0*sum(log.(diag(R)));   # normalization constant
  y = -(c+q)/2;
end


function mixturelog(GS, Y, SigmaNoise)
  PYZ = zeros(round(Int, GS.nmodels),size(Y,2));
  for i=1:Int(GS.nmodels)
    PYZ[i,:] = log(GS.mixweights[i]) + loggausspdf2(Y,GS.covs[:,:,i]+SigmaNoise);
  end
  PYZ
end


function logsumexp(X, dim=1)
# Compute log(sum(exp(X))) while avoiding numerical underflow.
# subtract the largest in each column
Y = maximum(X, dim);
Z = broadcast(-,X,Y)
S = Y + log.(sum(exp.(Z),dim));
i = find(.~isfinite.(Y));
if ~isempty(i)
  S[i] = Y[i];
end
return S
end

function EPLL(x, GDict) # regularizer value
  patchSize = Int(sqrt(GDict.dim));
  xmin = minimum(x);
  xmax = maximum(x);
  xrescl = (x - xmin)/xmax;
  patches = im2col(xrescl,(patchSize,patchSize));
  mean_patches = mean(patches,1);
  for i=1:size(patches,2)
    @inbounds patches[:,i] -= mean_patches[i];
  end
  Pmodels = zeros(round(Int, GDict.nmodels),size(patches,2));
  for k=1:Int(GDict.nmodels)
    Pmodels[k,:] = log(GDict.mixweights[k]) + loggausspdf2(patches,GDict.covs[:,:,k]);
  end
  P = logsumexp(Pmodels);
  f=-sum(P)/size(patches,2)
end



function EPLL_fg(x, g, GDict) # regularizer value plus gradient
  patchSize = Int(sqrt(GDict.dim));
  xmin = minimum(x);
  xmax = maximum(x);
  xrescl = (x - xmin)/xmax;
  patches = im2col(xrescl,(patchSize,patchSize));
  npatches = size(patches,2)
  nmodels = Int(GDict.nmodels)
  mean_patches = mean(patches,1);
  for i=1:npatches
    @inbounds patches[:,i] -= mean_patches[i];
  end
  Pmodels = zeros(round(Int, GDict.nmodels),size(patches,2));
  for k=1:nmodels
    Pmodels[k,:] = log(GDict.mixweights[k]) + loggausspdf2(patches,GDict.covs[:,:,k]);
  end
  P = logsumexp(Pmodels); # log p(PiX), hence vector of size=npatches
  f = -sum(P)/size(patches,2);
  GPmodels = zeros(round(Int, GDict.nmodels),(patchSize*patchSize), size(patches,2));
  for i=1:npatches
    for k=1:nmodels
      GPmodels[k,:,i] = -exp(Pmodels[k,i]-P[i])*(GDict.invcovs[:,:,k]*patches[:,i])
    end
  end
  GP = squeeze(sum(GPmodels,1),1);
  g[:,:]=-col2im(GP, (patchSize, patchSize), size(x), "sliding", "sum")/size(patches,2);
  return f
end


function prox_GMM(z0, alpha, dict)
  # proximal operator/resolvent for the EPLL prior
  # i.e. solution of:
  #   zhat = argmin{ alpha * EPPL(z) + 0.5*||z-z0||^2 }
  #       z - initial patches
  #      np - patch size
  #   alpha - EPLL coefficient
  #    dict - dictionary the gaussian mixture model structure
  np = Int(sqrt(size(z0,1)))
  W = alpha^2*eye(np^2);
  # remove DC component
  mean_z0 = mean(z0,1);
  broadcast!(-, z0, z0, mean_z0);
  PYZ = mixturelog(dict, z0, W);
  #PYZ = mixturelog(dict, z0, 0);
  ks = ind2sub(size(PYZ), vec(findmax(PYZ,1)[2]))[1]';
  zhat = zeros(size(z0));
  for i=1:Int(dict.nmodels)
    inds = find(ks.==i);
    zhat[:,inds] = ( (dict.covs[:,:,i]+W)\(dict.covs[:,:,i]*z0[:,inds] + W*repmat(dict.means[:,i],1,length(inds)) ) );
  end
  broadcast!(+, zhat, zhat, mean_z0)
  return zhat
end

#
# function prox_GMM_grad(z0, alpha, dict)
# using OptimPack
# # gradient method
# function prox_fg(z, g, alpha, z0)
# chi2_f = 0.5*norm(z-z0)^2
# chi2_g = z-z0;
# nx = Int(sqrt(length(x)));
# reg_g = zeros(Float64, (nx, nx));
# reg_f = EPLL_fg(reshape(x, (nx,nx)), reg_g, dict);
# g[:] = chi2_g + mu*vec(reg_g);
# return chi2_f + mu*reg_f
# end
# g = zeros(size(z0));
# crit = (z,g)->prox_fg(z, g, alpha, z0);
# z = OptimPack.vmlmb(crit, copy(z0), verb=true, lower=0, maxiter=80, blmvm=false);
# return z
# end



#
# function aprxMAPGMM(patches, patchSize, noiseSD, imsize, GS)
#   # approximate GMM MAP estimation - a single iteration of the "hard version"
#   # EM MAP procedure (see paper for a reference)
#   # Inputs:
#   #   patches - the noisy patches (in columns)
#   #   noiseSD - noise standard deviation
#   #   imsize - size of the original image (not used in this case, but may be
#   #   used for non local priors)
#   #   GS - the gaussian mixture model structure
#   #   excludeList - used only for inpainting, misleading name - it's a list
#   #   of patch indices to use for estimation, the rest are just ignored
#   #   SigmaNoise - if the noise is non-white, this is the noise covariance
#   #   matrix
#   # Outputs:
#   #   Xhat - the restored patches
#   # Supports general noise covariance matrices
#   SigmaNoise = noiseSD^2*eye(patchSize^2);
#   # remove DC component
#   mean_patches = mean(patches,1);
#   for i=1:size(patches,2)
#     @inbounds patches[:,i] -= mean_patches[i];
#   end
#   PYZ = mixturelog(GS, patches, SigmaNoise)
#   ks = ind2sub(size(PYZ), vec(findmax(PYZ,1)[2]))[1]'
#   Xhat = zeros(size(patches));
#   for i=1:Int(GS.nmodels)
#     inds = find(ks.==i);
#     Xhat[:,inds] = ( (GS.covs[:,:,i]+SigmaNoise)\(GS.covs[:,:,i]*patches[:,inds] + SigmaNoise*repmat(GS.means[:,i],1,length(inds)) ) );
#   end
#   for i=1:size(patches,2)
#     @inbounds Xhat[:,i] += mean_patches[i];
#   end
#   Xhat
# end





using StatsBase
#
#
# using StatsBase
# function EPLLhalfQuadraticSplit(noisy_image,lambda,patchSize,betas,T, MAPGMM, true_image)
#   #% estimate the "real" noise standard deviation from lambda
#   RealNoiseSD = sqrt(1/(lambda/patchSize^2));
#   # initialize with the noisy image
#   image_size = size(noisy_image)
#   current_image = copy(noisy_image);
#   mm = size(true_image,1);
#   nn = size(true_image,2);
#   temp = im2col( reshape(1:(mm*nn),mm,nn),(patchSize,patchSize));
#
#   #% go through all values of noise levels
#   for b=betas
#     println("beta = ", b, "\n")
#     # Z step, extract all overlapping patches from the current image estimate
#     patches = im2col(current_image,(patchSize,patchSize));
#     #       calculate the MAP estimate for patches using the given prior
#     MAPpatches = MAPGMM(patches, patchSize, b^-0.5,image_size);
#     #         X step, average the pixels in MAPpatches
#     avg = counts(temp[:],WeightVec(MAPpatches[:]))./counts(temp[:]);
#     #       and calculate the current estimate for the clean image
#     #avg[avg.<0]=0;
#     current_image = noisy_image*lambda/(lambda+b*patchSize^2) + reshape(avg', mm, nn)*(b*patchSize^2/(lambda+b*patchSize^2));
#     #        current_image[current_image.>1]=1;
#     #        current_image[current_image.<0]=0;
#
#     figure(3); imshow(current_image, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
#     psnr = 20*log10(1/std(current_image-true_image));
#     println("PSNR: ", psnr);
#     println("l1 distance: ", sum(abs.(current_image-true_image))/length(true_image), "\n");
#   end
#   #% clip values to be between 1 and 0, hardly changes performance
#   current_image[current_image.>1]=1;
#   current_image[current_image.<0]=0;
#
#   return current_image
# end

#
# function EPLL_denoise_1D(noisy_image,noiseSD,dict)
#   patchSize = Int(sqrt(dict.dim));
#   beta = (1/noiseSD^2.0)*[1 4 8 16 32 64 128 256 512 1024 2048 4000 8000 20000 40000];
#   lambda = patchSize^2/noiseSD^2;
#   mm = Int(sqrt(size(noisy_image,1)));
#   nn = Int(sqrt(size(noisy_image,1)));
#   init_image = reshape(noisy_image, (mm,nn));
#   xmin = minimum(init_image);
#   xmax = maximum(init_image);
#   init_image  = (init_image - xmin)/xmax;
#   current_image = copy(init_image);
#   imsize = size(current_image);
#   MAPGMM = (Z,patchSize,noiseSD,imsize)->aprxMAPGMM(Z,patchSize,noiseSD,imsize,dict);
#   temp = im2col( reshape(1:(mm*nn),mm,nn),(patchSize,patchSize));
#   for b=beta
#     println("beta = ", b, "\n")
#     # Z step, extract all overlapping patches from the current image estimate
#     patches = im2col(current_image,(patchSize,patchSize));
#     # calculate the MAP estimate for patches using the given prior
#     MAPpatches = MAPGMM(patches, patchSize, b^-0.5,imsize);
#     #  X step, average the pixels in MAPpatches
#     avg = counts(temp[:],WeightVec(MAPpatches[:]))./counts(temp[:]);
#     #  and calculate the current estimate for the clean image
#     current_image = init_image*lambda/(lambda+b*patchSize^2) + reshape(avg', mm, nn)*(b*patchSize^2/(lambda+b*patchSize^2));
#     figure(2); imshow(current_image, ColorMap("gist_heat"), interpolation="none");PyPlot.draw();PyPlot.pause(0.05);
#   end
#   #% clip values to be between 1 and 0, hardly changes performance
#   current_image[current_image.>1]=1;
#   current_image[current_image.<0]=0;
#   return (current_image*xmax+xmin)
# end
#
#
#
# function EPLL_denoise_2D(noisy_image,noiseSD,dict)
#   patchSize = Int(sqrt(dict.dim));
#   beta = (1/noiseSD^2.0)*[1 4 8 16 32 64 128 256 512 1024 2048 4000 8000 20000 40000];
#   lambda = patchSize^2/noiseSD^2;
#   mm = size(noisy_image,1);
#   nn = size(noisy_image,2);
#   current_image = copy(noisy_image);
#   imsize = size(current_image);
#   temp = im2col( reshape(1:(mm*nn),mm,nn),(patchSize,patchSize));
#   for b=beta
#     println("beta = ", b, "\n")
#     # Z step, extract all overlapping patches from the current image estimate
#     patches = im2col(current_image,(patchSize,patchSize));
#     # calculate the MAP estimate for patches using the given prior
#     MAPpatches = aprxMAPGMM(patches,patchSize,b^-0.5,imsize,dict);
#     #  X step, average the pixels in MAPpatches
#     avg = counts(temp[:],WeightVec(MAPpatches[:]))./counts(temp[:]);
#     #  and calculate the current estimate for the clean image
#     current_image = noisy_image*lambda/(lambda+b*patchSize^2) + reshape(avg', mm, nn)*(b*patchSize^2/(lambda+b*patchSize^2));
#     figure(2); imshow(current_image, ColorMap("gist_heat"), interpolation="none");PyPlot.draw();PyPlot.pause(0.05);
#   end
#   #% clip values to be between 1 and 0, hardly changes performance
#   current_image[current_image.>1]=1;
#   current_image[current_image.<0]=0;
#   return current_image
# end
#
#
#
#

# function EPLLhalfQuadraticSplit(noisy_image,lambda,patchSize,betas,T, MAPGMM, true_image)
#   #% estimate the "real" noise standard deviation from lambda
#   RealNoiseSD = sqrt(1/(lambda/patchSize^2));
#   # initialize with the noisy image
#   image_size = size(noisy_image)
#   current_image = copy(noisy_image);
#   mm = size(true_image,1);
#   nn = size(true_image,2);
#   temp = im2col( reshape(1:(mm*nn),mm,nn),(patchSize,patchSize));
#
#   #% go through all values of noise levels
#   for b=betas
#     println("beta = ", b, "\n")
#     # Z step, extract all overlapping patches from the current image estimate
#     patches = im2col(current_image,(patchSize,patchSize));
#     #       calculate the MAP estimate for patches using the given prior
#     MAPpatches = MAPGMM(patches, patchSize, b^-0.5,image_size);
#     #         X step, average the pixels in MAPpatches
#     avg = counts(temp[:],WeightVec(MAPpatches[:]))./counts(temp[:]);
#     current_image = noisy_image*lambda/(lambda+b*patchSize^2) + reshape(avg', mm, nn)*(b*patchSize^2/(lambda+b*patchSize^2));
#   #  figure(3); imshow(current_image, ColorMap("gray"));PyPlot.draw();PyPlot.pause(0.05);
#     psnr = 20*log10(1/std(current_image-true_image));
#     println("PSNR: ", psnr);
#     println("l1 distance: ", sum(abs.(current_image-true_image))/length(true_image), "\n");
#   end
#   #% clip values to be between 1 and 0, hardly changes performance
#   current_image[current_image.>1]=1;
#   current_image[current_image.<0]=0;
#
#   return current_image
# end

function EPLLhalfQuadraticSplitNew(y::Array{Float64,2}, sigma::Float64, np::Int64, betas::Array{Float64,1}, x_true::Array{Float64,2}, dict::GMM)
  # initialize with the noisy image
  x = copy(y);
  nx = size(x_true,1);
  precalc1 = vec(im2col( reshape(1:(nx*nx),nx,nx),(np,np)));
  precalc2 = counts(precalc1);
  P=a->im2col(a,(np,np)); # decomposition into patches
  Pt=a->reshape( ( counts(precalc1,fweights(vec(a)))./precalc2 )', (nx, nx)); # transpose
  λ = 1/sigma^2;
  #% go through all values of noise levels
  for β=betas
    println("beta = ", β, "\n")
    # Z step, extract all overlapping patches from the current image estimate, then calculate the resolvent
    #z = MAPGMM(P(x), np, sigma/sqrt(b), size(y));
    z = prox_GMM(P(x), 1/sqrt(β), dict);
    # X step, average the pixels in MAPpatches and calculate the current estimate for the clean image
    x = (λ*y + β*Pt(z))/(λ+β);
    figure(3); imshow(x, ColorMap("gist_heat"));PyPlot.draw();PyPlot.pause(0.05);
    println("EPLL: ", EPLL(x, dict), "\t F: ",  1/sigma^2*norm(x-y)^2+EPLL(x, dict));
    psnr = 20*log10(1/std(x-x_true));
    println("PSNR: ", psnr);
    println("l1 distance: ", sum(abs.(x-x_true))/length(x_true), "\n");
  end
  #% clip values to be between 1 and 0, hardly changes performance
  x[x.>1]=1;
  x[x.<0]=0;
  return x
end

function chi2andlag_fg(x, g, dft, data, z, beta) # criterion function plus its gradient w/r x#
#nx2 = length(x);
cvis_model = image_to_cvis(x, dft);
# compute observables from all cvis
v2_model = cvis_to_v2(cvis_model, data.indx_v2);
t3_model, t3amp_model, t3phi_model = cvis_to_t3(cvis_model, data.indx_t3_1, data.indx_t3_2 ,data.indx_t3_3);
chi2_v2 = sum( ((v2_model - data.v2_data)./data.v2_data_err).^2);
chi2_t3amp = sum( ((t3amp_model - data.t3amp_data)./data.t3amp_data_err).^2);
chi2_t3phi = sum( (mod360(t3phi_model - data.t3phi_data)./data.t3phi_data_err).^2);

g_v2 = 4.0*sum(((v2_model-data.v2_data)./data.v2_data_err.^2).*real(conj(cvis_model[data.indx_v2]).*dft[data.indx_v2,:]),1);
g_t3amp = 2.0*sum(((t3amp_model-data.t3amp_data)./data.t3amp_data_err.^2).*
                  (   real( conj(cvis_model[data.indx_t3_1]./abs(cvis_model[data.indx_t3_1])).*dft[data.indx_t3_1,:]).*abs(cvis_model[data.indx_t3_2]).*abs(cvis_model[data.indx_t3_3])       + real( conj(cvis_model[data.indx_t3_2]./abs(cvis_model[data.indx_t3_2])).*dft[data.indx_t3_2,:]).*abs(cvis_model[data.indx_t3_1]).*abs(cvis_model[data.indx_t3_3])+ real( conj(cvis_model[data.indx_t3_3]./abs(cvis_model[data.indx_t3_3])).*dft[data.indx_t3_3,:]).*abs(cvis_model[data.indx_t3_1]).*abs(cvis_model[data.indx_t3_2])),1);

t3model_der = dft[data.indx_t3_1,:].*cvis_model[data.indx_t3_2].*cvis_model[data.indx_t3_3] + dft[data.indx_t3_2,:].*cvis_model[data.indx_t3_1].*cvis_model[data.indx_t3_3] + dft[data.indx_t3_3,:].*cvis_model[data.indx_t3_1].*cvis_model[data.indx_t3_2];
g_t3phi =360./pi*sum(((mod360(t3phi_model-data.t3phi_data)./data.t3phi_data_err.^2)./abs2(t3_model)).*(-imag(t3_model).*real(t3model_der)+real(t3_model).*imag(t3model_der)),1);
imdisp(x);
flux = sum(x);
nx = Int64(sqrt(length(x)));
patchsize = Int64(sqrt(size(z,1)));
x_2d = reshape(x, (nx,nx));
px = im2col(x_2d,(patchsize,patchsize));
lag_f = beta*sum((px-z).^2);
lag_g = 2.*beta*vec(col2im(px-z, (patchsize, patchsize), (nx,nx), "sliding", "sum"));
g[:] = squeeze(g_v2 + g_t3amp + g_t3phi,1) + lag_g;
g[:] = (g - sum(x.*g) / flux ) / flux ; # gradient correction to take into account the non-normalized image
println("V2: ", chi2_v2/data.nv2, " T3A: ", chi2_t3amp/data.nt3amp, " T3P: ", chi2_t3phi/data.nt3phi," Flux: ", flux, " LAG: ", lag_f);
return (chi2_v2 + chi2_t3amp + chi2_t3phi) + lag_f
end
