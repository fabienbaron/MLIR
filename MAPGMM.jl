include("importgmm.jl")
include("blockproc.jl")


function loggausspdf2(X, sigma)
# log pdf of Gaussian with zero mean
d = size(X,1);
R = chol(sigma);
q = sum((R'\X).^2, 1);  # quadratic term (M distance)
c = d*log(2.0*pi)+2.0*sum(log(diag(R)));   # normalization constant
y = -(c+q)/2;
end

function loggausspdf3(X, R)
# log pdf of Gaussian with zero mean
d = size(X,1);
q = sum((R'\X).^2, 1);  # quadratic term (M distance)
c = d*log(2.0*pi)+2.0*sum(log(diag(R)));   # normalization constant
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
Y = maximum(Z, dim);
Z = broadcast(-,X,Y)
S = Y + log(sum(exp.(Z),dim));
i = find(.~isfinite(Y));
if .~isempty(i)
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
