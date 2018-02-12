using PyPlot
using JLD

mutable struct GMM
  nmodels::Float64
  dim::Float64
  covs::Array{Float64,3}
  invcovs::Array{Float64,3}
  chols::Array{Array{Float64,2},1}
  mixweights::Array{Float64,1}
  means::Array{Float64,2}
end
# Removes DC component from image patches
# Data given as a matrix where each patch is one column vectors
# That is, the patches are vectorized.

function removeDC(X)
# Subtract local mean gray-scale value from each patch in X to give output Y
DC = mean(X,1);
Y = copy(X);
for i=1:size(X,2)
  @inbounds Y[:,i] -= DC[1,i];
end
return Y
end

function randsample(n, k, replace = false) # Sample with replacement
if replace == true
  y = (Array{Int64})(ceil.(n*rand(k)));
else
  if k > n
        println("K must be less than or equal to N for sampling without replacement.");
  end
  # If the sample is a sizeable fraction of the population,
  # just randomize the whole population (which involves a full
  # sort of n random values), and take the first k.
  if 4*k > n
    rp = randperm(n);
    y = rp[1:k];
  # If the sample is a small fraction of the population, a full sort
  # is wasteful.  Repeatedly sample with replacement until there are
  # k unique values.
  else
    x = zeros(n); # flags
    sumx = 0;
    while sumx < k
      x[(Array{Int64})(ceil.(n * rand(k-sumx)))] = 1; # sample w/replacement
      sumx = Int(sum(x)); # count how many unique elements so far
    end
    y = find(x.>0);
    y = y[randperm(k)];
  end
end
return y
end

function  RandPatchesFromImagesCell(MiniBatchSize,PatchSize,Images)
# return a set of MiniBatchSize patches sized PatchSizexPatchSize from the
# 3D array Images (WxHxN) all assumed to be of the same size
N = length(Images);
X = zeros(Float64, PatchSize,PatchSize,MiniBatchSize);
ns = randsample(N,MiniBatchSize,true);
for n=sort(unique(ns))
    inds = find(ns.==n);
    H, W = size(Images[n]);
    image_inds = sub2ind((H,W),randsample(H-PatchSize+1,length(inds),true),randsample(W-PatchSize+1,length(inds),true));
    for y=0:PatchSize-1
        for x=0:PatchSize-1
          X[y+1,x+1,inds] = Images[n][image_inds + y + x*H];
        end
    end
end
X = reshape(X,(PatchSize^2, MiniBatchSize));
return X
end

function loggausspdf2(X, sigma)
  # log pdf of Gaussian with zero mean
  d = size(X,1);
  R = chol(sigma);
  q = sum((R'\X).^2, 1);  # quadratic term (M distance)
  c = d*log(2.0*pi)+2.0*sum(log.(diag(R)));   # normalization constant
  y = -(c+q)/2;
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



function OnlineGMMEM(GMM0,DataSource,NumIterations,MiniBatchSize,OutputFile,T0,alpha,FirstBatchSize,removeFlatPatches)
  # ONLINEGMMEM - Learns a GMM model in an online manner
  #
  # Inputs:
  #
  #   GMM0 - if a GMM struct, then the GMM is initialized to the given GMM
  #   struct, otherwise this is the number of components to be learned in the
  #   mixture
  #
  #   DataSource - a function handle for a function which gets as a input the
  #   number of samples to return and returns them whichever way seems good
  #   to you. See "RandPatchesFromImagesCell" for an example of such a
  #   function
  #
  #   NumIterations - how many iterations to learn (in mini batches)
  #
  #   OutputFile - name of the file to output the intermediate GMM to
  #
  #   T0 and alpha - learning rate parameters
  #
  #   FirstBatchSize - size of first mini batch, it helps to make this larger
  #   than the mini batches
  #
  #   removeFlatPatches - if true, we remove any patch with low std before
  #   learning
  #
  # Outputs:
  #
  #   GMMopt - the resulting GMM model
  #   llh - the log likelihood of each mini batch given through iterations
  # Original matlab version by Daniel Zoran, 2012 - daniez@cs.huji.ac.il
  # New Julia port by Fabien Baron, 2017 - baron@chara.gsu.edu

  if (typeof(GMM0) == GMM) # GMM0 is a starting GMM
    GMMopt = copy(GMM0);
    K = GMMopt.nmodels;
  else #GMM0 was only the number of models
    K = GMM0;
    GMMopt = GMM(K, size(DataSource(1),1), zeros(size(DataSource(1),1),size(DataSource(1),1),K), zeros(size(DataSource(1),1),size(DataSource(1),1),K), Array{Array{Float64, 2}}(K), zeros(K), zeros(size(DataSource(1),1),K));
#    GMMopt = GMM(K, size(DataSource(1),1), zeros(size(DataSource(1),1),size(DataSource(1),1),K), zeros(K), zeros(size(DataSource(1),1),K));
  end

  llh = zeros(NumIterations);

  if (isdefined(:T0) == false)
    T0 = 500;
  end
  if (isdefined(:alpha) == false)
    alpha = 0.6;
  end
  if (isdefined(:FirstBatchSize) == false)
    FirstBatchSize = MiniBatchSize*10;
  end
  if (isdefined(:removeFlatPatches) == false)
    removeFlatPatches = false;
  end

  # first E step
  X = DataSource(FirstBatchSize);
  if removeFlatPatches # flat patches have a low stddev
    nonflat = find(std(X,1)>0.002);
    X = X[:,nonflat];
  end

  N = size(X,2);
  if (typeof(GMM0) != GMM)
  #  idx = randsample(N,K);
  #  m = X[:,idx];
  #  TEMP = broadcast(-, m'*X,sum(m.^2,1)'/2);
  #  label,~ = ind2sub(size(TEMP),vec((findmax(TEMP,(1,)))[2]));
  #  while K != length(unique(label))
  #    idx = randsample(N,K);
  #    m = X[:,idx];
  #    TEMP = broadcast(-, m'*X,sum(m.^2,1)'/2);
  #    label,~ = ind2sub(size(TEMP),vec((findmax(TEMP,(1,)))[2]));
  #  end
    label = ceil.(K*rand(1,N));
    R = full(sparse(collect(1:N),vec(label),ones(N),N,K));
    eta = 1;
  else
    # normal E step
    R = zeros(N,K);
    for k = 1:K
      R[:,k] = loggausspdf2(X,GMMopt.covs[:,:,k])';
    end

    R = broadcast(+,R,log.(GMMopt.mixweights));
    T = logsumexp(R,2);
    R = broadcast(+,R,T);
    R = exp(R);
    eta = (1+T0)^-alpha;
  end

  for t=1:NumIterations
    # M step
    s = vec(sum(R,1));
    # if there are no zero probabilites there, use this mini batch if (all(s>0))
    GMMopt.mixweights = GMMopt.mixweights*(1-eta) + eta*s/N;
    for k = 1:K
      sR = sqrt.(R[:,k]);
      indx = find(sR.>0);
      Xo = broadcast(*,X[:,indx],sR[indx]');
      if s[k]>0
        Sig = (Xo*Xo')/s[k];
        Sig = Sig + 1e-5*eye(size(Xo,1));
        # make sure all eigenvalues are larger than 0
        D,V = eig(Sig);
        D[find(D.<=0)] = 1e-5;
        Sig = V*diagm(D)*V';
        Sig = (Sig+Sig')/2;
        GMMopt.covs[:,:,k] = GMMopt.covs[:,:,k]*(1-eta) + eta*Sig;
      end
    end

    if t<10
      eta = eta/2;
    else
      eta = (t+T0)^-alpha;
    end

    # Get more data!
    if t<10
      X = DataSource(FirstBatchSize);
    else
      X = DataSource(MiniBatchSize);
    end
    if removeFlatPatches
      nonflat = find(std(X,1)>0.002);
      X = X[:,nonflat];
    end
    N = size(X,2);

    # E step
    R = zeros(N,K);

    # calculate the likelihood on the N-1 leading eigenvectors due to DC removal
    for k = 1:K
      D,V = eigs(GMMopt.covs[:,:,k],nev=size(GMMopt.covs,1)-1);
      tt = At_mul_B(V, X);
      R[:,k] = -((size(D,1))/2)*log(2*pi) - 0.5*sum(log.(D)) - 0.5*sum(tt.*(diagm(D)\tt),1)';
    end

    R = broadcast(+,R,log.(GMMopt.mixweights)');
    T = logsumexp(R,2);
    llh[t] = (sum(T)/N)/(size(X,1)-1)/log(2); # loglikelihood
    R = exp.(broadcast(-,R,T));

    # output
    println("Iteration ", t, " out of ", NumIterations, " logL: ", llh[t], "\n");
    fig = figure("Progress",figsize=(10,5))
    subplot(121)
    plot(llh[1:t],linestyle="-",marker="o", color="black")
    title("log likelihood")
    subplot(122)
    ax = gca()
    ax[:cla]()
    ylim(1e-4,1)
    yscale("log")
    plot(sort(GMMopt.mixweights,rev=true), linestyle="-",marker="o", color="black");
    title("GMM mixweights")
    if mod(t,5)==0
      PyPlot.draw();PyPlot.pause(0.01);
    end
  if mod(t,20)==0
    println("Saving status in ",OutputFile,"\n");
    save(OutputFile,"GMM",GMMopt, "t",t, "NumIterations",NumIterations,"eta",eta, "MiniBatchSize",MiniBatchSize, "llh",llh,"alpha",alpha,"T0", T0);
  end # end of for loop on iteration number
end

  GMMopt.invcovs = Array{Float64}(size(GMMopt.covs));
  GMMopt.chols = Array{typeof(GMMopt.covs[:,:,1])}(size(GMMopt.covs,3));
  for i=1:size(GMMopt.covs,3)
    GMMopt.invcovs[:,:,i]=inv(GMMopt.covs[:,:,i]);
    GMMopt.chols[i]=chol(GMMopt.covs[:,:,i]);
  end
  save(OutputFile,"GMM",GMMopt, "t", NumIterations, "NumIterations",NumIterations,"eta",eta, "MiniBatchSize",MiniBatchSize, "llh",llh,"alpha",alpha,"T0", T0);
  return GMMopt,llh
end
