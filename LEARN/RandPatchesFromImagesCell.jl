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
