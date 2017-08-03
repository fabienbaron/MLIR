using MAT #to import matlab GMM

type GMM
  nmodels::Float64
  dim::Float64
  covs::Array{Float64,3}
  invcovs::Array{Float64,3}
  chols
  mixweights::Array{Float64,2}
  means::Array{Float64,2}
end


function importGMM(filename)
  vars = get(matread(filename), "GMM", 0);
  #clumsy
  covs = get(vars, "covs", 0);
  invcovs = Array{Float64}(size(covs));
  chols = Array{typeof(covs[:,:,1])}(size(covs,3));
  for i=1:size(covs,3)
  invcovs[:,:,i]=inv(covs[:,:,i]);
  end

  for i=1:size(covs,3)
    chols[i]=chol(covs[:,:,i]);
  end
  GDict = GMM(get(vars, "nmodels",0), get(vars, "dim", 0), covs, invcovs, chols, get(vars,"mixweights", 0), get(vars,"means", 0));
end
