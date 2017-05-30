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
