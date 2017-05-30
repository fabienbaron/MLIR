using FITSIO;
using Base.Cartesian #for nloops in blockproc
#using PyPlot
#PyPlot.show()

include("EPLLhalfQuadraticSplit.jl")
#A = 1.0*reshape(1:4096,(64,64))'
A = randn(64,64);
B = im2col(A,(8,8));
# display patch i: reshape(B[:,i],(2,2))
ref=sum(B.^2) # objective function

#numerical gradient
delta=1e-4;
num_grad = copy(A).*0;

for x=1:64
  for y=1:64
    Aprime = copy(A);
    Aprime[x,y]+=delta;
    B = im2col(Aprime,(8,8));
    num_grad[x,y]=(sum(B.^2)-ref)/delta;
  end
end

analytic_grad =col2im(2.*B, (8,8), size(A), "sliding", "sum");

sum((num_grad - analytic_grad).^2)
