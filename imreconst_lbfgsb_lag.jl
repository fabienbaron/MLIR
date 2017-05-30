using FITSIO
using Lbfgsb
include("readoifits.jl")
include("oichi2.jl")
include("oiplot.jl")
include("EPLL.jl")
PyPlot.show()

pixellation = 0.101; # in mas
fitsfile = "2004true137.fits";
oifitsfile = "2004-data1.oifits";
x_true2d = (read((FITS(fitsfile))[1]));
patchsize = 8;
#  z = im2col(x_true2d,(patchsize,patchsize));
nx = size(x_true2d,1);
x_true1d= vec(x_true2d);
data = read_oifits(oifitsfile);
dft = setup_ft(data, nx, pixellation);

#initial image is a simple Gaussian
x_start = Array(Float64, nx, nx);
for i=1:nx
  for j=1:nx
    x_start[i,j] = exp(-((i-(nx+1)/2)^2+(j-(nx+1)/2)^2)/(2*(nx/6)^2));
  end
end
x_start = vec(x_start);#/sum(x_start);

patchsize = 8;
dict = importGMM("GMM_YSO.mat");
nsize = Int(sqrt(length(x_start)));
MAPGMM = (Z,patchsize,noiseSD,imsize)->aprxMAPGMM(Z,patchsize,noiseSD,imsize,dict);
noiseSD = 30.0;
beta = (1.0/noiseSD^2.0)*[10 50 100 500 1000 2000 5000 50000];
maxit  = [60 20 20 20 20 20 20 20];
niter = length(beta);
x = copy(x_start);
for i=1:niter
  println("beta = ", beta[i], "\n");
  z = MAPGMM(im2col(reshape(x, (nsize,nsize)),(patchsize,patchsize)), patchsize, beta[i]^-0.5,(nsize, nsize));
  crit = (xx,gg)->chi2andlag_fg(xx, gg, dft, data, z, beta[i]);
  f, x, numCall, numIter, status = lbfgsb( crit, x, lb=zeros(size(x_start)), ub=ones(size(x_start)), m=5, maxiter = maxit[i], iprint=1);
end
