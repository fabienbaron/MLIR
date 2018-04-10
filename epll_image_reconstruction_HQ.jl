# Half quadratic minimization of chi2 + EPLL
using FITSIO
using OptimPack
using JLD
include("../OITOOLS.jl/oitools.jl")
include("EPLL.jl");
PyPlot.show()
oifitsfile = "2004-data1.oifits";
data = (readoifits(oifitsfile))[1,1]; # data can be split by wavelength, time, etc.
# Desired reconstruction parameters
# read the image file
fitsfile = "2004true137.fits";
pixsize = 0.1; # in mas/pixel
x_true = (read((FITS(fitsfile))[1])); nx = (size(x_true))[1]; x_true=vec(x_true);
dft = setup_dft(data.uv, nx, pixsize);
f_chi2 = chi2(x_true, dft, data);
#fft = setup_nfft(data, nx, pixsize);
#initial image is a simple Gaussian
 x_start = Array{Float64}(nx, nx);
     for i=1:nx
       for j=1:nx
         x_start[i,j] = exp(-((i-(nx+1)/2)^2+(j-(nx+1)/2)^2)/(2*(nx/6)^2));
       end
     end
 x_start = vec(x_start)/sum(x_start);
crit = (x,g)->chi2_centered_fg(x, g, dft, data);
x_sol = OptimPack.vmlmb(crit, x_start, verb=true, lower=0, maxiter=80, blmvm=false);

include("mlir_oichi2.jl")
# Setup EPLL
Gdict = load("GMM_YSO.jld","GMM");
np = Int(sqrt(Gdict.dim));
precalc1 = vec(im2col( reshape(1:(nx*nx),nx,nx),(np,np)));
precalc2 = counts(precalc1);
P=a->im2col(reshape(a,(nx,nx)),(np,np)); # decomposition into patches
Pt=a->( counts(precalc1,fweights(vec(a)))./precalc2 )'[1,:]; # transpose
x = deepcopy(x_start);
z = P(x);
βrange = [1e-6, 0.1, 0.5, 1,4,8,16,32,64,128,256,512,1024,2048,4000,8000,20000,40000,80000,160000,320000];
maxiter = 60;
for i=1:length(βrange)
  β=βrange[i]
  println("β=$(β)");
  #step 1
  crit = (x,g)->chi2_epll_hq_fg(x, g, dft, data, β, z);
  if i>1
    maxiter = 20
  end
  x = OptimPack.vmlmb(crit, x, verb=true, lower=0, maxiter=maxiter, blmvm=false);
  #step 2
  z = prox_GMM(P(x), 1/sqrt(β), Gdict);
  imdisp(x);
  #psnr = 20*log10(1/std(x-x_true));println("PSNR: ", psnr);
  # Diagnostics
  chi2all = chi2(x, dft, data, false);
  dist = β*0.5*sum((P(x)-z).^2);
  epll = EPLLz(z, Gdict);
  println("Crit =$(chi2all+dist+epll) Chi2 = $(chi2all) truedist = $(dist/β) epll = $(epll)");
end
