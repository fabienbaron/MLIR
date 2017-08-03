using FITSIO;
using PyPlot;
using OptimPack;
PyPlot.show();
include("readoifits.jl")
include("oichi2.jl")
include("oiplot.jl")
include("EPLL.jl");
include("initial.jl")
dict = importGMM("GMM_YSO.mat");
oifitsfile = "2004-data1.oifits";
pixellation = 0.101; # in mas
nx = 128;
mu = 1;
data = read_oifits(oifitsfile);
dft = setup_ft(data, nx, pixellation);

niter = 100;
z = zeros(niter, nx*nx);
x = zeros(niter, nx*nx);
s = zeros(niter, nx*nx);
r = zeros(niter, nx*nx);
ztilde = zeros(niter, nx*nx);
xtilde = zeros(niter, nx*nx);
u = zeros(niter, nx*nx);
rho = zeros(niter);
snorm = zeros(niter);
rnorm = zeros(niter);
xnorm = zeros(niter);
unorm = zeros(niter);
znorm = zeros(niter);
tauprim = zeros(niter);
taudual = zeros(niter);
tempg=zeros(nx*nx);
#
# Initialization
#
z[1,:] = initial_image(60);
chi2_fg(z[1,:], tempg, dft, data);
u[1,:]=copy(tempg);
rho[2] = initial_rho(z, u, dft, data);

t=2;
while t<=niter

println("ADMM iteration: ", t);
#
# Step 1
#

xtilde[t,:] = z[t-1,:] - u[t-1,:]/rho[t];
x[t, :] = step1(rho[t], mu, xtilde[t,:]);

#
# Step 2
#

ztilde[t,:] = x[t,:] + u[t-1,:]/rho[t];
z[t,:] = step2(z[t-1,:], ztilde[t,:], dft, data, rho[t]);

#
# Step 3
#
u[t,:] = u[t-1,:] + rho[t]*(x[t,:]-z[t,:]);


#temporary
s[t,:] = rho[t]*(z[t,:]-z[t-1,:]);
r[t,:] = x[t,:] - z[t,:];

#
# Convergence testing
#
snorm[t] = sum(s[t,:].*s[t,:]);
rnorm[t] = sum(r[t,:].*r[t,:]);
xnorm[t] = sum(x[t,:].*x[t,:]);
znorm[t] = sum(z[t,:].*z[t,:]);
unorm[t] = sum(u[t,:].*u[t,:]);

#tauprim[t] = sqrt(N)*eps_abs + eps_rel*maximum([xnorm[t],znorm[t]]);
#taudual[t] = sqrt(N)*eps_abs + eps_rel*unorm[t];

# determine new rho
rho[t+1]=rho[t];


t+=1
end
