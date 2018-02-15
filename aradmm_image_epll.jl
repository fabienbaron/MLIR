include("aradmm_core.jl")

mutable struct optsinfo
tol::Float64 #stop criterion, relative tolerance
maxiter::Int64  #max interation
tau::Float64  #initial stepsize
verbose::Int64 #verbose print 0: no print, 1: print every iteration 2: evaluate objective every iteration 3: more print out for debugging adaptive relaxed ADMM
adp_flag::Int64 # default method: ARADMM
adp_freq::Int64  #frequency for adaptive penalty
adp_start_iter::Int64 #start iteration of adaptive penalty
adp_end_iter::Int64 #end iteration of adaptive penalty
orthval::Float64  #threshold for correlation validation in ARADMM
beta_scale::Float64  #residual balancing parameter
res_scale::Float64  #residual balancing parameter
gamma::Float64 # relaxation hyperparameter
end

function aradmm_image_epll(x_data::Array{Float64,2}, mu::Float64, lam1::Float64, opts::optsinfo, dict::GMM)
nx = size(x_data,1);
np = Int(sqrt(dict.dim));
precalc1 = vec(im2col( reshape(1:(nx*nx),nx,nx),(np,np)));
precalc2 = counts(precalc1);
P=a->im2col(a,(np,np)); # decomposition into patches
Pt=a->reshape( ( counts(precalc1,fweights(vec(a)))./precalc2 )', (nx, nx)); # transpose

# minimize  mu/2 ||x-f||^2 + lam1 EPLL(x)
# mu/2 ||u-f||^2
h = x -> 0.5*mu*norm(x[:]-x_data[:])^2;  #regression term
#lam1 |x|
g = x -> lam1*EPLLz(x, dict); #regularizer form (warning, typically works on Px, not x)
#objective
obj = (u, v) -> h(u)+g(P(u));
# min mu/2 ||u-f||^2 + t/2 ||-A(u) + v + l/t||^2
#  opt condition:  (mu + t A'A)*u = mu*f + A'(t*v+l)
solvh = (v, l, t) -> (mu*x_data+Pt(t*v+l))/(mu+t); #update u
# min lam1 EPLL(x) + 1/(2t)|| x - z ||^2
solvg = (au, l, t) -> prox_GMM(au-l/t, 1/(lam1*t), dict); #update v
fA = P;
fAt = Pt;
fB =  x -> -x;
fb = 0.0;
#opts.obj = obj; #objective function, used when verbose
## initialization
v0 = P(x_data);
l0 = ones(size(v0));
## ADMM solver
((sol,~,~), outs) = aradmm_core(solvh, solvg, fA, fAt, fB, fb, v0, l0, obj, opts);
return (sol, outs)
end
