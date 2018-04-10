include("aradmm_core.jl")

mutable struct optsinfo
tol::Float64 #stop criterion, relative tolerance
maxiter::Int64  #max interation
τ::Float64  #initial stepsize
verbose::Int64 #verbose print 0: no print, 1: print every iteration 2: evaluate objective every iteration 3: more print out for debugging adaptive relaxed ADMM
adp_flag::Int64 # default method: ARADMM
adp_freq::Int64  #frequency for adaptive penalty
adp_start_iter::Int64 #start iteration of adaptive penalty
adp_end_iter::Int64 #end iteration of adaptive penalty
ϵcor::Float64  #threshold for correlation validation in ARADMM
beta_scale::Float64  #residual balancing parameter
res_scale::Float64  #residual balancing parameter
γ::Float64 # relaxation hyperparameter
end

function aradmm_cvis_denoising(vis_given::Array{Complex64,2}, H, Ht, HtH, mu::Float64, lam1::Float64, opts::optsinfo)
# demo total vational image denoising
# minimize  mu/2 ||Hx-f||^2 + lam1 |\grad x|

# stencil for gradient
sx = zeros(size(x_given));
sy = zeros(size(sx));
sx[1,1]=-1;
sx[1,end]=1;
fsx = fft(sx);
sy[1,1]=-1;
sy[end,1]=1;
fsy = fft(sy);
#gradient operator
grad2d = X -> cat(3, ifft(fsx.*fft(X)), ifft(fsy.*fft(X)));
grad2d_conj = G -> real(ifft( conj(fsx).*fft(G[:,:,1]) + conj(fsy).*fft(G[:,:,2]) ));

# mu/2 ||u-f||^2
h = x -> 0.5*mu*sum(abs2.(H*x-vis_given));  #regression term
#lam1 |x|
g = x -> lam1*norm(x[:],1); #regularizer
#objective
obj = (u, v) -> h(u)+g(grad2d(u));
# min mu/2 ||Hu-f||^2 + t/2 ||-A(u) + v + l/t||^2
#  opt condition:  (mu H'H + t A'A)*u = mu*H'f + A'(t*v+l)
rhsh = (v, l, t) -> mu*H'*vis_given+grad2d_conj(t*v+l);
#fs2 = conj(fsx).*fsx + conj(fsy).*fsy;
#solvh = (v, l, t) -> real.( ifft(1./(mu+t*fs2) .*fft(rhsh(v,l,t)))); #update u
solvh = (v, l, t) -> (mu*HtH + t*A'A)\rhsh(v,l,t);

# min lam1 |x| + 1/(2t)|| x - z ||^2
#  opt condition:  shrink(z, lam1*t)
shrink = (x,t) -> sign.(x).*max.(abs.(x) - t,0);
proxg = (z,t) -> shrink(z, lam1*t);
solvg = (au, l, t) -> proxg(au-l/t, 1/t); #update v
fA = grad2d;
fAt = grad2d_conj;
fB =  x -> -x;
fb = 0.0;
#opts.obj = obj; #objective function, used when verbose
## initialization
x0 = H'*vis_given;
l0 = ones(size(x0));
## ADMM solver
#tic
((sol,~,~), outs) = aradmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, obj, opts);
#outs.runtime  = toc
#sol = sol.u;
return (sol, outs)
end
