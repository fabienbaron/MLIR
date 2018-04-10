# general admm solver
# Julia version: Fabien Baron
# Matlab version: Zheng Xu
# objective functioin: min h(u)+g(v) st. Au+Bv=b
# input:
#   solvh: function update u from v,l
#   solvg: function update v from Au,l, note that solvg requires Au as
#       input instead of u to use the relaxation term
#   A: linear constraint Au + Bv = b
#   At: the transpose of A
#   B: linear constraint Au + Bv = b
#   b: linear constraint Au + Bv = b
#   v0: the initialization of v
#   l0: the initialization of dual varialbe \lambda
# output:
#   u,v,l : solution to the problem,
#
#   outs:
#       outs.pres  #primal residual
#       outs.dres  #dual residual
#       outs.mres  #monotone residual
#       outs.τs  #penalty parameter
#       outs.γs #relaxation parameter
#       outs.objs  #objective
#       outs.tols #relative residual
#       outs.iter  #total iteration before convergence


mutable struct outinfo
pres  #primal residual
dres  #dual residual
mres  #monotone residual
τs  #penalty parameter
γs #relaxation parameter
objs  #objective
tols #relative residual
iter  #total iteration before convergence
end

function aradmm_core(solvh, solvg, A, At, B, b, v0, l0, obj, opts) # returns (sol, outs, opts)
#A=fA; At=fAt; B=fB;b=fb; v0=x0; #for debug
#parameter, parse options
#general
maxiter = opts.maxiter;
tol = opts.tol; #relative tol for stop criterion
const minval = 1e-20; #max(opts.tol/10, 1e-20); #smallest value considered
τ = max(opts.τ, minval); #initial stepsize
adp = opts.adp_flag;
verbose = opts.verbose;
#ARADMM
freq = opts.adp_freq; #adaptive stepsize, update frequency
siter = max(opts.adp_start_iter-1, 1); #start iteration for adaptive stepsize, at least 1, then start at siter+1
eiter = min(opts.adp_end_iter, maxiter)+1; #end iteration for adaptive stepsize, at most the maximum iteration number
ϵcor = max(opts.ϵcor, minval);  #value to test correlation, curvature could be estimated or not
#Residual balance
bs = opts.beta_scale; #the scale for updating stepsize, τ = bs * τ or τ/bs, 2 in the paper
rs = opts.res_scale; #the scale for the criterion, pres/dres ~ rs or 1/rs, 0.1 in the paper
#relaxation parameter
γ = opts.γ;
const γ0 = 1.5;
const gmh = 1.9;
const gmg = 1.1;

#record
pres = zeros(maxiter, 1); # primal residual
dres = zeros(maxiter, 1); # dual residual
mres = zeros(maxiter, 1);  # the monotone residual
τs = zeros(maxiter+1, 1); # penalty
objs = zeros(maxiter, 1); # objective
tols = zeros(maxiter, 1); # relative residual
gms = zeros(maxiter+1, 1); # relaxation parameter
τs[1] = copy(τ);
gms[1] = copy(γ);

#initialize
v1 = copy(v0);
Bv1 = B(v1);
Bv0 = copy(Bv1);
l1 = copy(l0);
b_norm = norm(b);
iter=1;
u = []
v = []
l = []
Au0=[]
l_hat0=[]
for iter = 1:maxiter
    #update u
    u = solvh(v1, l1, τ);
    Au = A(u);
    #update v
    au =  γ*Au + (1.0-γ)*(b-Bv1);  #relax
    v = solvg(au, l1, τ);
    Bv = B(v);
    #update l
    l = l1 + τ*(b-au-Bv);
    #residual
    pres1 = b-Au-Bv;
    pres[iter] = norm(pres1[:]);  #primal residual
    dres1 = At(Bv-Bv1);
    dres[iter] = τ*norm(dres1[:]); #dual residual
    mres[iter] = τ*pres[iter]^2+τ*norm(Bv[:]-Bv1[:])^2; #monotone residual
    #stop criterion
    tmp = At(l);
    pres_norm = pres[iter]/maximum([norm(Au[:]),norm(Bv[:]),b_norm]);
    dres_norm = dres[iter]/norm(tmp[:]);
    tols[iter] = maximum([pres_norm, dres_norm]);
    if verbose>0
        objs[iter] = obj(u, v); #objective
        @printf("It: %d Obj: %e τ: %e Meth: %d tol: %e primresn: %e dualresn: %e \n", iter, objs[iter], τ, adp, tols[iter], pres_norm, dres_norm );
    end
    if tols[iter] < tol
        break;
    end

    ## adaptive stepsize
    if (adp==1)
         #AADMM with spectral penalty
            if iter == 1 #record at first iteration
                l0 = copy(l);
                l_hat0 = l1 + τ*(b-Au-Bv1);
                Bv0 = copy(Bv);
                Au0 = copy(Au);
            elseif mod(iter,freq)==0 && iter>siter && iter < eiter   #adaptive stepsize
                #l_hat
                l_hat = l1 + τ*(b-Au-Bv1);
                τ = aadmm_estimate(iter, τ, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, ϵcor, minval, verbose);
                # record for next estimation
                l0 = copy(l);
                l_hat0 = copy(l_hat);
                Bv0 = copy(Bv);
                Au0 = copy(Au);
            end #frequency if, AADMM
    elseif (adp==3) #residual balancing
            if iter>siter && iter < eiter
                if dres[iter] < pres[iter] * rs #dual residual is smaller, need large τ
                    τ = bs * τ;
                elseif pres[iter] < dres[iter] * rs #primal residual is smaller, need small τ
                    τ = τ/bs;
                    #else: same τ
                end
            end #converge if, RB
    elseif(adp==5) #ARADMM
            if iter == 1 #record at first iteration
                l0 = copy(l);
                l_hat0 = l1 + τ*(b-Au-Bv1);
                Bv0 = copy(Bv);
                Au0 = copy(Au);
            elseif mod(iter,freq)==0 && iter>siter && iter < eiter   #adaptive stepsize
                #l_hat
                l_hat = l1 + τ*(b-Au-Bv1);
                (τ, γ) = aradmm_estimate(iter, τ, γ, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, ϵcor, minval, verbose, gmh, gmg, γ0);
                # record for next estimation
                l0 = copy(l);
                l_hat0 = copy(l_hat);
                Bv0 = copy(Bv);
                Au0 = copy(Au);
            end #frequency if, AADMM
    end #adaptive switch
    #end of adaptivity
    τs[iter+1] = τ;
    gms[iter+1] = γ;
    Bv1 = copy(Bv);
    v1 = copy(v);
    l1 = copy(l);
    if verbose > 1
    figure(1)
    clf();
    imshow(u);
    figure(2);
    clf();
    subplot(2,1,1)
    semilogy(pres[1:iter], label="Primal residual")
    semilogy(dres[1:iter], label="Dual residual")
    semilogy(mres[1:iter], label="Monotonic residual")
    legend()
    subplot(2,1,2)
    semilogy(objs[1:iter], label="Objective function");
    end
end

outs = outinfo(pres[1:iter], #primal residual
    dres[1:iter], #dual residual
    mres[1:iter], #monotone residual
    τs[1:iter], #penalty parameter
    gms[1:iter], #relaxation parameter
    objs[1:iter], #objective
    tols[1:iter], #relative residual
    iter); #total iteration before convergence
return ((u,v,l), outs)
end

function curv_adaptive_BB(αSD, αMG)
#adaptive BB, reference: FASTA paper of Tom Golstein
tmph = αMG/αSD; #correlation
if tmph > .5
    τ_h = αMG;
else
    τ_h = αSD - 0.5*αMG;
end
return τ_h
end

function aradmm_estimate(iter, τ, γ, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, ϵcor, minval, verbose, gmh, gmg, γ0)
#inner product
ul_hat = sum(real(conj(Au-Au0).*(l_hat-l_hat0))); # <Δh,Δλhat>
vl = sum(real(conj(Bv-Bv0).*(l-l0))); # <Δg,Δλ>

#norm of lambda, lambda_hat
dl_hat = norm(vec(l_hat-l_hat0)); # ||Δλhat||
dl = norm(vec(l-l0)); # ||Δλ||

#norm of gradient change
du = norm(vec(Au-Au0)); # ||Δh||
dv = norm(vec(Bv-Bv0)); # ||Δg||

#flag to indicate whether the curvature can be estimated
hflag = false;
gflag = false;

#estimate curvature, only if it can be estimated
#use correlation/othogonal to test whether can be estimated
#use the previous stepsize when curvature cannot be estimated
if ul_hat > ϵcor*du*dl_hat + minval # αcor = ul_hat/du/dl_hat
    hflag = true;
    αSD = dl_hat^2/ul_hat;
    αMG = ul_hat/du^2;
    α = curv_adaptive_BB(αSD, αMG);
end
if vl > ϵcor*dv*dl + minval # βcor = vl/dv/dl
    gflag = true;
    βSD = dl^2/vl;
    βMG = vl/dv^2;
    β = curv_adaptive_BB(βSD, βMG);
end

if hflag && gflag
    γ = min(1 + 2*sqrt(α*β)/(α+β), γ0);
    τ = sqrt(α*β);
elseif hflag
    γ = copy(gmh); #1.9;
    τ = copy(α);
elseif gflag
    γ = copy(gmg); #1.1;
    τ = copy(β);
else
    γ = copy(γ0); #1.5;
end

if verbose == 3
    if ul_hat < 0 || vl < 0
        @printf("(%d) <u, l>=%f, <v, l>=%f\n", iter, ul_hat, vl);
    end
    if hflag
        @printf("(%d) corr_h=%f,  αSD=%f,  estimated τ=%f,  γ=%f \n", iter,ul_hat/du/dl_hat, αSD, τ, γ);
    end
    if gflag
        @printf("(%d) corr_g=%f,  βSD=%f,  estimated τ=%f γ=%f\n", iter,vl/dv/dl, βSD, τ, γ);
    end
    if ~hflag && ~gflag
        @printf("(%d) no curvature, corr_h=%f, corr_g=%f, τ=%f,  γ=%f\n", iter, ul_hat/du/dl_hat, vl/dv/dl, τ, γ);
    end
end
return (τ, γ)
end

## AADMM spectral penalty parameter
function aadmm_estimate(iter, τ, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, ϵcor, minval, verbose)
#inner product
ul_hat = sum(real(conj(Au-Au0).*(l_hat-l_hat0)));# <Δh,Δλhat>
vl = sum(real(conj(Bv-Bv0).*(l-l0)));# <Δg,Δλ>

#norm of lambda, lambda_hat
dl_hat = norm(vec(l_hat-l_hat0)); # ||Δλhat||
dl = norm(vec(l-l0)); # ||Δλ||

#norm of gradient change
du = norm(vec(Au-Au0)); # ||Δh||
dv = norm(vec(Bv-Bv0)); # ||Δg||

#flag to indicate whether the curvature can be estimated
hflag = false;
gflag = false;

#estimate curvature, only if it can be estimated
#use correlation/othogonal to test whether can be estimated
#use the previous stepsize when curvature cannot be estimated
if ul_hat > ϵcor*du*dl_hat + minval
    hflag = true;
    αSD = dl_hat^2/ul_hat;
    αMG = ul_hat/du^2;
    α = curv_adaptive_BB(αSD, αMG);
end
if vl > ϵcor*dv*dl + minval
    gflag = true;
    βSD = dl^2/vl;
    βMG = vl/dv^2;
    β = curv_adaptive_BB(βSD, βMG);
end

#if curvature can be estimated for both terms, balance the two
#if one of the curvature cannot be estimated, use the estimated one
#or use the previous stepsize to estimate
if hflag && gflag
    τ = sqrt(α*β);
elseif hflag
    τ = copy(α);
elseif gflag
    τ = copy(β);
end

if verbose == 3
    @printf("(%d) <u, l>=%f, <v, l>=%f\n", iter, ul_hat, vl);
    if hflag
        @printf("(%d) corr_h=%f,  αSD=%f,  estimated τ=%f \n", iter, ul_hat/du/dl_hat, αSD, τ);
    end
    if gflag
        @printf("(%d) corr_g=%f,  βSD=%f,  estimated τ=%f \n", iter, vl/dv/dl, βSD, τ);
    end
    if ~hflag && ~gflag
        @printf("(%d) no curvature, corr_h=%f, corr_g=%f, τ=%f \n", iter, ul_hat/du/dl_hat, vl/dv/dl, τ);
    end
end
return τ
end
