
function chi2_epll_hq_fg(x::Array{Float64,1}, g::Array{Float64,1}, dft, data, β, z)
flux = sum(x);
cvis_model = image_to_cvis_dft(x, dft);
v2_model = cvis_to_v2(cvis_model, data.indx_v2);
t3_model, t3amp_model, t3phi_model = cvis_to_t3(cvis_model, data.indx_t3_1, data.indx_t3_2 ,data.indx_t3_3);
chi2_v2 = sum( ((v2_model - data.v2)./data.v2_err).^2);
chi2_t3amp = sum( ((t3amp_model - data.t3amp)./data.t3amp_err).^2);
chi2_t3phi = sum( (mod360(t3phi_model - data.t3phi)./data.t3phi_err).^2);
# centering
rho = 1e4
cent_g = zeros(size(x))
cent_f = reg_centering(x, cent_g)
# epll
epll_hq_f = 0.5*sum((P(x)-z).^2)
epll_hq_g = Pt(P(x)-z);
g_v2 = 4.0*sum(((v2_model-data.v2)./data.v2_err.^2).*real(conj(cvis_model[data.indx_v2]).*dft[data.indx_v2,:]),1);
g_t3amp = 2.0*sum(((t3amp_model-data.t3amp)./data.t3amp_err.^2).*
(   real( conj(cvis_model[data.indx_t3_1]./abs.(cvis_model[data.indx_t3_1])).*dft[data.indx_t3_1,:]).*abs.(cvis_model[data.indx_t3_2]).*abs.(cvis_model[data.indx_t3_3])       + real( conj(cvis_model[data.indx_t3_2]./abs.(cvis_model[data.indx_t3_2])).*dft[data.indx_t3_2,:]).*abs.(cvis_model[data.indx_t3_1]).*abs.(cvis_model[data.indx_t3_3])+ real( conj(cvis_model[data.indx_t3_3]./abs.(cvis_model[data.indx_t3_3])).*dft[data.indx_t3_3,:]).*abs.(cvis_model[data.indx_t3_1]).*abs.(cvis_model[data.indx_t3_2])),1);
t3model_der = dft[data.indx_t3_1,:].*cvis_model[data.indx_t3_2].*cvis_model[data.indx_t3_3] + dft[data.indx_t3_2,:].*cvis_model[data.indx_t3_1].*cvis_model[data.indx_t3_3] + dft[data.indx_t3_3,:].*cvis_model[data.indx_t3_1].*cvis_model[data.indx_t3_2];
g_t3phi =360./pi*sum(((mod360(t3phi_model-data.t3phi)./data.t3phi_err.^2)./abs2.(t3_model)).*(-imag(t3_model).*real(t3model_der)+real(t3_model).*imag(t3model_der)),1);

g[1:end] = vec(g_v2 + g_t3amp + g_t3phi);
g[1:end] = (g - sum(vec(x).*g) / flux ) / flux  +  rho * cent_g + β*epll_hq_g; # gradient correction to take into account the non-normalized image
println("V2: ", chi2_v2/data.nv2, " T3A: ", chi2_t3amp/data.nt3amp, " T3P: ", chi2_t3phi/data.nt3phi," Flux: ", flux, " CENT: ", cent_f, " CDG ", cdg(reshape(x,nx,nx)))
return chi2_v2 + chi2_t3amp + chi2_t3phi + rho * cent_f + β*epll_hq_f
end



function TVSQ_fg(x, tv_g)
  nx = Int64(ceil(sqrt(length(x))))
  y = reshape(x,(nx,nx))
  g = zeros(size(y))
  # Add total variation regularization
  tv = sum( (y[2:end, :]-y[1:end-1, :]).^2 ) + sum( (y[:, 2:end]-y[:, 1:end-1]).^2)

  # first term, x differential
  g[1,:] = 2*(y[1,:]-y[2,:]);
  g[end,:] =  2*(y[end,:]-y[end-1,:]);
  g[2:end-1,:] =  4*y[2:end-1,:] - 2*y[1:end-2,:] - 2*y[3:end,:];

  # second term, y differential
  g[:,1] += 2*(y[:,1]-y[:,2])
  g[:, end] +=  2*(y[:,end]-y[:,end-1])
  g[:, 2:end-1] +=  4*y[:,2:end-1] - 2*y[:,1:end-2] - 2*y[:,3:end];

  tv_g[:] = g
  return tv
end

function TVSQ_f(x)
  nx = Int64(ceil(sqrt(length(x))))
  y = reshape(x,(nx,nx))
  g = zeros(size(y))
  # Add total variation regularization
  tv = sum( (y[2:end, :]-y[1:end-1, :]).^2 ) + sum( (y[:, 2:end]-y[:, 1:end-1]).^2)
  return tv
end








function chi2_TVSQ_fg(x::Array{Float64,1}, g::Array{Float64,1},μ, dft, data)
reg_g = zeros(size(x));
reg_f = TVSQ_fg(x, reg_g);
flux = sum(x);
cvis_model = image_to_cvis_dft(x, dft);
v2_model = cvis_to_v2(cvis_model, data.indx_v2);
t3_model, t3amp_model, t3phi_model = cvis_to_t3(cvis_model, data.indx_t3_1, data.indx_t3_2 ,data.indx_t3_3);
chi2_v2 = sum( ((v2_model - data.v2)./data.v2_err).^2);
chi2_t3amp = sum( ((t3amp_model - data.t3amp)./data.t3amp_err).^2);
chi2_t3phi = sum( (mod360(t3phi_model - data.t3phi)./data.t3phi_err).^2);
# centering
rho = 1e4
cent_g = zeros(size(x))
cent_f = reg_centering(x, cent_g)
g_v2 = 4.0*sum(((v2_model-data.v2)./data.v2_err.^2).*real(conj(cvis_model[data.indx_v2]).*dft[data.indx_v2,:]),1);
g_t3amp = 2.0*sum(((t3amp_model-data.t3amp)./data.t3amp_err.^2).*
(   real( conj(cvis_model[data.indx_t3_1]./abs.(cvis_model[data.indx_t3_1])).*dft[data.indx_t3_1,:]).*abs.(cvis_model[data.indx_t3_2]).*abs.(cvis_model[data.indx_t3_3])       + real( conj(cvis_model[data.indx_t3_2]./abs.(cvis_model[data.indx_t3_2])).*dft[data.indx_t3_2,:]).*abs.(cvis_model[data.indx_t3_1]).*abs.(cvis_model[data.indx_t3_3])+ real( conj(cvis_model[data.indx_t3_3]./abs.(cvis_model[data.indx_t3_3])).*dft[data.indx_t3_3,:]).*abs.(cvis_model[data.indx_t3_1]).*abs.(cvis_model[data.indx_t3_2])),1);
t3model_der = dft[data.indx_t3_1,:].*cvis_model[data.indx_t3_2].*cvis_model[data.indx_t3_3] + dft[data.indx_t3_2,:].*cvis_model[data.indx_t3_1].*cvis_model[data.indx_t3_3] + dft[data.indx_t3_3,:].*cvis_model[data.indx_t3_1].*cvis_model[data.indx_t3_2];
g_t3phi =360./pi*sum(((mod360(t3phi_model-data.t3phi)./data.t3phi_err.^2)./abs2.(t3_model)).*(-imag(t3_model).*real(t3model_der)+real(t3_model).*imag(t3model_der)),1);
g[1:end] = vec(g_v2 + g_t3amp + g_t3phi);
g[1:end] = (g - sum(vec(x).*g) / flux ) / flux  +  rho * cent_g + μ*reg_g; # gradient correction to take into account the non-normalized image
imdisp(x)
println("V2: ", chi2_v2/data.nv2, " T3A: ", chi2_t3amp/data.nt3amp, " T3P: ", chi2_t3phi/data.nt3phi," Flux: ", flux, " CENT: ", cent_f, " CDG ", cdg(reshape(x,nx,nx)))
return chi2_v2 + chi2_t3amp + chi2_t3phi + rho * cent_f + μ*reg_f
end















# gnum = zeros(size(x_true))
# fref =  TVSQ_f(x_true)
# x_fake=deepcopy(x_true)
# for i=1:length(x_true)
#   x_fake[i] = x_true[i] + 1e-10
#   gnum[i] = (TVSQ_f(x_fake) - fref)/1e-10
#   x_fake[i] = x_true[i]
# end



#
# function proj_positivity(ztilde)
#   z = copy(ztilde)
#   z[ztilde.>0]=0
#   return z
# end

# function chi2_epll_fg(x, g, dft, data, GDict) # criterion function plus its gradient w/r x
#   #nx2 = length(x);
#   cvis_model = image_to_cvis_dft(x, dft);
#   # compute observables from all cvis
#   v2_model = cvis_to_v2(cvis_model, data.indx_v2);
#   t3_model, t3amp_model, t3phi_model = cvis_to_t3(cvis_model, data.indx_t3_1, data.indx_t3_2 ,data.indx_t3_3);
#   chi2_v2 = sum( ((v2_model - data.v2)./data.v2_err).^2);
#   chi2_t3amp = sum( ((t3amp_model - data.t3amp)./data.t3amp_err).^2);
#   chi2_t3phi = sum( (mod360(t3phi_model - data.t3phi)./data.t3phi_err).^2);
#
#   g_v2 = 4.0*sum(((v2_model-data.v2)./data.v2_err.^2).*real(conj(cvis_model[data.indx_v2]).*dft[data.indx_v2,:]),1);
#   g_t3amp = 2.0*sum(((t3amp_model-data.t3amp)./data.t3amp_err.^2).*
#   (   real( conj(cvis_model[data.indx_t3_1]./abs.(cvis_model[data.indx_t3_1])).*dft[data.indx_t3_1,:]).*abs.(cvis_model[data.indx_t3_2]).*abs.(cvis_model[data.indx_t3_3])       + real( conj(cvis_model[data.indx_t3_2]./abs.(cvis_model[data.indx_t3_2])).*dft[data.indx_t3_2,:]).*abs.(cvis_model[data.indx_t3_1]).*abs.(cvis_model[data.indx_t3_3])+ real( conj(cvis_model[data.indx_t3_3]./abs.(cvis_model[data.indx_t3_3])).*dft[data.indx_t3_3,:]).*abs.(cvis_model[data.indx_t3_1]).*abs.(cvis_model[data.indx_t3_2])),1);
#
#   t3model_der = dft[data.indx_t3_1,:].*cvis_model[data.indx_t3_2].*cvis_model[data.indx_t3_3] + dft[data.indx_t3_2,:].*cvis_model[data.indx_t3_1].*cvis_model[data.indx_t3_3] + dft[data.indx_t3_3,:].*cvis_model[data.indx_t3_1].*cvis_model[data.indx_t3_2];
#   g_t3phi =360./pi*sum(((mod360(t3phi_model-data.t3phi)./data.t3phi_err.^2)./abs2(t3_model)).*(-imag(t3_model).*real(t3model_der)+real(t3_model).*imag(t3model_der)),1);
#   imdisp(x);
#   mu = 1;
#   nx = Int(sqrt(length(x)));
#   reg_g = zeros(Float64, (nx, nx));
#   reg_f = EPLL_fg(reshape(x, (nx,nx)), reg_g, GDict);
#   g[:] = squeeze(g_v2 + g_t3amp + g_t3phi,1) + mu*vec(reg_g);
#   g[:] = (g - sum(x.*g) / flux ) / flux ; # gradient correction to take into account the non-normalized image
#   println("V2: ", chi2_v2/data.nv2, " T3A: ", chi2_t3amp/data.nt3amp, " T3P: ", chi2_t3phi/data.nt3phi," Flux: ", flux, " LAG: ", lag_f);
#   return (chi2_v2 + chi2_t3amp + chi2_t3phi) + mu*reg_f
# end
