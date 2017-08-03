using OIFITS
#read data file into Fabien Baron's custom structure
mutable struct OIdata
  nuv::Int64
  uv::Array{Float64,2}
  nw::Int64
  indx_v2::UnitRange{Int64}
  indx_t3_1::UnitRange{Int64}
  indx_t3_2::UnitRange{Int64}
  indx_t3_3::UnitRange{Int64}
  nv2::Int64
  nt3amp::Int64
  nt3phi::Int64
  v2_data::Array{Float64,1}
  v2_data_err::Array{Float64,1}
  t3amp_data::Array{Float64,1}
  t3amp_data_err::Array{Float64,1}
  t3phi_data::Array{Float64,1}
  t3phi_data_err::Array{Float64,1}
  baseline_v2::Array{Float64,1}
  baseline_t3::Array{Float64,1}
  mean_mjd::Float64
end

function read_oifits(oifitsfile, grey=true)
  tables=OIFITS.load(oifitsfile);
  wavtable = OIFITS.select(tables,"OI_WAVELENGTH");
  lam = wavtable[1][:eff_wave];
  dlam = wavtable[1][:eff_band];
  nw_true = length(lam);

  if (grey==true)
    nw_eff = 1;
  end

  v2table = OIFITS.select(tables,"OI_VIS2");
  v2_ntables = length(v2table)

  v2_data = v2table[1][:vis2data];
  v2_data_err = v2table[1][:vis2err];
  v2_ucoord = -v2table[1][:ucoord];
  v2_vcoord = v2table[1][:vcoord];

lambda_v2 = wavtable[][:eff_wave]


  mjd = v2table[1][:mjd];
  # merge all tables
  for itable=2:v2_ntables
    v2_data = hcat(v2_data, v2table[itable][:vis2data]);
    v2_data_err = hcat(v2_data_err, v2table[itable][:vis2err]);
    v2_ucoord = vcat(v2_ucoord, -v2table[itable][:ucoord]);
    v2_vcoord = vcat(v2_vcoord, v2table[itable][:vcoord]);
    mjd = vcat(mjd, v2table[itable][:mjd]);
  end

  mean_mjd = mean(mjd);
  #println("MJD for data: ", mean_mjd, " Min: ", minimum(mjd), " Max: ", maximum(mjd));

  if (nw_eff == 1) # note: be more clever and use reshape or THINK for polychromatic
    v2_data = vec(v2_data);
    v2_data_err = vec(v2_data_err);
    v2_ucoord = vec(v2_ucoord);
    v2_vcoord = vec(v2_vcoord);
  end



  nuvcoord = length(v2_ucoord);
  v2_u = Array{Float64}(nuvcoord*nw_true,nw_eff);
  v2_v = Array{Float64}(nuvcoord*nw_true,nw_eff);

  for u=1:nuvcoord
    v2_u[nw_true*(u-1)+1:nw_true*u,1] = v2_ucoord[u]*1./lam;
    v2_v[nw_true*(u-1)+1:nw_true*u,1] = v2_vcoord[u]*1./lam;
  end

  nv2 = length(v2_data);
  v2_uv = hcat(vec(v2_u), vec(v2_v))' ; #need under this form for Fourier transform
  baseline_v2 = vec(sqrt.(v2_u.^2+v2_v.^2));

  t3table = OIFITS.select(tables,"OI_T3");
  t3_ntables = length(t3table);

  t3amp_data = t3table[1][:t3amp];
  t3amp_data_err = t3table[1][:t3amperr];
  t3phi_data = t3table[1][:t3phi];
  t3phi_data_err = t3table[1][:t3phierr];
  t3_u1coord = -t3table[1][:u1coord];
  t3_v1coord = t3table[1][:v1coord];
  t3_u2coord = -t3table[1][:u2coord];
  t3_v2coord = t3table[1][:v2coord];

  # merge all tables
  for itable=2:t3_ntables
    t3amp_data = hcat(t3amp_data,t3table[itable][:t3amp]);
    t3amp_data_err = hcat(t3amp_data_err,t3table[itable][:t3amperr]);
    t3phi_data = hcat(t3phi_data,t3table[itable][:t3phi]);
    t3phi_data_err = hcat(t3phi_data_err,t3table[itable][:t3phierr]);
    t3_u1coord = vcat(t3_u1coord,-t3table[itable][:u1coord]);
    t3_v1coord = vcat(t3_v1coord,t3table[itable][:v1coord]);
    t3_u2coord = vcat(t3_u2coord,-t3table[itable][:u2coord]);
    t3_v2coord = vcat(t3_v2coord,t3table[itable][:v2coord]);
  end

  t3_u3coord = -(t3_u1coord+t3_u2coord); # the minus takes care of complex conjugate
  t3_v3coord = -(t3_v1coord+t3_v2coord);

  if (nw_eff == 1) #vectorize
    t3amp_data = vec(t3amp_data);
    t3amp_data_err = vec(t3amp_data_err);
    t3phi_data = vec(t3phi_data);
    t3phi_data_err = vec(t3phi_data_err);
    t3_u1coord = vec(t3_u1coord);
    t3_v1coord = vec(t3_v1coord);
    t3_u2coord = vec(t3_u2coord);
    t3_v2coord = vec(t3_v2coord);
  end

  nuvcoord_t3 = length(t3_u1coord);
  t3_u1 = Array{Float64}(nuvcoord_t3*nw_true,nw_eff);
  t3_v1 = Array{Float64}(nuvcoord_t3*nw_true,nw_eff);
  t3_u2 = Array{Float64}(nuvcoord_t3*nw_true,nw_eff);
  t3_v2 = Array{Float64}(nuvcoord_t3*nw_true,nw_eff);
  t3_u3 = Array{Float64}(nuvcoord_t3*nw_true,nw_eff);
  t3_v3 = Array{Float64}(nuvcoord_t3*nw_true,nw_eff);
  for u=1:nuvcoord_t3
    t3_u1[nw_true*(u-1)+1:nw_true*u,1] = t3_u1coord[u]*1./lam;
    t3_v1[nw_true*(u-1)+1:nw_true*u,1] = t3_v1coord[u]*1./lam;
    t3_u2[nw_true*(u-1)+1:nw_true*u,1] = t3_u2coord[u]*1./lam;
    t3_v2[nw_true*(u-1)+1:nw_true*u,1] = t3_v2coord[u]*1./lam;
    t3_u3[nw_true*(u-1)+1:nw_true*u,1] = t3_u3coord[u]*1./lam;
    t3_v3[nw_true*(u-1)+1:nw_true*u,1] = t3_v3coord[u]*1./lam;
  end
  nt3amp = length(t3amp_data);
  nt3phi = length(t3phi_data);
  baseline_t3 = vec((sqrt.(t3_u1.^2+t3_v1.^2).*sqrt.(t3_u2.^2+t3_v2.^2).*sqrt.(t3_u3.^2+t3_v3.^2)).^(1./3.));
  t3_uv = hcat(vcat(t3_u1, t3_u2, t3_u3), vcat(t3_v1, t3_v2, t3_v3))'; #need under this form for nfft

  # Merge observable uv points -- TBD: test for uniqueness to improve performance
  full_uv = hcat(v2_uv,t3_uv);
  nuv = size(full_uv,2);
  indx_v2 = 1:nv2;
  indx_t3_1 = nv2+(1:nt3amp);
  indx_t3_2 = nv2+(nt3amp+1:2*nt3amp);
  indx_t3_3 = nv2+(2*nt3amp+1:3*nt3amp);

  data = OIdata(nuv, full_uv, nw_eff,indx_v2,indx_t3_1,indx_t3_2,indx_t3_3,nv2,nt3amp,nt3phi,v2_data,v2_data_err,t3amp_data,t3amp_data_err,t3phi_data,t3phi_data_err, baseline_v2,baseline_t3, mean_mjd);
end

function readoifits_multiepochs(oifitsfiles)
nepochs = length(oifitsfiles);
tepochs = Array{Float64}(nepochs);
data = Array{OIdata}(nepochs);
for i=1:nepochs
  data[i] = read_oifits(oifitsfiles[i]);
  tepochs[i] = data[i].mean_mjd;
  println(oifitsfiles[i], "\t MJD: ", tepochs[i], "\t nV2 = ", data[i].nv2, "\t nT3amp = ", data[i].nt3amp, "\t nT3phi = ", data[i].nt3phi);
end
return nepochs, tepochs, data
end
