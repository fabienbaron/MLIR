using OIFITS
include("remove_redundance.jl")
mutable struct OIdata
  v2::Array{Float64,1}
  v2_err::Array{Float64,1}
  v2_baseline::Array{Float64,1}
  v2_mjd::Array{Float64,1}
  mean_mjd::Float64
  v2_lam::Array{Float64,1}
  v2_dlam::Array{Float64,1}
  v2_flag::Array{Bool,1}
  t3amp::Array{Float64,1}
  t3amp_err::Array{Float64,1}
  t3phi::Array{Float64,1}
  t3phi_err::Array{Float64,1}
  t3_baseline::Array{Float64,1}
  t3_maxbaseline::Array{Float64,1}
  t3_mjd::Array{Float64,1}
  t3_lam::Array{Float64,1}
  t3_dlam::Array{Float64,1}
  t3_flag::Array{Bool,1}
  uv::Array{Float64,2}
  nv2::Int64
  nt3amp::Int64
  nt3phi::Int64
  nuv::Int64
  indx_v2::UnitRange{Int64}
  indx_t3_1::UnitRange{Int64}
  indx_t3_2::UnitRange{Int64}
  indx_t3_3::UnitRange{Int64}
end

"""
Last updated on September 6, 2017
Made by Arturo O. Martinez
Improved from Fabien Baron's original readoifits.jl

How to use the readoifits_timespec function:
- Insert the name of your oifits file to readoifits_timespec
- If user wants to have a spectral bin, one could put as
spectralbin = [[1.e-6,1.5e-6]] (e.g., in microns) where this array would only
include wavelengths within this range. If spectralbin = [[1.e-6,1.5e-6],[1.5e-6,1.6e-6,1.6e-6,1.7e-6]]
then the first array would be an array of wavelengths from a given interval and
the second array would be a set of ranges in which the user wants their data
individually from the first set. This code can also "exclude" ranges into a
seperate array [[1.e-6,1.5e-6,1.6e-6,2.e-6],[1.5e-6,1.6e-6]] where the first
array is all the data from 1.e-6 to 1.5e-6 microns AND 1.6e-6 to 2.e-6 microns.
That second array would only include data from 1.5e-6 to 1.6e-6 (e.g., used for
when there would be a feature to be studied, like an emission line).
- temporalbin works the same way as spectralbin.

How to use the time_split function:
Once readoifits_timespec is given a first run through, then if the user wanted
to split up data with a given period (in days), the user inputs the full mjd
and the period desired. The output would be a temporalbin that could be used on
readoifits_timespec second run though. This is in the range of [start, end).
"""

function readoifits(oifitsfile; spectralbin=[[]], temporalbin=[[]],
  get_specbin_file=true, get_timebin_file=true,redundance_chk=false,uvtol=1.e3)

  # oifitsfile="AlphaCenA.oifits";spectralbin=[[]]; temporalbin=[[]];  get_specbin_file=false; get_timebin_file=true;redundance_chk=true;uvtol=1.e3;

  tables = OIFITS.load(oifitsfile);
  wavtable = OIFITS.select(tables,"OI_WAVELENGTH");
  wavtableref = [wavtable[i][:insname] for i=1:length(wavtable)];
  v2table = OIFITS.select(tables,"OI_VIS2");
  v2_ntables = length(v2table);

  t3table = OIFITS.select(tables,"OI_T3");
  t3_ntables = length(t3table);

  # get V2 data from tables
  v2_old = Array{Array{Float64,2}}(v2_ntables);
  v2_err_old = Array{Array{Float64,2}}(v2_ntables);
  v2_ucoord_old = Array{Array{Float64,1}}(v2_ntables);
  v2_vcoord_old = Array{Array{Float64,1}}(v2_ntables);
  v2_mjd_old = Array{Array{Float64,2}}(v2_ntables);
  v2_lam_old = Array{Array{Float64,2}}(v2_ntables);
  v2_dlam_old = Array{Array{Float64,2}}(v2_ntables);
  v2_flag_old = Array{Array{Bool,2}}(v2_ntables);
  v2_u_old = Array{Array{Float64,1}}(v2_ntables);
  v2_v_old = Array{Array{Float64,1}}(v2_ntables);
  v2_uv_old = Array{Array{Float64,2}}(v2_ntables);
  v2_baseline_old = Array{Array{Float64,1}}(v2_ntables);

  for itable = 1:v2_ntables
      v2_old[itable] = v2table[itable][:vis2data]; # Visibility squared
      v2_err_old[itable] = v2table[itable][:vis2err]; # error in Visibility squared
      v2_ucoord_old[itable] = -v2table[itable][:ucoord]; # u coordinate in uv plane
      v2_vcoord_old[itable] = v2table[itable][:vcoord]; #  v coordinate in uv plane
      v2_mjd_old[itable] = repeat(v2table[itable][:mjd]',
        outer=[size(v2_old[itable],1),1]); # Modified Julian Date (JD - 2400000.5)
      whichwav = find(v2table[itable][:insname].==wavtableref);
      if (length(whichwav) != 1)
        println("Wave table confusion\n");
      end
      v2_lam_old[itable] = repeat(wavtable[whichwav[1]][:eff_wave],
        outer=[1,size(v2_old[itable],2)]); # spectral channels
      v2_dlam_old[itable] = repeat(wavtable[whichwav[1]][:eff_band],
        outer=[1,size(v2_old[itable],2)]); # width of spectral channels
      v2_flag_old[itable] = v2table[itable][:flag]; # flag for v2 table
      nv2_lam_old = length(v2_lam_old[itable][:,1]);

      v2_u_old[itable] = v2_ucoord_old[itable][1]./v2_lam_old[itable][:,1];
      v2_v_old[itable] = v2_vcoord_old[itable][1]./v2_lam_old[itable][:,1];
      for u = 2:length(v2_ucoord_old[itable])
        v2_u_old[itable] = vcat(v2_u_old[itable],
          v2_ucoord_old[itable][u]./v2_lam_old[itable][:,1]);
        v2_v_old[itable] = vcat(v2_v_old[itable],
          v2_vcoord_old[itable][u]./v2_lam_old[itable][:,1]);
      end

      v2_uv_old[itable] = hcat(vec(v2_u_old[itable]),vec(v2_v_old[itable]));
      v2_baseline_old[itable] = vec(sqrt.(v2_u_old[itable].^2 +
        v2_v_old[itable].^2));
  end

  # same with T3, VIS
  # Get T3 data from tables
  t3amp_old = Array{Array{Float64,2}}(t3_ntables);
  t3amp_err_old = Array{Array{Float64,2}}(t3_ntables);
  t3phi_old = Array{Array{Float64,2}}(t3_ntables);
  t3phi_err_old = Array{Array{Float64,2}}(t3_ntables);
  t3_u1coord_old = Array{Array{Float64,1}}(t3_ntables);
  t3_v1coord_old = Array{Array{Float64,1}}(t3_ntables);
  t3_u2coord_old = Array{Array{Float64,1}}(t3_ntables);
  t3_v2coord_old = Array{Array{Float64,1}}(t3_ntables);
  t3_u3coord_old = Array{Array{Float64,1}}(t3_ntables);
  t3_v3coord_old = Array{Array{Float64,1}}(t3_ntables);
  t3_mjd_old = Array{Array{Float64,2}}(t3_ntables);
  t3_lam_old = Array{Array{Float64,2}}(t3_ntables);
  t3_dlam_old = Array{Array{Float64,2}}(t3_ntables);
  t3_flag_old = Array{Array{Bool,2}}(t3_ntables);
  t3_u1_old = Array{Array{Float64,1}}(t3_ntables);
  t3_v1_old = Array{Array{Float64,1}}(t3_ntables);
  t3_u2_old = Array{Array{Float64,1}}(t3_ntables);
  t3_v2_old = Array{Array{Float64,1}}(t3_ntables);
  t3_u3_old = Array{Array{Float64,1}}(t3_ntables);
  t3_v3_old = Array{Array{Float64,1}}(t3_ntables);
  #t3_uv_old = Array{Array{Float64,2}}(t3_ntables);
  t3_baseline_old = Array{Array{Float64,1}}(t3_ntables);
  t3_maxbaseline_old = Array{Array{Float64,1}}(t3_ntables);

  for itable = 1:t3_ntables
    t3amp_old[itable] = t3table[itable][:t3amp];
    t3amp_err_old[itable] = t3table[itable][:t3amperr];
    t3phi_old[itable] = t3table[itable][:t3phi];
    t3phi_err_old[itable] = t3table[itable][:t3phierr];
    t3_u1coord_old[itable] = -t3table[itable][:u1coord];
    t3_v1coord_old[itable] = t3table[itable][:v1coord];
    t3_u2coord_old[itable] = -t3table[itable][:u2coord];
    t3_v2coord_old[itable] = t3table[itable][:v2coord];
    t3_u3coord_old[itable] = -(t3_u1coord_old[itable] + t3_u2coord_old[itable]); # the minus takes care of complex conjugate
    t3_v3coord_old[itable] = -(t3_v1coord_old[itable] + t3_v2coord_old[itable]);
    t3_mjd_old[itable] = repeat(t3table[itable][:mjd]',
      outer=[size(t3amp_old[itable],1),1]); # Modified Julian Date (JD - 2400000.5)
    whichwav = find(t3table[itable][:insname].==wavtableref);
    t3_lam_old[itable] = repeat(wavtable[whichwav[1]][:eff_wave],
      outer=[1,size(t3amp_old[itable],2)]); # spectral channels
    t3_dlam_old[itable] = repeat(wavtable[whichwav[1]][:eff_band],
      outer=[1,size(t3amp_old[itable],2)]); # width of spectral channels
    t3_flag_old[itable] = t3table[itable][:flag]; # flag for t3 table
    nt3_lam_old = length(t3_lam_old[itable][:,1]);

    t3_u1_old[itable] = t3_u1coord_old[itable][1]./t3_lam_old[itable][:,1];
    t3_v1_old[itable] = t3_v1coord_old[itable][1]./t3_lam_old[itable][:,1];
    t3_u2_old[itable] = t3_u2coord_old[itable][1]./t3_lam_old[itable][:,1];
    t3_v2_old[itable] = t3_v2coord_old[itable][1]./t3_lam_old[itable][:,1];
    t3_u3_old[itable] = t3_u3coord_old[itable][1]./t3_lam_old[itable][:,1];
    t3_v3_old[itable] = t3_v3coord_old[itable][1]./t3_lam_old[itable][:,1];
    for u = 2:length(t3_u1coord_old[itable])
      t3_u1_old[itable] = vcat(t3_u1_old[itable],t3_u1coord_old[itable][u]./t3_lam_old[itable][:,1]);
      t3_v1_old[itable] = vcat(t3_v1_old[itable],t3_v1coord_old[itable][u]./t3_lam_old[itable][:,1]);
      t3_u2_old[itable] = vcat(t3_u2_old[itable],t3_u2coord_old[itable][u]./t3_lam_old[itable][:,1]);
      t3_v2_old[itable] = vcat(t3_v2_old[itable],t3_v2coord_old[itable][u]./t3_lam_old[itable][:,1]);
      t3_u3_old[itable] = vcat(t3_u3_old[itable],t3_u3coord_old[itable][u]./t3_lam_old[itable][:,1]);
      t3_v3_old[itable] = vcat(t3_v3_old[itable],t3_v3coord_old[itable][u]./t3_lam_old[itable][:,1]);
    end

    #t3_uv_old[itable] = hcat(vcat(t3_u1_old[itable],t3_u2_old[itable],t3_u3_old[itable]),
    #    vcat(t3_v1_old[itable],t3_v2_old[itable],t3_v3_old[itable]));
    t3_baseline_old[itable] = vec((sqrt.(t3_u1_old[itable].^2 + t3_v1_old[itable].^2).*
        sqrt.(t3_u2_old[itable].^2 + t3_v2_old[itable].^2).*
        sqrt.(t3_u3_old[itable].^2 + t3_v3_old[itable].^2)).^(1./3.));
    t3_maxbaseline_old[itable] = vec(max.(
    sqrt.(t3_u1_old[itable].^2 + t3_v1_old[itable].^2),
    sqrt.(t3_u2_old[itable].^2 + t3_v2_old[itable].^2),
    sqrt.(t3_u3_old[itable].^2 + t3_v3_old[itable].^2)));

  end

  # combine all data into one
  v2_all = Float64[];
  v2_err_all = Float64[];
  #v2_ucoord_new = fill((Float64[]),nspecbin,ntimebin);
  #v2_vcoord_new = fill((Float64[]),nspecbin,ntimebin);
  v2_mjd_all = Float64[];
  v2_lam_all = Float64[];
  v2_dlam_all = Float64[];
  v2_flag_all = Bool[];
  #v2_u_new = fill((Float64[]),nspecbin,ntimebin);
  #v2_v_new = fill((Float64[]),nspecbin,ntimebin);
  v2_uv_all = vcat(Float64[]',Float64[]');
  v2_baseline_all = Float64[];
  t3amp_all = Float64[];
  t3amp_err_all = Float64[];
  t3phi_all = Float64[];
  t3phi_err_all = Float64[];
  #t3_u1coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_v1coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_u2coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_v2coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_u3coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_v3coord_new = fill((Float64[]),nspecbin,ntimebin);
  t3_mjd_all = Float64[];
  t3_lam_all = Float64[];
  t3_dlam_all = Float64[];
  t3_flag_all = Float64[];
  t3_u1_all = Float64[];
  t3_v1_all = Float64[];
  t3_u2_all = Float64[];
  t3_v2_all = Float64[];
  t3_u3_all = Float64[];
  t3_v3_all = Float64[];
  t3_uv_all = vcat(Float64[]',Float64[]');
  t3_baseline_all = Float64[];
  t3_maxbaseline_all = Float64[];

  for i = 1:v2_ntables
    v2_all = vcat(v2_all,vec(v2_old[i]));
    v2_err_all = vcat(v2_err_all,vec(v2_err_old[i]));
    v2_mjd_all = vcat(v2_mjd_all,vec(v2_mjd_old[i]));
    v2_lam_all = vcat(v2_lam_all,vec(v2_lam_old[i]));
    v2_dlam_all = vcat(v2_dlam_all,vec(v2_dlam_old[i]));
    v2_flag_all = vcat(v2_flag_all,vec(v2_flag_old[i]));
    v2_uv_all = hcat(v2_uv_all,v2_uv_old[i]'); # Must have in this form for Fourier transform
    v2_baseline_all = vcat(v2_baseline_all,vec(v2_baseline_old[i]));
  end

  for i = 1:t3_ntables
    t3amp_all = vcat(t3amp_all,vec(t3amp_old[i]));
    t3amp_err_all = vcat(t3amp_err_all,vec(t3amp_err_old[i]));
    t3phi_all = vcat(t3phi_all,vec(t3phi_old[i]));
    t3phi_err_all = vcat(t3phi_err_all,vec(t3phi_err_old[i]));
    t3_mjd_all = vcat(t3_mjd_all,vec(t3_mjd_old[i]));
    t3_lam_all = vcat(t3_lam_all,vec(t3_lam_old[i]));
    t3_dlam_all = vcat(t3_dlam_all,vec(t3_dlam_old[i]));
    t3_flag_all = vcat(t3_flag_all,vec(t3_flag_old[i]));
    t3_u1_all = vcat(t3_u1_all,vec(t3_u1_old[i]));
    t3_v1_all = vcat(t3_v1_all,vec(t3_v1_old[i]));
    t3_u2_all = vcat(t3_u2_all,vec(t3_u2_old[i]));
    t3_v2_all = vcat(t3_v2_all,vec(t3_v2_old[i]));
    t3_u3_all = vcat(t3_u3_all,vec(t3_u3_old[i]));
    t3_v3_all = vcat(t3_v3_all,vec(t3_v3_old[i]));
    #t3_uv_all = hcat(t3_uv_all,t3_uv_old[i]'); # Must have in this form for Fourier transform
    t3_baseline_all = vcat(t3_baseline_all,vec(t3_baseline_old[i]));
    t3_maxbaseline_all = vcat(t3_maxbaseline_all,vec(t3_maxbaseline_old[i]));
  end
  t3_uv_all = hcat(vcat(t3_u1_all,t3_u2_all,t3_u3_all),vcat(t3_v1_all,t3_v2_all,t3_v3_all))';

  # calculate default timebin if user picks timebin = [[]]
  if ((temporalbin == [[]]) && (get_timebin_file == true))
    temporalbin = [[]]
    temporalbin[1] = [minimum(v2_mjd_all),maximum(v2_mjd_all)]; # start & end mjd
    temporalbin[1][2] += 0.001
  end

  # get spectralbin if get_spectralbin_from_file == true
  if ((spectralbin == [[]]) && (get_specbin_file == true))
    spectralbin[1] = vcat(spectralbin[1],
      minimum(v2_lam_all)-minimum(v2_dlam_all[indmin(v2_lam_all)])*0.5,
      maximum(v2_lam_all)+maximum(v2_dlam_all[indmax(v2_lam_all)])*0.5);
  end

  # count how many spectral bins user input into file
  nspecbin_old = length(spectralbin);
  ncombspec = Int(length(spectralbin[1])/2);
  if (nspecbin_old == 1) # exclude other data
    nspecbin = ntotspec = 1;
  elseif (nspecbin_old == 2)
    nsplitspec = Int(length(spectralbin[2])/2);
    ntotspec = ncombspec + nsplitspec;
    nspecbin = Int(1) + nsplitspec;
  end

  # count how many temporal bins user input into file
  ntimebin_old = length(temporalbin);
  ncombtime = Int(length(temporalbin[1])/2);
  if (ntimebin_old == 1)
    ntimebin = ntottime = 1;
  elseif (ntimebin_old == 2)
    nsplittime = Int(length(temporalbin[2])/2);
    ntottime = ncombtime + nsplittime;
    ntimebin = Int(1) + nsplittime;
  end

  OIdataArr = Array{OIdata}(nspecbin,ntimebin);

  # Define new arrays so that they get binned properly
  v2_new = fill((Float64[]),nspecbin,ntimebin);
  v2_err_new = fill((Float64[]),nspecbin,ntimebin);
  #v2_ucoord_new = fill((Float64[]),nspecbin,ntimebin);
  #v2_vcoord_new = fill((Float64[]),nspecbin,ntimebin);
  v2_mjd_new = fill((Float64[]),nspecbin,ntimebin);
  v2_lam_new = fill((Float64[]),nspecbin,ntimebin);
  v2_dlam_new = fill((Float64[]),nspecbin,ntimebin);
  v2_flag_new = fill((Bool[]),nspecbin,ntimebin);
  #v2_u_new = fill((Float64[]),nspecbin,ntimebin);
  #v2_v_new = fill((Float64[]),nspecbin,ntimebin);
  v2_uv_new = fill((vcat(Float64[]',Float64[]')),nspecbin,ntimebin);
  v2_baseline_new = fill((Float64[]),nspecbin,ntimebin);
  t3amp_new = fill((Float64[]),nspecbin,ntimebin);
  t3amp_err_new = fill((Float64[]),nspecbin,ntimebin);
  t3phi_new = fill((Float64[]),nspecbin,ntimebin);
  t3phi_err_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_u1coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_v1coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_u2coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_v2coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_u3coord_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_v3coord_new = fill((Float64[]),nspecbin,ntimebin);
  t3_mjd_new = fill((Float64[]),nspecbin,ntimebin);
  t3_lam_new = fill((Float64[]),nspecbin,ntimebin);
  t3_dlam_new = fill((Float64[]),nspecbin,ntimebin);
  t3_flag_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_u1_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_v1_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_u2_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_v2_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_u3_new = fill((Float64[]),nspecbin,ntimebin);
  #t3_v3_new = fill((Float64[]),nspecbin,ntimebin);
  t3_uv_new = fill((vcat(Float64[]',Float64[]')),nspecbin,ntimebin);
  t3_baseline_new = fill((Float64[]),nspecbin,ntimebin);
  t3_maxbaseline_new = fill((Float64[]),nspecbin,ntimebin);
  mean_mjd = Array{Float64}(nspecbin,ntimebin);
  full_uv = Array{Array{Float64,2}}(nspecbin,ntimebin);
  nv2 = Array{Int64}(nspecbin,ntimebin);
  nt3amp = Array{Int64}(nspecbin,ntimebin);
  nt3phi = Array{Int64}(nspecbin,ntimebin);
  nuv = Array{Int64}(nspecbin,ntimebin);
  indx_v2 = Array{UnitRange{Int64}}(nspecbin,ntimebin);
  indx_t3_1 = Array{UnitRange{Int64}}(nspecbin,ntimebin);
  indx_t3_2 = Array{UnitRange{Int64}}(nspecbin,ntimebin);
  indx_t3_3 = Array{UnitRange{Int64}}(nspecbin,ntimebin);

  iter_mjd = 0; iter_wav = 0;
  t3_uv_mjd = zeros(length(t3_mjd_all)*3);
  t3_uv_lam = zeros(length(t3_lam_all)*3);
  for i=1:length(t3_mjd_all)
    t3_uv_mjd[i*3-2:i*3] = t3_mjd_all[i];
    t3_uv_lam[i*3-2:i*3] = t3_lam_all[i];
  end

  # New iteration for binning data
  for itime = 1:ntottime
    # combine data to one bin
    iter_mjd += 1;
    if (itime <= ncombtime)
      itimebin = 1;
    else
      if (ncombtime == 0)
        itimebin = itime;
      else
        itimebin = itime - ncombtime + 1;
        if (itimebin == 2)
          iter_mjd = iter_mjd - ncombtime;
        end
      end
    end

    # get ranges for time binning
    if (itime == 1) # make sure logic is right
      lo_time = temporalbin[1][[(i%2 == 1) for i=1:length(temporalbin[1])]];
      hi_time = temporalbin[1][[(i%2 == 0) for i=1:length(temporalbin[1])]];
    else
      lo_time = temporalbin[2][[(i%2 == 1) for i=1:length(temporalbin[2])]];
      hi_time = temporalbin[2][[(i%2 == 0) for i=1:length(temporalbin[2])]];
    end

    for ispec = 1:ntotspec
      # combine data to one bin
      iter_wav += 1;
      if (ispec <= ncombspec)
        ispecbin = 1;
      else
        if (ncombspec == 0)
          ispecbin = ispec;
        else
          ispecbin = ispec - ncombspec + 1;
          if (ispecbin == 2)
            iter_wav = iter_wav - ncombspec;
          end
        end
      end

      # get ranges for wavelength binning
      if (ispec == 1) # make sure logic is right
        lo_wav = spectralbin[1][[(i%2 == 1) for i=1:length(spectralbin[1])]];
        hi_wav = spectralbin[1][[(i%2 == 0) for i=1:length(spectralbin[1])]];
      else
        lo_wav = spectralbin[2][[(i%2 == 1) for i=1:length(spectralbin[2])]];
        hi_wav = spectralbin[2][[(i%2 == 0) for i=1:length(spectralbin[2])]];
      end

      # filter out unwanted data
      filter_v2 = (v2_mjd_all.<hi_time[iter_mjd]).&(v2_mjd_all.>=lo_time[iter_mjd]).&(v2_lam_all.<hi_wav[iter_wav]).&(v2_lam_all.>lo_wav[iter_wav]);
      filter_t3 = (t3_mjd_all.<hi_time[iter_mjd]).&(t3_mjd_all.>=lo_time[iter_mjd]).&(t3_lam_all.<hi_wav[iter_wav]).&(t3_lam_all.>lo_wav[iter_wav]);
      filter_t3uv = (t3_uv_mjd.<hi_time[iter_mjd]).&(t3_uv_mjd.>=lo_time[iter_mjd]).&(t3_uv_lam.<hi_wav[iter_wav]).&(t3_uv_lam.>lo_wav[iter_wav]);

      v2_new[ispecbin,itimebin] = vcat(v2_new[ispecbin,itimebin],v2_all[filter_v2]);
      v2_err_new[ispecbin,itimebin] = vcat(v2_err_new[ispecbin,itimebin],v2_err_all[filter_v2]);
      v2_mjd_new[ispecbin,itimebin] = vcat(v2_mjd_new[ispecbin,itimebin],v2_mjd_all[filter_v2]);
      v2_lam_new[ispecbin,itimebin] = vcat(v2_lam_new[ispecbin,itimebin],v2_lam_all[filter_v2]);
      v2_dlam_new[ispecbin,itimebin] = vcat(v2_dlam_new[ispecbin,itimebin],v2_dlam_all[filter_v2]);
      v2_flag_new[ispecbin,itimebin] = vcat(v2_flag_new[ispecbin,itimebin],v2_flag_all[filter_v2]);
      v2_uv_new[ispecbin,itimebin] = hcat(v2_uv_new[ispecbin,itimebin],vcat(v2_uv_all[1,:][filter_v2]',v2_uv_all[2,:][filter_v2]'));
      v2_baseline_new[ispecbin,itimebin] = vcat(v2_baseline_new[ispecbin,itimebin],v2_baseline_all[filter_v2]);

      t3amp_new[ispecbin,itimebin] = vcat(t3amp_new[ispecbin,itimebin],t3amp_all[filter_t3]);
      t3amp_err_new[ispecbin,itimebin] = vcat(t3amp_err_new[ispecbin,itimebin],t3amp_err_all[filter_t3]);
      t3phi_new[ispecbin,itimebin] = vcat(t3phi_new[ispecbin,itimebin],t3phi_all[filter_t3]);
      t3phi_err_new[ispecbin,itimebin] = vcat(t3phi_err_new[ispecbin,itimebin],t3phi_err_all[filter_t3]);
      t3_mjd_new[ispecbin,itimebin] = vcat(t3_mjd_new[ispecbin,itimebin],t3_mjd_all[filter_t3]);
      t3_lam_new[ispecbin,itimebin] = vcat(t3_lam_new[ispecbin,itimebin],t3_lam_all[filter_t3]);
      t3_dlam_new[ispecbin,itimebin] = vcat(t3_dlam_new[ispecbin,itimebin],t3_dlam_all[filter_t3]);
      t3_flag_new[ispecbin,itimebin] = vcat(t3_flag_new[ispecbin,itimebin],t3_flag_all[filter_t3]);
      t3_uv_new[ispecbin,itimebin] = hcat(t3_uv_new[ispecbin,itimebin],vcat(t3_uv_all[1,:][filter_t3uv]',t3_uv_all[2,:][filter_t3uv]'));
      t3_baseline_new[ispecbin,itimebin] = vcat(t3_baseline_new[ispecbin,itimebin],t3_baseline_all[filter_t3]);
      t3_maxbaseline_new[ispecbin,itimebin] = vcat(t3_maxbaseline_new[ispecbin,itimebin],t3_maxbaseline_all[filter_t3]);


      mean_mjd[ispecbin,itimebin] = mean(v2_mjd_new[ispecbin,itimebin]);
      full_uv[ispecbin,itimebin] = hcat(v2_uv_new[ispecbin,itimebin],t3_uv_new[ispecbin,itimebin]);
      nv2[ispecbin,itimebin] = length(v2_new[ispecbin,itimebin]);
      nt3amp[ispecbin,itimebin] = length(t3amp_new[ispecbin,itimebin]);
      nt3phi[ispecbin,itimebin] = length(t3phi_new[ispecbin,itimebin]);
      nuv[ispecbin,itimebin] = size(full_uv[ispecbin,itimebin],2);
      indx_v2[ispecbin,itimebin] = 1:nv2[ispecbin,itimebin];
      indx_t3_1[ispecbin,itimebin] = nv2[ispecbin,itimebin]+(1:nt3amp[ispecbin,itimebin]);
      indx_t3_2[ispecbin,itimebin] = nv2[ispecbin,itimebin]+(nt3amp[ispecbin,itimebin]+1:2*nt3amp[ispecbin,itimebin]);
      indx_t3_3[ispecbin,itimebin] = nv2[ispecbin,itimebin]+(2*nt3amp[ispecbin,itimebin]+1:3*nt3amp[ispecbin,itimebin]);

      if (redundance_chk == true) # temp fix?
        full_uv[ispecbin,itimebin], indx_redun = rm_redundance_kdtree(full_uv[ispecbin,itimebin],uvtol);
        nuv[ispecbin,itimebin] = size(full_uv[ispecbin,itimebin],2);
        indx_v2[ispecbin,itimebin] = indx_redun[indx_v2[ispecbin,itimebin]];
        indx_t3_1[ispecbin,itimebin] = indx_redun[indx_t3_1[ispecbin,itimebin]];
        indx_t3_2[ispecbin,itimebin] = indx_redun[indx_t3_2[ispecbin,itimebin]];
        indx_t3_3[ispecbin,itimebin] = indx_redun[indx_t3_3[ispecbin,itimebin]];
      end

      OIdataArr[ispecbin,itimebin] = OIdata(v2_new[ispecbin,itimebin], v2_err_new[ispecbin,itimebin], v2_baseline_new[ispecbin,itimebin], v2_mjd_new[ispecbin,itimebin],
        mean_mjd[ispecbin,itimebin], v2_lam_new[ispecbin,itimebin], v2_dlam_new[ispecbin,itimebin], v2_flag_new[ispecbin,itimebin], t3amp_new[ispecbin,itimebin],
        t3amp_err_new[ispecbin,itimebin], t3phi_new[ispecbin,itimebin], t3phi_err_new[ispecbin,itimebin], t3_baseline_new[ispecbin,itimebin],t3_maxbaseline_new[ispecbin,itimebin],
        t3_mjd_new[ispecbin,itimebin], t3_lam_new[ispecbin,itimebin], t3_dlam_new[ispecbin,itimebin], t3_flag_new[ispecbin,itimebin], full_uv[ispecbin,itimebin],
        nv2[ispecbin,itimebin], nt3amp[ispecbin,itimebin], nt3phi[ispecbin,itimebin], nuv[ispecbin,itimebin], indx_v2[ispecbin,itimebin],
        indx_t3_1[ispecbin,itimebin], indx_t3_2[ispecbin,itimebin], indx_t3_3[ispecbin,itimebin]);
    end
    iter_wav = 0;
  end

  return OIdataArr;
end

function readoifits_multiepochs(oifitsfiles)
nepochs = length(oifitsfiles);
tepochs = Array{Float64}(nepochs);
data = Array{OIdata}(nepochs);
for i=1:nepochs
  data[i] = readoifits(oifitsfiles[i])[1,1];
  tepochs[i] = data[i].mean_mjd;
  println(oifitsfiles[i], "\t MJD: ", tepochs[i], "\t nV2 = ", data[i].nv2, "\t nT3amp = ", data[i].nt3amp, "\t nT3phi = ", data[i].nt3phi);
end
return nepochs, tepochs, data
end

# period in days
function time_split(mjd,period;mjd_start=mjd[1])
  timebins = (maximum(mjd) - mjd_start)/(period);
  itimebin = Int(ceil(timebins));
  temporalbin = [[],[]];
  temporalbin[1] = [mjd_start,mjd_start+period];
  temporalbin[2] = [mjd_start+period,mjd_start+2*period];
  for i = 3:itimebin
    temporalbin[2] = vcat(temporalbin[2],mjd_start+(i-1)*period);
    temporalbin[2] = vcat(temporalbin[2],mjd_start+(i)*period);
  end
  return temporalbin
end
