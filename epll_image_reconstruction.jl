
# gradient based minimization of EPLL
  using FITSIO
  using OptimPack
  using JLD
  include("readoifits.jl")
  include("oichi2.jl")
  include("oiplot.jl")
  include("initial.jl")
  include("EPLL.jl");
  PyPlot.show()

#  fitsfile = "2004true137.fits";
#  x_true = (read((FITS(fitsfile))[1])); nx = (size(x_true))[1]; x_true=vec(x_true);
  oifitsfile = "2004-data1.oifits";

  # Desired reconstruction parameters
  pixellation = 0.101; # in mas
  nx = 128;

  # Starting image
  x_start = initial_image(nx/6);

  data = readoifits(oifitsfile)[1,1];
  dft = setup_dft(data, nx, pixellation);
  Gdict = load("GMM_YSO.jld","GMM");
  crit = (x,g)->chi2_epll_fg(x, g, dft, data, Gdict);
  x = OptimPack.vmlmb(crit, x_start, verb=true, lower=0, upper=1, maxiter=50, blmvm=false);
