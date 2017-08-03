  using FITSIO
  using OptimPack
  include("readoifits.jl")
  include("oichi2.jl")
  include("oiplot.jl")
  include("initial.jl")

  PyPlot.show()

#  fitsfile = "2004true137.fits";
#  x_true = (read((FITS(fitsfile))[1])); nx = (size(x_true))[1]; x_true=vec(x_true);
  oifitsfile = "2004-data1.oifits";

  # Desired reconstruction parameters
  pixellation = 0.101; # in mas
  nx = 128;

  # Starting image
  x_start = initial_image(nx/6);

  data = read_oifits(oifitsfile);
  dft = setup_ft(data, nx, pixellation);
  crit = (x,g)->chi2_fg(x, g , dft, data);
  x = OptimPack.vmlmb(crit, x_start, verb=true, lower=0, upper=1, maxiter=50, blmvm=false);
