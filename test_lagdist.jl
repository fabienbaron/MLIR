include("EPLL.jl")
include("oiplot.jl")
x = rand(64,64);
patchsize = 8;
px = im2col(x,(patchsize,patchsize));
z = rand(size(px));

f = sum((px-z).^2)

# numerical gradient
gnum=zeros(64,64);
tempx=copy(x);
delta = 1e-6;
for i=1:64
  println(i, "\n");
  for j=1:64
      tempx[i,j] = x[i,j] + delta;
      p_hi = im2col(tempx,(patchsize,patchsize));
      f_hi = sum((p_hi-z).^2)
      tempx[i,j] = x[i,j] - delta;
      p_lo = im2col(tempx,(patchsize,patchsize));
      f_lo = sum((p_lo-z).^2)
      tempx[i,j] = x[i,j];
      gnum[i,j]=(f_hi-f_lo)/(2.0*delta);
  end
end
imdisp(gnum)

ganly = col2im(2.*(px-z), (patchSize, patchSize), size(x), "sliding", "sum");
