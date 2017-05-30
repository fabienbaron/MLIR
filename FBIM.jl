module FBIM
export fbim
using GR
function fbim(image)
sz=size(image);
setviewport(0.1, 0.9, 0.1, 0.9)
setwindow(-sz[1], sz[1], -sz[1], sz[1])
setcharheight(0.016)
settextalign(2, 0)
settextfontprec(126, 1)
x=collect(linspace(-sz[1],sz[1], sz[1]));
y=collect(linspace(-sz[1],sz[1], sz[1]));
setspace(minimum(image), maximum(image), 0, 0)
setcolormap(132)
surface(x,y,image,5)
axes(20, 20, -sz[1], -sz[1], 5, 5, -0.01)
axes(20, 20, sz[1], sz[1], 5, 5, 0.01)
end
using FBIM
end
