using FITSIO;
using PyPlot;
PyPlot.show();
include("EPLL.jl");

# read in true image (for reference) and noisy image (= data from which we reconstruct)
true_image = rotl90(read((FITS("2004true137.fits"))[1])); #rotl90 to get same orientation as IDL
noiseSD = 50/255;
noisy_image = true_image + noiseSD*randn(size(true_image));
dict = importGMM("GMM_YSO.mat");
figure(1);imshow(true_image, ColorMap("hot"));PyPlot.draw();PyPlot.pause(0.05);
figure(2);imshow(noisy_image, ColorMap("hot"));PyPlot.draw();PyPlot.pause(0.05);
tic();
reconstructed_image = EPLL_denoise(noisy_image,noiseSD,dict);
toc();
println("PSNR: ",20*log10(1/std(reconstructed_image-true_image)), "\n");
println("EPLL: ", EPLL(reconstructed_image, dict), "\n");
println("l1dist: ", sum(abs(reconstructed_image-true_image))/length(true_image), "\n");
figure(3);imshow(reconstructed_image, ColorMap("hot"));PyPlot.draw();PyPlot.pause(0.05);
