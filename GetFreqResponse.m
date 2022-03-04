function [X, f_axis] = GetFreqResponse(x,fs)

N = length(x);
X = fftshift(fft(x));

f_axis = -fs/2:fs/N:fs/2-1/N;