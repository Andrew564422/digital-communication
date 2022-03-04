function Noise = AWGN(input_signal,No)
 
Noise = normrnd(0,No,size(input_signal)) ; 
 
end
