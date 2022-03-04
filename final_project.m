clear all; clc
%%%%%%%%%part1
%% parameters:
fs = 1e7; 
Ts = 1/fs; 
N = 1e5;  
f_axis = -fs/2:fs/N:fs/2-1/N;  
t_axis = (0:N-1)*Ts;  
E_bit = 1;  
B = 100e3;
T_sq = 2/B;

%% guidelines1: band_limited_channel
one_square = ones(1,2000);
zero_me = zeros(1,98000/2);
Band_limited_channel= [zero_me one_square zero_me];

figure;
plot(f_axis,Band_limited_channel,'linewidth',2)
grid on
ylim([0 2])
xlim([-1/T_sq 1/T_sq]*5)
xlabel('Frequency (Hz)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('A Band-limited channel in frequency domains','linewidth',10)

%% guidelines2: 
%%%%%%%%%%%%%%%generate square pulse 
x_bits=[1];
figure;
pulse1 = GenerateSquarePulses(t_axis,T_sq,E_bit,fs,x_bits,'unipolar');
pulse1_fft = GetFreqResponse(pulse1,fs);
subplot(2,1,1); plot(t_axis,pulse1,'b','linewidth',2);
grid on
xlim([0 T_sq*4.2])
ylim([0 0.3])
xlabel('Time (s)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('square pulse in time domain','linewidth',10)

subplot(2,1,2); plot(f_axis,abs(pulse1_fft),'b','linewidth',2);
grid on
ylim([0 30])
xlim([-1/T_sq 1/T_sq]*5)
xlabel('Frequency (Hz)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('square pulse in frequency domain','linewidth',10)

%%%%%%%% output from channel 
pulse1_after_chann = pulse1_fft .* Band_limited_channel;
pulse1_after_chann_T = real(ifft(ifftshift(pulse1_after_chann)));

figure;
subplot(2,1,1); plot(t_axis,pulse1_after_chann_T,'b','linewidth',2);
xlim([0 T_sq*5])
xlabel('Time (s)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('output of generated pulse in time domain ','linewidth',10)

subplot(2,1,2); plot(f_axis,abs(pulse1_after_chann),'b','linewidth',2);
grid on
xlim([-1/T_sq 1/T_sq]*5)
ylim([0 50])
xlabel('Frequency (Hz)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('output of generated pulse in frequency domain ','linewidth',10)

%%%%%%%%%%%%%%%%%%%%% part (1) b)
x_bits = [1];
pulse1 = GenerateSquarePulses(t_axis,T_sq,E_bit,fs,x_bits,'unipolar'); 
x_bits = [0 1];
pulse2 = GenerateSquarePulses(t_axis,T_sq,E_bit,fs,x_bits,'unipolar');

pulse1_fft = GetFreqResponse(pulse1,fs);
pulse2_fft = GetFreqResponse(pulse2,fs);

figure;
plot(t_axis,pulse1,'b','linewidth',2); hold on;
plot(t_axis,pulse2,'r','linewidth',2); 
grid on
xlim([0 T_sq*4.2])
ylim([0 0.3])
xlabel('Time (s)','linewidth',2)
ylabel('Amplitude','linewidth',2)
legend({'Pulse 1','Pulse 2'},'fontsize',10)
title('Square pulses in time domain ','linewidth',10)


pulse1_after_chann = pulse1_fft .* Band_limited_channel;
pulse1_after_chann_T = real(ifft(ifftshift(pulse1_after_chann)));
pulse2_after_chann = pulse2_fft .* Band_limited_channel;
pulse2_after_chann_T = real(ifft(ifftshift(pulse2_after_chann)));

figure; plot(t_axis,pulse1_after_chann_T,'b','linewidth',2); 
grid on;
xlim([0 T_sq*5])
xlabel('Time (s)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('output of Square pulses in time domain','linewidth',10)
figure; plot(t_axis,pulse2_after_chann_T,'r','linewidth',2); 
grid on
xlim([0 T_sq*5])
xlabel('Time (s)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('output of Square pulses in time domain','linewidth',10)

%% another pulse shape
s1= 0.4*sinc(t_axis*B- pi/3);
s2= 0.4*sinc(t_axis*B- pi);
S1 = GetFreqResponse(s1,fs);
S2 = GetFreqResponse(s2,fs);
figure;
%subplot(2,1,1); 
plot(t_axis,s1,'b','linewidth',2); hold on;
plot(t_axis,s2,'r','linewidth',2); 

grid on; 
xlim([0 T_sq*4.2])
ylim([-0.1 0.4])
xlabel('Time (s)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('sinc pulses in time domain','linewidth',10)

%subplot(2,1,2); plot(f_axis,abs(S1),'b','linewidth',2);
%grid on
%ylim([0 30])
%xlim([-1/T_sq 1/T_sq]*5)
%xlabel('Frequency (Hz)','linewidth',2)
%ylabel('Amplitude','linewidth',2)
%title('square pulse in frequency domain','linewidth',10)


pulse11_after_chann = S1 .* Band_limited_channel;
pulse22_after_chann = S2 .* Band_limited_channel;
pulse11_after_chann_T = real(ifft(ifftshift(pulse11_after_chann)));
pulse22_after_chann_T = real(ifft(ifftshift(pulse22_after_chann)));

figure;
subplot(2,1,1); 
plot(t_axis,pulse11_after_chann_T,'b','linewidth',2);
grid on;
xlim([0 T_sq*5])
xlabel('Time (s)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('output of generated sinc pulse in time domain ','linewidth',10)

subplot(2,1,2); plot(f_axis,abs(pulse11_after_chann),'b','linewidth',2);
grid on
xlim([-1/T_sq 1/T_sq]*5)
ylim([0 50])
xlabel('Frequency (Hz)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('output of generated sinc pulse in frequency domain ','linewidth',10)

figure;
subplot(2,1,1); plot(t_axis,pulse22_after_chann_T,'r','linewidth',2);
grid on;
xlim([0 T_sq*5])
xlabel('Time (s)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('output of generated sinc pulse in time domain ','linewidth',10)

subplot(2,1,2); plot(f_axis,abs(pulse22_after_chann),'b','linewidth',2);
grid on
xlim([-1/T_sq 1/T_sq]*5)
ylim([0 50])
xlabel('Frequency (Hz)','linewidth',2)
ylabel('Amplitude','linewidth',2)
title('output of generated sinc pulse in frequency domain ','linewidth',10)

%%%%%%%%%%%%part2
%% parameters
N = 1000; 
No= 1; 
L = 1000;

X = randi([0 1],1,N);
for i=1:N
    if(X(i) == 0)
     X(i) = -1;
    end
end 
X = X';
H = MultipathChannel(L);
N = AWGN(X,No);
Y = H*X + N ; 
%% get x from y
% Minimum Mean Squared Error Equalizer
X_MMSE = ((inv((H' * H) + eye(size(H)) * No ))* H') * Y;
for i=1:length(Y)
    if(real(X_MMSE(i)) > 0)
        X_MMSE(i) = 1;
    else
        X_MMSE(i) = -1;
    end
end ;
% ZERO FORCING Equalizer
X_ZFE =  ( inv( (H' * H) )* H') * Y;
for i=1:length(Y)
    if(real(X_ZFE(i)) > 0)
        X_ZFE(i) = 1;
    else
        X_ZFE(i) = -1;
    end
end; 

%% Get BER For MMSE
BER_MMSE = ComputeBER(X,X_MMSE)
% Get BER For ZFE
BER_ZF = ComputeBER(X,X_ZFE) 

%% getting the graph
Eb_No_axis = -20:0;
BER = zeros(size(Eb_No_axis));
BER_MMSE = zeros(size(Eb_No_axis));
BER_ZF = zeros(size(Eb_No_axis));
for i=1:length(Eb_No_axis)
    No = (10^( -1 * (Eb_No_axis(i)/10)));
    N = AWGN(X,No);
    Y = H * X + N;
    X_estimated = inv(H) * Y;
    X_MMSE = ((inv((H' * H) + eye(size(H)) * No ))* H') * Y;
    for n=1:length(Y)
        if(real(X_MMSE(n)) > 0)
            X_MMSE(n) = 1;
        else
            X_MMSE(n) = -1;
        end
    end;
    X_ZFE = ( inv( (H' * H) )* H') * Y;
    for m=1:length(Y)
        if(real(X_ZFE(m)) > 0)
            X_ZFE(m) = 1;
        else
            X_ZFE(m) = -1;
        end
    end;
    BER(i) = ComputeBER(X,X_estimated);
    BER_MMSE(i) = ComputeBER(X,X_MMSE); 
    BER_ZF(i) = ComputeBER(X,X_ZFE);
end
%graph
figure()
semilogy(Eb_No_axis,BER_MMSE,'--','linewidth',2)
hold on
semilogy(Eb_No_axis,BER_ZF,'-r','linewidth',1)
semilogy(Eb_No_axis,BER,'-g','linewidth',2)
hold off
legend({'MMSE Equilizer','ZF Equilizer','Without Equilizer'},'linewidth',2)
xlabel('Eb/No','linewidth',2)
ylabel('BER','linewidth',2)

%%%%%%%%%%part3
%% Parameters
r = 1/9;            %code rate
p = 0.1:0.01:0.5;   %bit flipping probability
L=1/r;              %number of repetitions
N_bits = 10000;     %number of bits in the sequence

%% Generate Data Bits
data_bits = round(rand(1,N_bits));

%% Generate Chips
chips = zeros(size(data_bits*L));
for i = 1:(length(data_bits)*L)
   chips(i) = data_bits(ceil(i/L));
end

%% Binary Symmetric Channel
ch_effect(1:length(p),1:(L*N_bits)) = 0;
for i = 1:length(p)
    ch_effect(i,:) = rand(size(chips))<=p(i);
end
received_chips(1:length(p),1:(L*N_bits)) = 0;
for i = 1:length(p)
    received_chips(i,:) = xor(chips,ch_effect(i,:));
end

%% Decoder
received_data_bits(1:length(p),1:(L*N_bits)/L) = 0;
majority=0;
for j = 1:length(p)
    for i = 1:length(received_chips)
        majority=majority+received_chips(j,i);
        if ~(mod(i,L))                  %Enter the conditionn statement every L times
            if majority>=(floor(L/2)+1) %half the value of L 
                received_data_bits(j,i/L)=1;
            else
                received_data_bits(j,i/L)=0;
            end
            majority=0;                 %reset the variable for the next iteration
        end
    end
end

%% Compute BER
BER = zeros(size(p));
for i = 1:length(p)
    BER(i) = sum(xor(data_bits,received_data_bits(i,:)))/length(data_bits);
end
figure;
plot(p,BER);
title('BER vs. p');
xlabel('p'); 
ylabel('BER');
ax = gca;
ax.FontSize = 16;