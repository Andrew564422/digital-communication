function BER = ComputeBER(bit_seq,rec_bit_seq)
 
BER=sum(bit_seq~=rec_bit_seq)/length(bit_seq);
 
end
