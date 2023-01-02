#!/usr/bin

train="train_zh_en_id_tl_lo_my"
test="zh_test en_test id_test tl_test th_test vi_test"
data=data
save_wav_dir=source-data/asia-noise
save_dir=`pwd`/${save_wav_dir}

if [ ! -d ${save_wav_dir} ];then
    mkdir -p ${save_wav_dir}
fi

cd Add-Noise

bash add-noise-for-lid.sh --steps 2 --src-train ../${data}/${train} --noise_dir ../${data}/rats_noise_channel_BCDFG 
for x in ${test};do
    bash add-noise-for-lid.sh --steps 2 --src-train ../${data}/${x} --noise_dir ../${data}/rats_noise_channel_AEH
done

cd ..

for x in ${train} ${test};do
    for snr in 20 15 10 5;do
        dir=${x}_${snr}_snrs
	mkdir ${save_wav_dir}/${dir}
        cat ${data}/${dir}/wav.scp | awk -v n=$dir -v p=${save_dir} '{l = length($0); a = substr($0, 0,length-3); print $2" "$3" "$4" "$5" "$6" "$7 " " p "/" n "/" $1 ".wav"}' > ${data}/${dir}/${dir}.cmd
        bash utils/generate_new_wav_cmd.sh ${data}/${dir}/${dir}.cmd
    done
done
for x in ${test};do
    for snr in 20 15 10 5;do
	dir=${x}_${snr}_snrs
        local=${save_dir}/${dir}/
        cat ${data}/${dir}/wav.scp | awk -v p=$local '{print $1 " " p $1 ".wav"}' > ${data}/${dir}/new_wav.scp 
        mv ${data}/${dir}/new_wav.scp ${data}/${dir}/wav.scp
    done
done
for x in ${train};do
    for y in 20 15 10 5;do
	dir=${x}_${y}_snrs
	snrs=_${y}_snrs
        local=${save_dir}/${dir}/
	rm ${data}/${dir}/{reco2dur,utt2uniq,spk2utt,wav.scp}
	cat ${data}/${dir}/utt2lang | awk -v p=$local -v s=${snrs} '{print $1 s" " p $1".wav"}' > ${data}/${dir}/wav.scp
	cat ${data}/${dir}/utt2lang | awk -v p=$local -v s=${snrs} '{print $1 s" " $2}' > ${data}/${dir}/utt2spk
	./utils/utt2spk_to_spk2utt.pl ${data}/${dir}/utt2spk > ${data}/${dir}/spk2utt
        cp ${data}/${dir}/utt2spk ${data}/${dir}/utt2lang  
        utils/fix_data_dir.sh ${data}/${dir}
    done

done


./utils/combine_data.sh ${data}/${train}_5th ${data}/${train} ${data}/${train}_20_snrs ${data}/${train}_15_snrs ${data}/${train}_10_snrs ${data}/${train}_5_snrs  
