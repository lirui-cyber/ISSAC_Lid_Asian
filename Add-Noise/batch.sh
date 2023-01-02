#!/usr/bin

test="en_test zh_test id_test tl_test th_test vi_test test_zh_en_id_tl test_zh_en_id_tl_th_vi"

for x in $test;do

  bash add-noise-for-lid.sh --steps 2 --src-train ../data/$x --noise_dir ../data/rats_noise_channel_AEH

done
