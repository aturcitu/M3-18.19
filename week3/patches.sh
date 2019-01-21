#! /bin/bash

patch_size_values=( 4 8 16 32 64 )
bow_values=( yes )
epoch=150
index=6
for patch_size in "${patch_size_values[@]}"; do
for bow in "${bow_values[@]}"; do
                    work_dir=/home/grupo01/projectF/results/patches_$index
                    mkdir -p $work_dir
                    config_file=$work_dir/config.ini
                    echo "[DEFAULT]" >> $config_file
                    echo "PATCH_SIZE = ${patch_size}" >> $config_file
                    echo "EPOCH = ${epoch}" >> $config_file
                    echo "BOW = ${bow}" >> $config_file
		    index=$((index+1))
done
done

