#! /bin/bash

img_size_values=( 64 )
batch_size_values=( 16 )
activation1_values=( relu )
activation2_values=( softmax  )
optimizer_values=( sgd )
density_values=( dense3 )
for img_size in "${img_size_values[@]}"; do
    for batch_size in "${batch_size_values[@]}"; do
        for activation1 in "${activation1_values[@]}"; do
            for activation2 in "${activation2_values[@]}"; do
                for optimizer in "${optimizer_values[@]}"; do
                for density in "${density_values[@]}"; do
                    #work_dir=/home/grupo01/workT/job_`date +"%s%N"`
                    work_dir=/home/grupo01/projectL/results_`date +"%s%N"`
                    mkdir -p $work_dir
                    config_file=$work_dir/config.ini
                    echo "[DEFAULT]" >> $config_file
                    echo "IMG_SIZE = ${img_size}" >> $config_file
                    echo "BATCH_SIZE = ${batch_size}" >> $config_file
                    echo "ACTIVATION_FUNCTION1 = ${activation1}" >> $config_file
                    echo "ACTIVATION_FUNCTION2 = ${activation2}" >> $config_file
                    echo "OPTIMIZER = ${optimizer}" >> $config_file
                    echo "DENSITY = ${density}" >> $config_file
                    sbatch job $work_dir
                done
                done
            done
        done
    done
done

