#!/usr/bin/sh

file_name="mnist_cnn_scale_epochs.json"
x=""
times=""
batchsize=32
steps=1
for epochs in {1,2,3,4,5,6,7,8,9,10}
do
    time=$(srun python cnn_mnist.py --epochs="$epochs" --batchsize="$batchsize" --steps_per_epoch="$steps" --repeat=5 --device="GPU:0")
    x="$x $steps,"
    times="$times $time"
done

echo  "{" > $file_name
printf '"batchsize" : %s,' "$batchsize" >> $file_name
printf '"steps" : [%s],' "$x" >> $file_name
printf '"times" : [%s]' "$times" >> $file_name
echo "}" >> $file_name

