#!/usr/bin/bash

file_name="scale_steps.json"
x=""
times=""
batchsize=32
for steps in {1,10,20,30,40,50,60,70,80,90,100}
do
    #echo "$batchsize"
    time=$(python vgg16.py --epochs=1 --batchsize="$batchsize" --steps_per_epoch="$steps" --repeat=5)
    #echo $time
    x="$x $steps,"
    times="$times $time"
done

echo  "{" > $file_name
printf '"batchsize" : %s,' "$batchsize" >> $file_name
printf '"steps" : [%s],' "$x" >> $file_name
printf '"times" : [%s]' "$times" >> $file_name
echo "}" >> $file_name
