#!/usr/bin/bash

x=""
times=""
for epochs in {1..2..4}
do
    time=$(python vgg16.py --epochs="$epochs" --batchsize=32 --steps_per_epoch=100 --repeat=1)
    x="$x $epochs,"
    times="$times $time"
done

echo "$times"
#echo  "{" > results.json
#printf '"epochs" : [%s],' "$x" >> results.json
#printf '"times" : [%s]' "$times" >> results.json
#echo "}" >> results.json
