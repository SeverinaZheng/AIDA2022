#!/bin/bash


TASK="nn"
for ((i=1;i <= 2 ;i++))
do
        NAME="$TASK$i"
        echo "$NAME"
        #python ../new_Switch.py $NAME 1000000 4000&
	python ../read_*Swi* $NAME 180000 18000&
        sleep 3
done

NAME="NN_CPU"
python ../read_*Swi* $NAME 1000 40000&
sleep 50

for ((i=3;i <= 7 ;i++))
do
        NAME="$TASK$i"
        echo "$NAME"
        #python ../new_Switch.py $NAME 1000000 4000&
        python ../read_*Swi* $NAME 180000 18000&
        sleep 3
done

NAME="NN_GPU"
python ../read_*Swi* $NAME 25000 60000&
sleep 3


