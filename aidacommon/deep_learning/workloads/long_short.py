#!/bin/bash

NAME="long1"
python ../read_from_db_no_iter.py $NAME 0.8&
#python ../read_from_db_no_iter.py $NAME 10&
sleep 3

TASK="nn"
for ((i=1;i <= 3 ;i++))
do
        NAME="$TASK$i"
        echo "$NAME"
	python ../read_*Swi* $NAME 180000 7000&
        sleep 3
done



NAME="long2"
python ../read_from_db_no_iter.py $NAME 0.8&
#python ../read_from_db_no_iter.py $NAME 10&
