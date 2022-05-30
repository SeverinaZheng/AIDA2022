#!/bin/bash

# the first argument is the total number of epoches in one job 
# the second argument is # epoches in one task
# the third is the time interval betwwen jobs
# the fourth is the total number of jobs
# e.g ./loop.sh 10000 1000 5 4 means run 4 jobs,each with
# 10000 epoches in total, 1000 epoches in each task, submit a job every 5s 
TASK="task"
for ((i=1;i <= $4;i++))
do
	NAME="from1:$TASK$i"
	echo "$NAME"
	python Switch_between.py $NAME $1 $2&
	sleep $3
done
