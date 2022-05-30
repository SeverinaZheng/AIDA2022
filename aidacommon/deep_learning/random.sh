#!/bin/bash

# calculation: one job on CPU: 80s, on GPU: 15s
# maximum number of jobs in 15 min:  11.25 jobs on CPU, 60 jobs on GPU
# in total: 61 jobs
# 80% capacity of system allows: 61*0.8 = 48.8 jobs -> 48 jobs
# first find the randomized submission time&interval
# ./random.sh 20 5 10000 1000 means submit 20 jobs in 5 min interval randomly

num=$1
sec=$(($2*60))
totalTime=0
startTime=()
startInterval=()
for ((i=1;i <= $num;i++))
do
	Time=$(($RANDOM%$sec))
	#echo $Time
	startTime+=( $Time )

done

readarray -t sorted < <(printf '%s\n' "${startTime[@]}" | sort -n)
printf '%s ' "${sorted[@]}"
printf '\n'
echo "Done"

startInterval+=(${sorted[0]})
for ((i=1;i < $num;i++))
do
	interval=$((${sorted[$i]}-${sorted[$(($i-1))]}))
        startInterval+=($interval)
done

for ((i=1;i <= $num;i++))
do
	sleep ${startInterval[$(($i-1))]}
        NAME="from1:task$i"
        echo "$NAME"
        python Switch_between.py $NAME $3 $4&
done


