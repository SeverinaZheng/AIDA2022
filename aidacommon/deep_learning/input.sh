#!/bin/bash

#change $sorted and $num to be corresponding the submission time and #job
#./input.sh 10000 1000

sorted=(10 29 46 47 49 74 91 129 137 148 153 166 167 169 174 186 211 215 232 267 301 338 423 423 460 460 471 542 552 576 580 588 591 594 601 620 645 660 710 725 735 747 759 782 783 805 811 828 831 839 866 875 903 914 928 934 936 941 948 973 979 990 997 1001 1003 1004 1036 1040 1067 1076 1083 1106 1110 1130 1138 1140 1143 1147 1166 1177 1199 1206 1240 1284 1288 1335 1344 1347 1349 1350 1351 1393 1401 1407 1436 1450 1457 1457 1457 1459 1490 1491 1498 1506 1508 1541 1557 1606 1655 1656 1657 1668 1683 1710 1715 1750 1751 1761 1778 1788)
startInterval=()
startInterval+=(${sorted[0]})

echo ${#sorted[@]}
for ((i=1;i < ${#sorted[@]};i++))
do
	interval=$((sorted[$i]-sorted[$(($i-1))]))
        startInterval+=($interval)
done

for ((i=1;i <= ${#sorted[@]};i++))
do
	sleep ${startInterval[$(($i-1))]}
        NAME="from1:task$i"
        echo "$NAME"
        python Switch_between.py $NAME $1 $2&
done


