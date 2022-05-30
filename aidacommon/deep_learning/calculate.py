#usage python calculate.py 90
import sys

num= sys.argv[1]
with open("result.txt","r") as numbers_file:
    total = 0
    squareDev = 0
    for line in numbers_file:
        try:
            total += float(line)
        except ValueError:
            pass
    average = total/int(num)

with open("result.txt","r") as numbers_file:
    for line in numbers_file:
        try:
            squareDev += (float(line)-average)*(float(line)-average)
        except ValueError:
            pass
open("result.txt", "w").close()
var = squareDev/int(num)
print("average:" + str(average)+" ;var:"+str(var))
