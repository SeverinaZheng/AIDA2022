import psutil
import time
import GPUtil

#for i in range(10):
#    print(float(psutil.cpu_percent()))
#    time.sleep(0.5)


gpus = GPUtil.getGPUs()
print("id:" + str(gpus[1].id)+"util:"+ str(gpus[1].load))
