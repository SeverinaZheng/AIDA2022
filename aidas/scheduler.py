from abc import ABCMeta;
import signal;
from collections import deque;
import threading;
import GPUtil;
import time;
from timeloop import Timeloop;
from datetime import timedelta;
from tblib import pickling_support;
pickling_support.install();
import dill as custompickle;

import logging;

from aidacommon.aidaConfig import AConfig;

class ScheduleManager(metaclass=ABCMeta):
    """Singleton class, there will be only one schedule manager in the system"""
    __ClassLock = threading.RLock();
    __ScheduleManagerObj = None;

    @staticmethod
    def getScheduleManager():
        class __ScheduleManager:
            """Class that dispatching jobs"""

            __RepoLock = threading.RLock();
            __maybe_available = threading.Condition();
            __GPUQueue = deque();
            __CPUQueue = deque();
            __GPU_inuse = deque();
            __CPU_inuse = deque();

            def __init__(self):
                SchMgrObj = self;

                def start(self):
                    logging.info("background thread is running");
                    self.__maybe_available.acquire();

                    while(True):
                        logging.info("wait");
                        self.__maybe_available.wait();
                        #incase too quick to send another request 
                        #before the former is really start working
                        self.__maybe_available.acquire();
                        if(len(self.__GPUQueue)> 0):
                            cv = self.__GPUQueue.popleft();
                            logging.info(cv);
                            time.sleep(2);
                            deviceID = GPUtil.getFirstAvailable(order = 'last', maxLoad=0.5, maxMemory=0.5, attempts=1)
                            if 1 == deviceID[0]:
                                logging.info("try to invoke gpu");
                                self.invoke_GPU(cv);
                            elif(len(self.__CPU_inuse) == 0):
                                self.invoke_CPU(cv);
                        elif(len(self.__CPUQueue)> 0):
                            if(len(self.__CPU_inuse) == 0):
                                self.invoke_CPU;

                #Handle signals to exit gracefully.
                if(threading.current_thread() == threading.main_thread()):
                    signal.signal(signal.SIGINT, self.terminate);
                    signal.signal(signal.SIGTERM, self.terminate);

                #Start the server polling as a daemon thread.
                self.__srvrThread = threading.Thread(target=start,args=(self,));
                self.__srvrThread.daemon = True;
                self.__srvrThread.start();


            def wakeup_scheduler(self):
                self.__maybe_available.acquire();
                self.__maybe_available.notify();
                self.__maybe_available.release();


            def finish_GPU(self,condition_var):
                self.__GPU_inuse.remove(condition_var);
                self.wakeup_scheduler();

            def finish_CPU(self,condition_var):
                self.__CPU_inuse.remove(condition_var);
                self.wakeup_scheduler();

            def schedule_GPU(self, condition_var):
                with __ScheduleManager.__RepoLock:
                    self.__GPUQueue.append(condition_var);
                    logging.info(len(self.__GPUQueue));
                    logging.info(self.__GPUQueue[0]);
                    self.wakeup_scheduler();
                    logging.info("end schedule");

            def schedule_CPU(self, condition_var):
                with __ScheduleManager.__RepoLock:
                    self.__CPUQueue.append(condition_var);
                    self.wakeup_scheduler();


            def in_GPU(self,condition_var):
                if(self.__GPU_inuse.count(condition_var) > 0):
                    return True;
                else: return False;

            def invoke_GPU(self,cv):
                logging.info("reach invoke_gpu");
                self.__GPU_inuse.append(cv);
                cv.acquire();
                cv.notify();
                cv.release();
                logging.info("to GPU");


            def invoke_CPU(self,cv):
                #tasks in GPUQueue has higher priority
                self.__CPU_inuse.append(cv);
                logging.info("ready to cpu");
                cv.acquire();
                cv.notify();
                cv.release();
                logging.info("to CPU");


            def close(self):
                self.__srvr.shutdown();
                self.__srvr.server_close();

            def terminate(self, signum, frame):
                self.close();

        with ScheduleManager.__ClassLock:
            if (ScheduleManager.__ScheduleManagerObj is None):  # There is no connection manager object currently.
                schmgr = __ScheduleManager();
                ScheduleManager.__ScheduleManagerObj = schmgr;
            logging.info("end of init");

            # Return the connection manager object.
            return ScheduleManager.__ScheduleManagerObj;

