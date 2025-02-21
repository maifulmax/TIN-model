#p = Pool()
    #p.map(funcname,iterable) 默认异步的执行任务,且自带close,join功能
    #p.apply(), 同步调用进程池的方法
    #p.apply_async(),异步调用,和主进程完全异步,需要手动close和join
from multiprocessing import Pool
import time
def func(i): #返回值只有进程池才有,父子进程没有返回值
    time.sleep(0.5)
    print(i)
    return i*i

if __name__ == '__main__':
    p = Pool(5)
    res_l = [] #从异步提交任务获取结果
    for i in range(10):
        # res = p.apply(func,args=(i,)) #apply的结果就是func的返回值,同步提交
        # print(res)

        res = p.apply_async(func, args=(i,)) #apply_sync的结果就是异步获取func的返回值
        res_l.append(res) #从异步提交任务获取结果
        print(res.get())
    #for res in res_l:  print(res.get()) #等着func的计算结果

    p.close()
    p.join()
