import time
from apscheduler.schedulers.blocking import BlockingScheduler



def scheduled_job():   
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

sched = BlockingScheduler(timezone="Asia/Jakarta")
sched.add_job(scheduled_job, "interval", minutes=2)

sched.start()