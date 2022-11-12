import logging
import re, os
from logging.handlers import TimedRotatingFileHandler

from hyperparameter import args

if not os.path.exists(os.path.dirname(args.logfile)):
    os.makedirs(os.path.dirname(args.logfile))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(asctime)s %(filename)s %(message)s')

# ============================================================================#
# 设置日志的handler格式：                                                                                                                                                     #
# * handler: 每 1(interval) 天(when) 重写1个文件,保留30(backupCount) 个旧文件                                                                  #
# * when还可以是s/m/h, 大小写不区分                                                                                                                                    #
# ============================================================================#
filehandler = TimedRotatingFileHandler(args.logfile, when='d', interval=1, backupCount=30)
filehandler.suffix = r"%Y-%m-%d_%H-%M-%S.log"  # 设置历史文件后缀
filehandler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.log$")
filehandler.setFormatter(formatter)

logger.addHandler(filehandler)
