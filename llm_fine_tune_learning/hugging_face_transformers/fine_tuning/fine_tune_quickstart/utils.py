import logging

logging.basicConfig(filename='log.txt',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO, 
                    filemode='a', )
logger = logging.getLogger(__name__)  # 定义一个日志器，定义不同的日志器，方便区分不同的日志信息

