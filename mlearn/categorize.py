from __init__ import *

logging.basicConfig(format=colored('%(levelname)s:', attrs=['bold'])
                    + colored('%(name)s:', 'blue') + ' %(message)s')
logger = logging.getLogger('CATML')
logger.setLevel(level=logging.INFO)





