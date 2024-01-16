from enum import Enum
import os
import time
import sys
from collections import namedtuple

import requests

POP20_CC = ('CN IN US ID BR PK NG BD RU JP ' \
            'MX PH VN ET EG DE IR TR CD FR').split()

BASE_URL = 'http://flupy.org/data/flags'

DEST_DIR = './downloads/'

default_concur_req = 3
max_concur_req = 5


Result = namedtuple('Result', 'status name')
HTTPStatus = Enum('Status', 'ok not_found error')

def save_flag(img, filename):
    path = os.path.join(DEST_DIR, filename)
    with open(path, 'wb') as fw:
        fw.write(img)