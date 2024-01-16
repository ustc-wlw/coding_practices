import collections
from concurrent import futures

# import tqdm

from flags2_sequential import download_one
from flags2_common import *

def download_many(cc_list, base_url, verbose, concur_req):
    cc_list = sorted(cc_list)[:5]
    counter = collections.Counter()
    with futures.ThreadPoolExecutor(max_workers=concur_req) as executor:
        to_do_map = {}
        for cc in cc_list:
            future = executor.submit(download_one, cc, base_url, verbose)
            to_do_map[future] = cc

        results = []
        future_gen = futures.as_completed(to_do_map)
        for future in future_gen:
            '''handle exception'''
            try:
                result = future.result()
            except requests.exceptions.HTTPError as exc:
                error_msg = 'HTTP error {res.status_code} - {res.reason}'
                error_msg = error_msg.format(res=exc.response)
            except requests.exceptions.ConnectionError as exc:
                error_msg = 'Connection error'
            else:
                error_msg = ''
                status = HTTPStatus.ok
            
            if error_msg:
                status = HTTPStatus.error
            counter[status] += 1
            if verbose and error_msg:
                cc = to_do_map[future]
                print('**** Error for {} {}'.format(cc, error_msg))

    return counter

def main(download_many):
    t0 = time.time()
    count = download_many(POP20_CC, BASE_URL, True, 3)
    elapsed = time.time()
    msg = '\n{} flags downloaded in {:.2f}s'
    print(msg.format(count, elapsed))

if __name__=='__main__':
    main(download_many)
