from flags_threadpool import *

def download_many(cc_list):
    cc_list = cc_list[:5]
    with futures.ThreadPoolExecutor(max_workers=3) as executor:
        to_do = []
        for cc in cc_list:
            future = executor.submit(download_one, cc)
            to_do.append(future)
            msg = 'Scheduled for {} {}'
            print(msg.format(cc, future))

        results = []
        future_iter = futures.as_completed(to_do)
        print('futrue iter: ', future_iter)
        for future in future_iter:
            result = future.result()
            msg = '{} result {!r}'
            print(msg.format(future, result))
            results.append(result)
    
    return len(results)


if __name__=='__main__':
    main(download_many)