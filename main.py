import time
import argparse

import getpak
from getpak.commons import Utils as U
from getpak import automation as A


"""
Author: David Guimaraes - dvdgmf@gmail.com
"""

# ,-------------,
# | ENTRY POINT |
# '-------------'
if __name__ == '__main__':
    # ,--------------,
    # | Start timers |
    # '--------------'
    U.tic()
    t1 = time.perf_counter()

    # ,------,
    # | LOGO |
    # '------'
    U.print_logo()

    # ,-----------------,
    # | Present options |
    # '-----------------'
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', help='Displays current package version.', action='store_true')
    parser.add_argument('-mg', help="Given a path, return GRS metadata", default='local', type=str)
    args = parser.parse_args().__dict__  # Converts the input arguments from Namespace() to dict

    print('User Input Arguments:')
    for key in args:
        print(f'{key}: {args[key]}')

    # ,----------------------------------------,
    # | Automation pipelines class declaration |
    # '----------------------------------------'
    gpk_pipe = A.Pipelines()

    # ,---------------,
    # | Treat options |
    # '---------------'
    if args['version']:
        print(f'GET-Pak version: {getpak.__version__}')
    elif args['mg']:
        gpk_pipe.get_grs_metadict(args['mg'])
    else:
        print('No commands supplied, exiting.\n')

    # ,------------------------------,
    # | End timers and report to log |
    # '------------------------------'
    t_hour, t_min, t_sec, _ = U.tac()
    t2 = time.perf_counter()
    final_msg_1 = f'Finished in {round(t2 - t1, 2)} second(s).'
    final_msg_2 = f'Elapsed execution time: {t_hour}h : {t_min}m : {t_sec}s'
    print(final_msg_1)
    print(final_msg_2)
    # ,-----,
    # | END |
    # '-----'
