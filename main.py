import time
import pickle
import getpak
import argparse

from getpak import automation as A
from getpak.commons import Utils as U


"""
Author: David Guimaraes - dvdgmf@gmail.com
"""

# ,-------------,
# | ENTRY POINT |
# '-------------'
def main():
    # ,------,
    # | LOGO |
    # '------'
    U.print_logo()
    
    # ,-----------------,
    # | Present options |
    # '-----------------'
    parser = argparse.ArgumentParser()
    
    # I/O
    parser.add_argument('-i', '--input', help='Input file/folder', type=str)
    parser.add_argument('-o', '--output', help='Output directory', type=str)
    # Pipeline mode
    parser.add_argument('-gp', '--getpipe', help='Run in pipeline mode', action='store_true')
    parser.add_argument('-tid', '--tileid', help='internal tile ID for L2B calculation', type=str)
    # version
    parser.add_argument('-v', '--version', help='Displays current package version.', action='store_true')

    args = parser.parse_args().__dict__  # Converts the input arguments from Namespace() to dict

    print('User Input Arguments:')
    for key in args:
        print(f'{key}: {args[key]}')

    # ,----------------------------------------,
    # | Automation pipelines class declaration |
    # '----------------------------------------'
    gpk_pipe = A.Pipelines()
    id = gpk_pipe.INSTANCE_TIME_TAG
    print(f'GET-Pak instance run #{id}')  # (YYYYMMDDTHHMMSS)

    # ,---------------,
    # | Treat options |
    # '---------------'
    if args['version']:
        print(f'GET-Pak version: {getpak.__version__}')

    elif args['getpipe']:
        
        matches, str_matches, dates, meta = gpk_pipe.get_matchups()
        
        l2b_results = gpk_pipe.matchups_to_l2b(matches, str_matches, dates, meta)

        # l2b_results contains the paths to start the time-series processing
        pkl_path_name = f'getpak_{id}.pkl'
        print(f'Saving processing metadata to: {pkl_path_name} ..')
        with open(pkl_path_name, "wb") as f:
            pickle.dump(l2b_results, f)

    else:
        print('Exiting.\n')


if __name__ == '__main__':
    # ,--------------,
    # | Start timers |
    # '--------------'
    U.tic()
    t1 = time.perf_counter()
    # ,--------------,
    # | Call GET-Pak |
    # '--------------'
    main()
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
