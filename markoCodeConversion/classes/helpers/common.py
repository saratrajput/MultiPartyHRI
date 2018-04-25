import os
import shutil

from config import config


def listify(obj):
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def dictify(lst, inv=False):
    if inv:
        return {k: v for k, v in zip(lst, range(len(lst)))}
    else:
        return {k: v for k, v in zip(range(len(lst)), lst)}


def clear_dir(path, subdir=False):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path) and subdir:
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def match_lists_to_idx(list_to_match, sublist):
    return [list_to_match.index(item) for item in sublist]


def list_take(lst, indexes):
    ret = []
    for index in indexes:
        ret.append(lst[index])
    return ret


def get_f_names_in_dir(directory, full_path=False):
    # read all files in the corresponding directory, either full path or name only
    if full_path:
        f_names = [''.join([root, name])
                   for root, dirs, files in os.walk(directory)
                   for name in files]
    else:
        f_names = [name
                   for root, dirs, files in os.walk(directory)
                   for name in files]
    return f_names
