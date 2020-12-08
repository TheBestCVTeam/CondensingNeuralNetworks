"""
Has no dependencies on any other project code.
Used for functions that implement behaviour that doesn't need dependencies
and is useful to make reusable
"""
import ntpath
import os
import re
import time
from pathlib import Path
from typing import Any, List, Type


def mkdir_w_par(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)


def str_to_class(cls_name: str) -> type:
    """
    Converts a string into the class that it represents

    NB: Code based on https://stackoverflow.com/questions/452969/does-python
    -have-an-equivalent-to-java-class-forname
    :param cls_name: The string representation of the desired class
    :return: A pointer to the class (Able to be used as a constructor)
    """
    parts = cls_name.split('.')
    modules = '.'.join(parts[:-1])
    result = __import__(modules)
    for comp in parts[1:]:
        result = getattr(result, comp)
    return result


def strs_to_classes(cls_names: List[str]) -> List[type]:
    result = []
    for s in cls_names:
        result.append(str_to_class(s))
    return result


def get_run_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H%M")


def get_log_folder() -> str:
    """
    Returns the folder logs should be written to and ensures the folder exists
    :return: folder logs should be written to
    """
    result = 'log/'
    mkdir_w_par(result)
    return result


def get_valid_ident(org):
    """Turns the parameter passed into a valid identifier"""
    result = re.sub(r' ', '_', str(org))
    result = re.sub(r'[^A-Za-z0-9_]', '', result)
    return result


def public_members_as_dict(class_: Type[Any]) -> dict:
    """
    Converts a class into a dict of it's public values
    :param class_: the class to be converted
    :return: the public values from the class
    """
    result = {}
    for i in class_.__dict__.items():
        if not (i[0][0] == "_" or isinstance(i[1], classmethod)):
            if not isinstance(i[1], type):
                result[i[0]] = i[1]
            else:
                result[i[0]] = public_members_as_dict(i[1])
    return result


def save_list_to_file(file_list: List[str], file_name: str, end: str = ''):
    """
    Saves file_list to a file with file_name and appends end
    :param file_list: List of string to be saved to the file
    :param file_name: Name of file to be created or overwritten
    :param end: Optional line ending to append to each line
    :return:
    """
    ensure_folder_created(file_name)
    with open(file_name, 'w') as f:
        for item in file_list:
            f.write('%s%s' % (item, end))


def change_filename_ext(fn: str, new_ext: str) -> str:
    """
    Returns the filename with the next extension
    :param fn: The filename with the old extension
    :param new_ext: The new extension to put on the filename
    :return: The filename with the new extension
    """
    result, _ = os.path.splitext(fn)
    return f'{result}{new_ext}'


def ensure_folder_created(fn: str):
    """
    Expects fn to be a file name. Removes the file name and ensures that
    it's parent folder is exists.
    :param fn: Filename to use to find parent folder
    :return:
    """
    folder, _ = ntpath.split(fn)
    mkdir_w_par(folder)
