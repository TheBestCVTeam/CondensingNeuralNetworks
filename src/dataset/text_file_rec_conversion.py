def get_filename(line: str):
    return line[:-3]


def get_label(line: str):
    return line[-2:]


def get_bb_fn(filename):
    return filename[:-4] + '_BB.txt'
