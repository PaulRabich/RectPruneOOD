import os

def get_top_level_path():
    # get the absolute path of this file
    abs_path = os.path.dirname(os.path.abspath(__file__))
    # get the parent directory
    top_level_path = os.path.abspath(os.path.join(abs_path, "../../../"))

    return top_level_path