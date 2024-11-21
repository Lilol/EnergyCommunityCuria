from enum import IntEnum


class Stage(IntEnum):
    READ = 0
    EXTRACT = 1
    TRANSFORM = 2
    STORE = 3
    WRITE_OUT = 4
    INVALID = 5
