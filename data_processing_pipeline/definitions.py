from enum import IntEnum


class Stage(IntEnum):
    READ = 0
    EXTRACT = 1
    STORE = 2
    TRANSFORM = 3
    INVALID = 4
