from enum import IntEnum, auto


class Stage(IntEnum):
    READ = 0
    EXTRACT = auto()
    TRANSFORM = auto()
    CHECK = auto()
    STORE = auto()
    WRITE_OUT = auto()
    VISUALIZE = auto()
    INVALID = auto()
