# class SubclassRegistryMeta(type):
#     def __init__(cls, name, bases, dct):
#         super().__init__(name, bases, dct)
#         if hasattr(cls, "_key") and cls._key is not None:
#             cls.register_subclass(cls._key, cls)


class SubclassRegistrationBase:
    _subclasses = {}
    _key = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if len(cls._subclasses) == 0:
            cls._subclasses = {}
        cls._subclasses[cls._key] = cls

    @classmethod
    def register_subclass(cls, key, subclass):
        cls._subclasses[key] = subclass

    @classmethod
    def get_subclass(cls, key):
        return cls._subclasses.get(key, None)
