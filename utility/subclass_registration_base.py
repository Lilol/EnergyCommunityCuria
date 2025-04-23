class SubclassRegistrationBase:
    _subclasses = {}
    _key = None

    def __init_subclass__(cls, **kwargs):
        # if cls._key is None or (hasattr(cls._key, "value") and cls._key.value == "invalid"):
        #     return
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

    @classmethod
    def create(cls, key, *args, **kwargs):
        return cls._subclasses.get(key, None)(*args, **kwargs)