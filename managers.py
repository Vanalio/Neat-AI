class IdManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(IdManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.current_id = 0
        return cls._instance

    @staticmethod
    def get_new_id():
        instance = IdManager()
        instance.current_id += 1
        return instance.current_id

class InnovationManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(InnovationManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.current_innovation = 0
        return cls._instance

    @staticmethod
    def get_new_innovation_number():
        instance = InnovationManager()
        instance.current_innovation += 1
        return instance.current_innovation