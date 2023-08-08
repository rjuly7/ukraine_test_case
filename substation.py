import numpy as np

class Substation:
    def __init__(self, name: str, latitude: float, longitude: float, voltage_level : set):
        self.name = name

        self.latitude = latitude
        self.longitude = longitude

        self.voltage = voltage_level

        self.gen_list = []
        self.load_list = []
        self.sub_list = [] 

        self.dict = {'latitude': self.latitude,
                     'longitude': self.longitude,
                     'voltage': self.voltage}
        
    def __str__(self) -> str:
        return str(self.dict)




