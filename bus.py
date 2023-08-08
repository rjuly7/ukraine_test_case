import numpy as np

class Bus:
    def __init__(self, name: int, latitude: float, longitude: float, voltage_level : int, substation: int):
        self.name = name  

        self.latitude = latitude
        self.longitude = longitude

        self.voltage = voltage_level

        self.v_nom = voltage_level
        self.v_min = voltage_level*0.9
        self.v_max = voltage_level*1.1

        self.substation = substation 

        self.dict = {'latitude': self.latitude,
                     'longitude': self.longitude,
                     'voltage': self.voltage,
                     'v_nom': self.v_nom,
                     'v_min': self.v_min,
                     'v_max': self.v_max,
                     'substation': self.substation}
        
    def __str__(self) -> str:
        return str(self.dict)