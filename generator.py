import numpy as np

COSTS = {"Hydro": [0.9, 0.1], "Nuclear": [7.2504, 0.7534], "Coal": [24.7919, 8.0866], "Solar": [0, 0], "Gas": [34.2731, 10.9810]}

np.random.seed(42)

class Generator:
    def __init__(self, name: str, min_real_power: float, max_real_power: float, min_reactive_power: float, max_reactive_power: float, longitude: float, latitude: float, power_type: str, voltage_level: set):
        self.name = name 
        self.min_real_power = min_real_power
        self.max_real_power = max_real_power

        self.min_reactive_power = min_reactive_power
        self.max_reactive_power = max_reactive_power

        self.longitude = longitude
        self.latitude = latitude

        self.power_type = power_type

        # if(self.power_type == "Coal" or "Nuclear" or "Gas"):
        #     self.voltage = 450 #V
        # else:
        #     self.voltage = 220 #V

        self.voltage = voltage_level

        mu = COSTS[self.power_type][0]
        sigma = COSTS[self.power_type][1]
        sample = np.random.normal(mu, sigma)
        self.costs = sample
        self.gen_bus = -1
        self.substation = -1   

        self.dict = {'pmin': self.min_real_power, 
                     'pmax': self.max_real_power, 
                     'qmin': self.min_reactive_power, 
                     'qmax': self.max_reactive_power, 
                     'latitude': self.latitude, 
                     'longitude': self.longitude, 
                     'power_type': self.power_type, 
                     'voltage': self.voltage,
                     'gen_bus': self.gen_bus}
        
        self.bus_type = 1

    def __str__(self) -> str:
        return str(self.dict)