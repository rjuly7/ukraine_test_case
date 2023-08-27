import numpy as np

#peak demand January 2022 22 GW, 43.9 million people 
REAL_POWER_PER_CAPITA = 0.503/1000 #MW
alpha = 2/np.sqrt(2**2 + 0.57**2)
REACTIVE_POWER_PER_CAPITA = np.sqrt(np.square(REAL_POWER_PER_CAPITA)*(1/np.square(alpha)-1)) #Mvar

LOAD_VOLTAGE_CHOICES = np.array([220, 330, 450])

class Load:
    def __init__(self, name: str, latitude: float, longitude: float, population: int, voltage_level : set):
        self.name = name

        self.latitude = latitude
        self.longitude = longitude

        self.population = population

        self.voltage = voltage_level

        self.real_power_demanded = self.population * REAL_POWER_PER_CAPITA
        self.reactive_power_demanded = self.population * REACTIVE_POWER_PER_CAPITA

        self.dict = {'pd': self.real_power_demanded,
                     'qd': self.reactive_power_demanded, 
                     'latitude': self.latitude,
                     'longitude': self.longitude,
                     'voltage': self.voltage}
        
        self.is_assigned = False

        self.bus_type = 2

        self.load_bus = -1 
        self.substation = -1 
        
    def __str__(self) -> str:
        return str(self.dict)