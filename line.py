class Line:
    def __init__(self, 
                 voltage: int,
                 from_bus: int, 
                 to_bus: int, 
                 resistance: float, 
                 reactance: float, 
                 capacitance: float, 
                 angle_max: float,
                 line_flow_limit: float,
                 length: float,
                 max_i_ka: float):
        
        self.voltage = voltage
        self.from_bus = from_bus
        self.to_bus = to_bus

        self.resistance = resistance
        self.reactance = reactance
        self.capacitance = capacitance

        self.angle_max = angle_max
        self.line_flow_limit = line_flow_limit
        self.max_i_ka = max_i_ka

        self.length = length

        self.dict = {"f_bus": self.from_bus, 
                     "t_bus": self.to_bus, 
                     "R": self.resistance, 
                     "X": self.reactance, 
                     "C": self.capacitance, 
                     "angmax": self.angle_max, 
                     "line_flow_limit": self.line_flow_limit, 
                     "length": self.length, 
                     "max_i_ka": self.max_i_ka}
    
    def __str__(self) -> str:
        return str(self.dict)