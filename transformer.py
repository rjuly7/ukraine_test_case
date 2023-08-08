class Transformer:
    def __init__(self, 
                 hv_bus: int,
                 lv_bus: int, 
                 hv: int, 
                 lv: int, 
                 sn_mva: int, 
                 vk_percent: float, 
                 vkr_percent: float,
                 pfe_kw: float,
                 i0_percent: float):
                
        self.hv_bus = hv_bus
        self.lv_bus = lv_bus
        self.hv = hv
        self.lv = lv
        self.sn_mva = sn_mva

        self.vk_percent = vk_percent
        self.vkr_percent = vkr_percent
        self.pfe_kw = pfe_kw

        self.i0_percent = i0_percent

        self.dict = {"hv_bus": self.hv_bus, 
                     "lv_bus": self.lv_bus, 
                     "hv": self.hv, 
                     "lv": self.lv, 
                     "sn_mva": self.sn_mva, 
                     "vk_percent": self.vk_percent, 
                     "vkr_percent": self.vkr_percent, 
                     "pfe_kw": self.pfe_kw, 
                     "i0_percent": self.i0_percent}
    
    def __str__(self) -> str:
        return str(self.dict)
