import numpy as np

class Filterbank:
    def __init__(self) -> None:
        pass

class BaseFilter:
    def __init__(self) -> None:
        self.test = 1

    def resonatorABCD(self):
        pass


class TransmissionLine:
    def __init__(self,Z0,kc,l) -> None:
        self.kc = kc
        self.Z0 = Z0
        self.l = l

class Resonator(TransmissionLine):
    def __init__(self, Z0, kc, l_res, C_top, C_bottom) -> None:
        super().__init__(Z0, kc, l_res)
        self.l_res = self.l
        self.C_top = C_top
        self.C_bottom = C_bottom
    

class DirectionalFilter(BaseFilter):
    def __init__(self) -> None:
        super().__init__()
    
