import unittest
import numpy as np

from filterbank.components import Resonator,TransmissionLine,DirectionalFilter,ReflectorFilter,ManifoldFilter,Filterbank,BaseFilter
from filterbank.transformations import chain


class TestChainingABCD(unittest.TestCase):
    def test_chain(self):
        """
        Test that chaining two equally sized matrices works.
        """
        ABCD1 = [[6, 7],
                 [8, 9]]
        ABCD2 = [[6, 7],
                 [8, 9]]
        result = chain(ABCD1,ABCD2)
        self.assertIsNone(np.testing.assert_array_equal(result, [[92,105],[120,137]]))

if __name__ == '__main__':
    unittest.main()
