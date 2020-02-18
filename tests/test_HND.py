import numpy as np
import numpy.testing as npt

import HND

def test_HND_smoke():
    # Smoke test
    obj = HND.HouseNumberDetector()

def test_HND_func1():
    # Test func1
    obj = HND.HouseNumberDetector()
    output = obj.func1()

    npt.assert_equal(output, 'Test')
