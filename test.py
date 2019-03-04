from const import const
from maker import make_const, make_pwr, make_pwr_expr, make_plus, make_prod, make_quot, make_e_expr, make_ln, make_absv
from tof import tof
from deriv import deriv
from antideriv import antideriv
import unittest
import math

class Assign01UnitTests(unittest.TestCase):

    def test_01(self):
        #x^2 => (1/3)x^3
        print("****Unit Test 01********")
        fex = make_pwr('x', 2.0)
        print(fex)
        afex = antideriv(fex)
        assert not afex is None
        def gt(x): return (1.0/3.0)*(x**3.0)
        afexf = tof(afex)
        assert not afexf is None
        err = 0.0001
        for i in range(101):
            assert abs(afexf(i) - gt(i)) <= err
        print(afex)
        print("Unit Test 01: pass")


    def test_02(self):
        #e^(-2x)=> -0.5e^(-2x)
        print("****Unit Test 02********")
        fex = make_e_expr(make_prod(make_const(-2.0), make_pwr('x', 1.0)))
        print(fex)
        afex = antideriv(fex)
        assert not afex is None

        def gt(x): return (-0.5) * (math.e**(-2.0*x))

        afexf = tof(afex)
        assert not afexf is None
        err = 0.0001
        for i in range(0, 101):
            assert abs(afexf(i) - gt(i)) <= err
        print(afex)
        print("Unit Test 02: pass")

    def test_03(self):
        #x^0.5 => (2/3)*x^(3/2)
        print("****Unit Test 03********")
        fex = make_pwr('x', 0.5)
        print(fex)
        afex = antideriv(fex)
        assert not afex is None

        def gt(x): return (2.0/3.0)*(x**(3.0/2.0))

        afexf = tof(afex)
        assert not afexf is None
        err = 0.0001
        for i in range(1, 101):
            assert abs(afexf(i) - gt(i)) <= err
        print(afex)
        print("Unit Test 03: pass")

    def test_04(self):
        #1/(x^2)=> -1.0/x
        print("****Unit Test 04********")
        fex = make_pwr('x',-2.0)
        print(fex)
        afex = antideriv(fex)
        assert not afex is None
        def gt(x): return -1.0/x
        afexf = tof(afex)
        assert not afexf is None
        err = 0.0001
        for i in range(1, 101):
            assert abs(afexf(i) - gt(i)) <= err
        print(afex)
        print("Unit Test 04 pass")

    def test_05(self):
        #1/(x)=> ln|x|
        print("****Unit Test 05********")
        fex = make_pwr('x',-1.0)
        print(fex)
        afex = antideriv(fex)
        assert not afex is None
        def gt(x): return math.log(abs(x), math.e)
        afexf = tof(afex)
        assert not afexf is None
        err = 0.0001
        for i in range(2, 101):
            assert abs(afexf(i)/gt(i) - 1) <= err
        for i in range(-100, 0):
            assert abs(afexf(i) - gt(i)) <= err
        print(afex)
        print("Unit Test 05 pass")

    def test_06(self):
        #x^-3+7e^(5x)+4*x^-1 => (1/-2)*(x^-2)+0 +(7*(1/5)*e(5x)+ (4*ln|x|)
        print("****Unit Test 06********")
        fex1 = make_pwr('x', -3.0)
        fex2 = make_prod(make_const(7.0),
                         make_e_expr(make_prod(make_const(5.0),
                                               make_pwr('x', 1.0))))
        fex3 = make_prod(make_const(4.0),
                         make_pwr('x', -1.0))
        fex4 = make_plus(fex1, fex2)
        fex = make_plus(fex4, fex3)
        print(fex)
        afex = antideriv(fex)
        assert not afex is None
        def gt(x):
            v1 = -0.5*(x**(-2.0))
            v2 = (7.0/5.0)*(math.e**(5.0*x))
            v3 = 4.0*(math.log(abs(x), math.e))
            return v1 + v2 + v3
        afexf = tof(afex)
        assert not afexf is None
        err = 0.001
        for i in range(1, 10):
            print(afexf(i), gt(i))
            assert abs(afexf(i)-gt(i)) <= err * gt(i)
        print("Unit Test 06 pass")

    def test_07(self):
        # 4x^3 find antideriv and then plug it into deriv to get 4x^3 back
        print("****Unit Test 07********")
        fex = make_prod(make_const(4.0), make_pwr('x', 3.0))
        print(fex)
        afex = antideriv(fex)
        assert not afex is None
        print(afex)
        fexf = tof(fex)
        assert not fexf is None
        fex2 = deriv(afex)
        assert not fex2 is None
        print(fex2)
        fex2f = tof(fex2)
        assert not fex2f is None
        err = 0.0001
        for i in range(11): assert abs(fexf(i) - fex2f(i)) <= err
        print('Test 07:pass')

    def test_08(self):
        #(5x-7)^-2  => -1/5 * (5x-7)^-1
        print("****Unit Test 08********")
        fex1 = make_plus(make_prod(make_const(5.0), make_pwr('x', 1.0)), make_const(-7.0))
        fex = make_pwr_expr(fex1, -2.0)
        print(fex)
        afex = antideriv(fex)
        assert not afex is None
        print("antideriv: ", afex)
        afexf = tof(afex)
        err = 0.0001
        def gt(x):
            return (-1.0 / 5.0) * ((5 * x - 7.0) ** -1)

        for i in range(1, 100):
            assert abs(afexf(i) - gt(i)) <= err
        fexf = tof(fex)
        assert not fexf is None
        fex2 = deriv(afex)
        assert not fex2 is None
        print("deriv fex2: ",fex2)
        fex2f = tof(fex2)
        assert not fex2f is None
        for i in range(1, 100):
            print(fexf(i), " ", fex2f(i))
            assert abs(fexf(i) - fex2f(i)) <= err
        print('Test 08:pass')

    def test_09(self):
        #3*(x+2)^-1 => 3*ln|x+2|
        print("****Unit Test 09********")
        fex0 = make_plus(make_pwr('x', 1.0), make_const(2.0))
        fex1 = make_pwr_expr(fex0, -1.0)
        fex = make_prod(make_const(3.0), fex1)
        print(fex)
        afex = antideriv(fex)
        print("antideriv: ", afex)
        err = 0.0001
        afexf = tof(afex)
        def gt(x):
            return 3.0 * math.log(abs(2.0 + x), math.e)
        for i in range(1, 101):
            assert abs(afexf(i) - gt(i)) <= err
        assert not afex is None
        print(afex)
        fexf = tof(fex)
        assert not fexf is None
        fex2 = deriv(afex)
        assert not fex2 is None
        print(fex2)
        fex2f = tof(fex2)
        assert not fex2f is None
        for i in range(1, 1000):
            assert abs(fexf(i) - fex2f(i)) <= err
        print('Test 09:pass')

    def test_10(self):
        #(3x+2)^4
        print("****Unit Test 10********")
        fex0 = make_prod(make_const(3.0), make_pwr('x', 1.0))
        fex1 = make_plus(fex0, make_const(2.0))
        fex = make_pwr_expr(fex1, 4.0)
        print(fex)
        afex = antideriv(fex)
        assert not afex is None
        print(afex)
        afexf = tof(afex)
        err = 0.0001

        def gt(x):
            return (1.0 / 15) * ((3 * x + 2.0) ** 5)

        for i in range(1, 10): assert abs(afexf(i) - gt(i)) <= err
        fexf = tof(fex)
        assert not fexf is None
        fex2 = deriv(afex)
        assert not fex2 is None
        print(fex2)
        fex2f = tof(fex2)
        assert not fex2f is None
        for i in range(1, 1000):
            assert abs(fexf(i) - fex2f(i)) <= err
        print('Test 10:pass')


    if __name__ == "__main__":
        unittest.main()