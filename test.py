from const import const
from maker import make_const, make_pwr, make_pwr_expr, make_plus, make_prod, make_quot, make_e_expr, make_ln, make_absv
from tof import tof
from riemann import riemann_approx, riemann_approx_with_gt, plot_riemann_error
from deriv import deriv
from antideriv import antideriv, antiderivdef
from defintegralapprox import midpoint_rule, trapezoidal_rule, simpson_rule
import unittest
import math

class Assign01UnitTests(unittest.TestCase):

    def test_01(self):
        #(3x^2 +e^x) with the midpoint riemann sum on a partition of 10 subintervals
        print("****Unit Test 01********")
        fex1 = make_prod(make_const(3.0), make_pwr('x', 2.0))
        fex = make_plus(fex1, make_e_expr(make_pwr('x', 1.0)))
        print(fex)
        err_list = riemann_approx_with_gt(fex,
                                          make_const(-1.0),
                                          make_const(1.0),
                                          make_const(4.35),
                                          make_const(10),
                                          pp=0)

        for n, err in err_list:
            print(n, err)
        print("Unit Test 01: pass")

    def test_02(self):
        #(3x^2 +e^x) with the left point riemann sum on a partition of 10 subintervals
        print("****Unit Test 02********")
        fex1 = make_prod(make_const(3.0), make_pwr('x', 2.0))
        fex = make_plus(fex1, make_e_expr(make_pwr('x', 1.0)))
        print(fex)
        err_list = riemann_approx_with_gt(fex,
                                          make_const(-1.0),
                                          make_const(1.0),
                                          make_const(4.35),
                                          make_const(10),
                                          pp=-1)

        for n, err in err_list:
            print(n, err)
        print("Unit Test 02: pass")

    def test_03(self):
        #(3x^2 +e^x) with the right point riemann sum on a partition of 10 subintervals
        print("****Unit Test 03********")
        fex1 = make_prod(make_const(3.0), make_pwr('x', 2.0))
        fex = make_plus(fex1, make_e_expr(make_pwr('x', 1.0)))
        print(fex)
        err_list = riemann_approx_with_gt(fex,
                                          make_const(-1.0),
                                          make_const(1.0),
                                          make_const(4.35),
                                          make_const(10),
                                          pp=+1)

        for n, err in err_list:
            print(n, err)
        print("Unit Test 03: pass")

    def test_04(self):
        # ln(x )with the middle point riemann sum on a partition of 100 subintervals
        print("****Unit Test 04********")

        fex = make_ln(make_pwr('x', 1.0))
        print(fex)
        err = 0.0001
        approx = riemann_approx(fex,
                                  make_const(1.0),
                                  make_const(2.0),
                                  make_const(100),
                                  pp=0)
        print(approx.get_val())
        assert  abs(approx.get_val() - 0.386296444432) <= err
        print("Unit Test 04: pass")

    def test_04_left_and_right_riemann(self):
        # x^2 with 10 subintervals
        print("****Unit Test 04********")

        fex = make_pwr('x', 2.0)
        print(fex)
        err = 0.0001
        approx = riemann_approx(fex,
                                  make_const(1.0),
                                  make_const(10.0),
                                  make_const(10),
                                  pp=-1)

        # assert  abs(approx.get_val() - 333) <= err
        print(approx.get_val())
        print("Unit Test 04: pass")

    def test_05(self):
        #x^2 + 5 with the midpoint rule on a partition of 250
        print("****Unit Test 05********")

        fexpr = make_plus(make_pwr('x', 2.0),
                                   make_const(5.0))
        a, b, n = make_const(0.0), make_const(4.0), make_const(250)
        approx = midpoint_rule(fexpr, a, b, n)
        print(approx)
        err = 0.001
        iv = antiderivdef(fexpr, a, b)
        print(iv)
        assert abs(approx.get_val() - iv.get_val()) <= err
        print("Unit Test 05: pass")

    def test_06(self):
        #x^2 + 5 with the trapezoidal rule on a partition of 350
        print("****Unit Test 06********")

        fexpr = make_plus(make_pwr('x', 2.0),
                                   make_const(5.0))
        a, b, n = make_const(0.0), make_const(4.0), make_const(350)
        approx = trapezoidal_rule(fexpr, a, b, n)
        print(approx)
        err = 0.001
        iv = antiderivdef(fexpr, a, b)
        print(iv)
        assert abs(approx.get_val() - iv.get_val()) <= err
        print("Unit Test 06: pass")

    def test_07(self):
        # x^2 + 5 with the simpson rule on a partition of 10 subintervals
        print("****Unit Test 07********")

        fexpr = make_plus(make_pwr('x', 2.0),
                                   make_const(5.0))
        a, b, n = make_const(0.0), make_const(4.0), make_const(10)
        approx = simpson_rule(fexpr, a, b, n)
        print(approx)
        err = 0.001
        iv = antiderivdef(fexpr, a, b)
        print(iv)
        assert abs(approx.get_val() - iv.get_val()) <= err
        print("Unit Test 07: pass")

    def test_08(self):
        # 2xe^(x^2) with the simpson rule on a partition of 100 subintervals
        print("****Unit Test 08********")

        fexpr = make_prod(make_prod(make_const(2.0),
                                  make_pwr('x', 1.0)),
                        make_e_expr(make_pwr('x', 2.0)))
        a, b, n = make_const(0.0), make_const(2.0), make_const(100)
        approx = simpson_rule(fexpr, a, b, n)
        print(approx)
        err = 0.001
        assert abs(approx.get_val() - 53.5981514272) <= err
        print("Unit Test 08: pass")

    def test_09(self):
        # (1+x^3)^0.5 with the simpson rule on a partition of 100 subintervals
        print("****Unit Test 09********")

        fex1 = make_plus(make_const(1.0),
                        make_pwr('x', 3.0))
        fex = make_pwr_expr(fex1, 0.5)
        a, b, n = make_const(0.0), make_const(2.0), make_const(100)
        approx = simpson_rule(fex, a, b, n)
        print(approx)
        err = 0.001
        assert abs(approx.get_val() - 3.24124) <= err
        print("Unit Test 09: pass")

    def test_10(self):
        print("****Unit Test 10********")
        fex = make_prod(make_const(3.0), make_pwr('x', 2.0))
        fex = make_plus(fex, make_e_expr(make_pwr('x', 1.0)))
        plot_riemann_error(fex, make_const(-1.0), make_const(1.0), make_const(4.35),
                                                    make_const(50))

    def test_11(self):
        print("****Unit Test 11********")
        fex = make_const(4.0)
        print(antiderivdef(fex, const(0.0), const(3.0)))

    if __name__ == "__main__":
        unittest.main()