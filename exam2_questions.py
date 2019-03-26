from const import const
from maker import make_const, make_pwr, make_pwr_expr, make_plus, make_prod, make_quot, make_e_expr, make_ln, make_absv
from tof import tof
from riemann import riemann_approx, riemann_approx_with_gt, plot_riemann_error
from deriv import deriv
from antideriv import antideriv, antiderivdef
from defintegralapprox import midpoint_rule, trapezoidal_rule, simpson_rule
import math
import numpy as np
import matplotlib.pyplot as plt
########PROBLEM 1- DEMAND ELASTICITY###############

def demand_elasticity(p):
    assert isinstance(p, const)
    #100-2p
    demand_eq = make_plus(make_const(100), make_prod(make_const(-2.0), make_pwr('p', 1.0)))
    fx_drv = tof(deriv(demand_eq))
    fx = tof(demand_eq)

    num = (-1.0*p.get_val()) * fx_drv(p.get_val())
    denom = fx(p.get_val())

    return num / denom

##########PROBLEM 2-LOGISTIC GROWTH MODELS###########
#logistic growth  y = M/(1+Be^(-MKt))

def spread_of_disease_model(p, t0, p0, t1, p1):
    assert isinstance(p, const) and isinstance(t0, const)
    assert isinstance(p0, const) and isinstance(t1, const)

    #find B first
    B = const(((p.get_val()/p0.get_val())-1.0)/math.e**(t0.get_val()))

    x = const(((p.get_val()/p1.get_val())-1.0)/B.get_val())
    #find k
    k = const(math.log(x.get_val()) / (-1.0*t1.get_val()*p.get_val()))

    exponent = const(-1.0*p.get_val()*k.get_val())

    bottom = make_plus(make_const(1.0), make_prod(B, make_e_expr(make_prod(exponent, make_pwr('t', 1.0)))))
    return make_quot(p, bottom)


def plot_spread_of_disease(p, t0, p0, t1, p1, tl, tu):
    assert isinstance(p, const) and isinstance(t0, const)
    assert isinstance(p0, const) and isinstance(t1, const)
    rt = spread_of_disease_model(p, t0, p0, t1, p1)
    rt_tof = tof(rt)
    derv_rt = deriv(rt)

    derv_tof = tof(derv_rt)

    xvals = np.linspace(tl.get_val(), tu.get_val(), 10000)
    yvals1 = np.array([rt_tof(x) for x in xvals])

    xvals2 = np.linspace(tl.get_val(), tu.get_val(), 10000)
    yvals2 = np.array([derv_tof(x) for x in xvals])

    fig1 = plt.figure(1)
    fig1.suptitle('Spread of Disease')
    plt.xlabel('t')
    plt.ylabel('sdf and dsdf')
    plt.ylim([0, 100000])
    plt.xlim([tl.get_val(), tu.get_val()])
    plt.grid()
    plt.plot(xvals, yvals1, label='sdf', c='r')
    plt.plot(xvals2, yvals2, label='dsdf', c='b')

    plt.legend(loc='best')
    plt.show()

def spread_of_news_model(p, k):
    assert isinstance(p, const) and isinstance(k, const)

    expon = const(-1.0*k.get_val())
    return make_prod(p, make_plus(const(1.0), make_prod(const(-1.0), make_e_expr(make_prod(expon, make_pwr('t', 1.0))))))


def plot_spread_of_news(p, k, tl, tu):
    assert isinstance(p, const) and isinstance(k, const)
    assert isinstance(tl, const) and isinstance(tu, const)
    nm = spread_of_news_model(p, k)
    nm_tof = tof(nm)
    deriv_nm = deriv(nm)
    deriv_nm_tof = tof(deriv_nm)

    xvals = np.linspace(tl.get_val(), tu.get_val(), 10000)
    yvals1 = np.array([nm_tof(x) for x in xvals])

    xvals2 = np.linspace(tl.get_val(), tu.get_val(), 10000)
    yvals2 = np.array([deriv_nm_tof(x) for x in xvals])

    fig1 = plt.figure(1)
    fig1.suptitle('Spread of News')
    plt.xlabel('t')
    plt.ylabel('snf and dsnf')
    plt.ylim([-2000, 52000])
    plt.xlim([tl.get_val(), tu.get_val()])
    plt.grid()
    plt.plot(xvals, yvals1, label='snf', c='r')
    plt.plot(xvals2, yvals2, label='dsnf', c='b')

    plt.legend(loc='best')
    plt.show()
##########PROBLEM 3- ANTIDERIV ####################



########PROBLEM 4- IMAGES##########################
#In another file


#########PROBLEM 5- NET CHANGE####################
def net_change(mrexpr, pl1, pl2):
    assert isinstance(pl1, const)
    assert isinstance(pl2, const)

    #F(b) - F(a)
    a = pl1
    b = pl2
    F = tof(antideriv(mrexpr))

    return F(b.get_val()) - F(a.get_val())


############PROBLEM 6 - CONSUMER SURPLUS##########
    #f(x) -B |0 to A
def consumer_surplus(dexpr, a):
    assert isinstance(a, const)

    B = const(-1*tof(dexpr)(a.get_val()))

    f = make_plus(dexpr, B)
    surplus = tof(antideriv(f))
    return surplus(a.get_val()) - surplus(0)



#########PROBLEM 7 - APPROXIMATING DEFINITE INTEGRALS ###########



##########PROBLEM 8 - BELL CURVE ######################
##problems with using gaussian pdf it uses a lambda expression but all of the
## other approximation tools using expressions

def midpoint_rule_bell(fexpr, a, b, n):
    assert isinstance(a, const)
    assert isinstance(b, const)
    assert isinstance(n, const)

    area = 0
    # fex_tof = tof(fexpr)
    partition = (b.get_val() - a.get_val())/ n.get_val()

    a = int(a.get_val())
    b = int(b.get_val())

    for i in np.arange(a, b, partition):
        mid = i + (partition / 2)
        area += fexpr(mid) * partition

    return const(area)

def trapezoidal_rule_bell(fexpr, a, b, n):
    assert isinstance(a, const)
    assert isinstance(b, const)
    assert isinstance(n, const)
    area = 0
    # fex_tof = tof(fexpr)
    partition = (b.get_val() - a.get_val())/ n.get_val()

    a = int(a.get_val())
    b = int(b.get_val())

    for i in np.arange(a, b, partition):
        area += partition * ((fexpr(i)+fexpr(i+partition))/2)

    return const(area)

def simpson_rule(fexpr, a, b, n):
    assert isinstance(a, const)
    assert isinstance(b, const)
    assert isinstance(n, const)

    #Simpson = (2M+T)/3
    T = trapezoidal_rule_bell(fexpr, a, b, n)
    M = midpoint_rule_bell(fexpr, a, b, n)

    return const((2*M.get_val() + T.get_val())/3)

def gaussian_pdf(x, sigma=1, mu=0):
    a = 1.0/(sigma*math.sqrt(2*math.pi))
    b = math.e**(-0.5(((x - mu)/sigma)**2))
    return a*b

def bell_curve_iq_approx(a, b):
    assert isinstance(a, const)
    assert isinstance(b, const)

    iqc = lambda x: gaussian_pdf(x, sigma=16.0, mu=100)
    print(simpson_rule(iqc, a, b, const(6)))


###########PROBLEM 9 - LEAST SQUARES#####################




###########PROBLEM 10  -TAYLOR POLYNOMIALS ################
def taylor_poly(fexpr, a, n):
    assert isinstance(a, const)
    assert isinstance(n, const)

    tof_exp = tof(fexpr)
    ex = fexpr
    # print(ex)
    # tof_exp(a.get_val()) + (deriv(fexpr)/math.factorial(n.get_val()))*(x - a.get_val)^n
    result = const(tof_exp(a.get_val()))

    for i in range(1, int(n.get_val())):
        drv = deriv(ex)
        drv_tof = tof(drv)

        inside = const(drv_tof(a.get_val())/math.factorial(i))
        x = make_plus(make_pwr('x', 1.0), make_prod(const(-1.0), a))

        pw = make_pwr_expr(x, i)

        result = make_plus(result, make_prod(inside, pw))
        ex = drv

    return result

