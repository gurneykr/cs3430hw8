#!/usr/bin/python

###############################################
# module: riemann.py
# Krista Gurney
# A01671888
###############################################

## modify these imports as you see fit.
import numpy as np
from const import const
from antideriv import antiderivdef
from tof import tof
import matplotlib.pyplot as plt
from maker import make_const

def riemann_approx(fexpr, a, b, n, pp=0):
  '''
  pp=0 - approximate with reimann midpoint
  pp=+1 - approximate with reimann right point
  pp=-1 - approximate with reiman left point
  '''
  assert isinstance(a, const)
  assert isinstance(b, const)
  assert isinstance(n, const)
  # your code here
  pass

def riemann_approx_with_gt(fexpr, a, b, gt, n_upper, pp=0):
  assert isinstance(a, const)
  assert isinstance(b, const)
  assert isinstance(gt, const)
  assert isinstance(n_upper, const)
  ## your code here
  pass

def plot_riemann_error(fexpr, a, b, gt, n):
  assert isinstance(a, const)
  assert isinstance(b, const)
  assert isinstance(gt, const)
  assert isinstance(n, const)
  # your code here



