# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:52:58 2020

@author: reonid

Objects and utils for data filtration

ITOT = mftr('itot')
ZD = mftr('zd')

itot = signal('HIBP::Itot1')
zd = signal('HIBP::Zd1')

filter = (ITOT > 0.5) & (ZD.inside(-0.8, 0.8))
mask = filter.calc_mask(globals())

"""

import sys
import numpy as np

from ..cienptas import x_op_y, ispair

_log_op_symbols = {'and': '&', 'or': '|', 'xor': '^', 'not': '~', 'inside': '[..)', 'outside': '.)[.'}
_log_op_names = {'&': 'and', '|': 'or', '^': 'xor', '~': 'not', '[..)': 'inside', '.)[.': 'outside'}
_log_operators =  ['<', '<=', '==', '!=', '>', '>=', 
                   '(..)', '[..]', '.)(.', '.][.', 
                   '(..]', '[..)', '.](.', '.)[.']

_inverse_cmp_op = {'<' :'>=',  '<=': '>',  '==': '!=', 
                   '!=': '==', '>' : '<=', '>=': '<', 
                   '(..)': '.][.', '[..]': '.)(.', '.)(.': '[..]', '.][.': '(..)', 
                   '(..]': '.](.', '[..)': '.)[.', '.](.': '(..]', '.)[.': '[..)'} 

#_log_operators = _inverse_cmp_op.keys

def inverse_cmp_op(op): 
    return _inverse_cmp_op[op]

def calc_cmp_mask(data, op, other): 
    '''
    mask = calc_cmp_mask(data, '>', 3.0)
    mask = calc_cmp_mask(data, '(..)', (3.0, 5.0))   
    
    '''
    assert other is not None
    assert data is not None 
    assert op in _log_operators # 'calc_cmp_mask: invalod operation symbol %s' % op'
    
    return x_op_y(data, op, other)


def get_test_data(arg, name_provider=None, L=None): 
    if arg is None: 
        return None
    if ispair(arg): 
        return arg
    elif isinstance(arg, FilterOperand): 
        sig = arg._get_data(name_provider)
    else: 
        sig = arg

    if isinstance(sig, str): 
        if isinstance(name_provider, dict): 
            sig = name_provider[sig]
        elif name_provider is not None: 
            sig = name_provider.__dict__[sig]
        else: 
            sig = globals()[sig]

    #if isinstance(sig, XYSignal): 
    if hasattr(sig, 'y') and hasattr(sig, '_setxy'):  
        data = sig.y
    else: 
        data = sig

    if L is not None: 
        assert L == len(data)

    return data

class BasicMaskFilter: 
    def __init__(self): 
        pass

    def calc_mask(self, name_provider=None): 
        print('BasicMaskFilter.calc_mask')
        return None

    def __and__(self, other): return FilterComposition(self, '&', other)
    def __or__(self, other):  return FilterComposition(self, '|', other)
    def __xor__(self, other): return FilterComposition(self, '^', other)
    def __invert__(self):     return FilterComposition(self, '~')


class FilterOperand: 
    def __init__(self, signal=None): 
        self.signal = signal

    def __lt__(self, other): return FilterCmp(self, '<',  other)
    def __le__(self, other): return FilterCmp(self, '<=', other)
    def __eq__(self, other): return FilterCmp(self, '==', other)
    def __ne__(self, other): return FilterCmp(self, '!=', other)
    def __gt__(self, other): return FilterCmp(self, '>',  other)
    def __ge__(self, other): return FilterCmp(self, '>=', other)
 
    def __add__(self, other): return OperandComposition(self, '+', other)
    def __sub__(self, other): return OperandComposition(self, '-', other)
    def __mul__(self, other): return OperandComposition(self, '*', other)
    def __truediv__(self, other): return OperandComposition(self, '/', other)
    def __floordiv__(self, other): return OperandComposition(self, '//', other)
    def __mod__(self, other): return OperandComposition(self, '%', other)
    def __pow__(self, other): return OperandComposition(self, '**', other)        
    def __abs__(self, other): return OperandComposition(self, abs, None)

    def __rshift__(self, other): return OperandComposition(self, other, None)    #  DENS >> np.sqrt

    def inside(self, v0, v1, strict=False): 
        return FilterCmp(self, '[..]' if strict else '(..)', (v0, v1))        

    def outside(self, v0, v1, strict=False): 
        return FilterCmp(self, '.][.' if strict else '.)(.', (v0, v1))
        
    def _get_data(self, name_provider): 
        return get_test_data(self.signal, name_provider)


class OperandComposition(FilterOperand): 
    def __init__(self, operand1, op, operand2): 
        assert isinstance(operand1, FilterOperand)
        
        #if operand2 is not None: 
        #    assert isinstance(operand2, FilterOperand)

        super().__init__()                
        self.operand1 = operand1
        self.operand2 = operand2
        self.op = op
        
    def _get_data(self, name_provider): 
        data1 = get_test_data(self.operand1, name_provider)          
        data2 = get_test_data(self.operand2, name_provider)        
        return x_op_y(data1, self.op, data2)

   
class FilterCmp(BasicMaskFilter):     
    def __init__(self, operand, op, other): 
        super().__init__()
        self.left_operand = operand
        self.op = op
        self.right_operand = other

    def calc_mask(self, name_provider=None): 
        data1 = get_test_data(self.left_operand, name_provider)
        data2 = get_test_data(self.right_operand, name_provider)
        return calc_cmp_mask(data1, self.op, data2)


#   '&'  '|'  '^'  '~'
class FilterComposition(BasicMaskFilter): 
    '''
    ftr = FilterComposition(ZD < -0.8, 'or', ZD > 0.8) 
    ftr = FilterComposition(ZD < -0.8, '|', ZD > 0.8) 
    ftr = (ZD < -0.8) | (ZD > 0.8)
    '''
    def __init__(self, filter1, log_op, filter2=None): 
        assert isinstance(filter1, BasicMaskFilter)
        if filter2 is not None: 
            assert isinstance(filter2, BasicMaskFilter)

        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2

        log_op = _log_op_symbols.get(log_op, log_op)
        assert log_op in ['&', '|', '^', '~']
        self.op = log_op

    def calc_mask(self, name_provider=None): 
        mask1 = self.filter1.calc_mask(name_provider)
        if self.op == '~':
            return ~mask1
        
        mask2 = self.filter2.calc_mask(name_provider)
        if   self.op == '&': return mask1 & mask2
        elif self.op == '|': return mask1 | mask2
        elif self.op == '^': return mask1 ^ mask2
        else: 
            raise Exception('Invalid operator')



class NotNan(BasicMaskFilter): 
    def __init__(self, operand): 
        super().__init__()
        self.operand = operand

    def calc_mask(self, name_provider=None): 
        data = get_test_data(self.operand, name_provider)   
        mask = ~np.isnan(data)
        return mask

class NotZero(BasicMaskFilter): 
    def __init__(self, operand): 
        super().__init__()
        self.operand = operand

    def calc_mask(self, name_provider=None): 
        data = get_test_data(self.operand, name_provider)   
        mask = data != 0.0
        return mask

def mftr(arg): 
    return FilterOperand(arg)

if __name__ == '__main__': 
    arr  = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    arr2 = np.array([0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2], dtype=float)
    
    ARR = mftr('arr')    
    ftr = ((ARR <= 8)&(ARR > 2)) | (ARR==0)
    #mask = (~ftr).calc_mask()
    mask = ftr.calc_mask()
    print(mask)
    
    ARR2 = mftr(arr2)    
    ftr = ((ARR2 <= 8)&(ARR2 > 2)) | (ARR2==0)
    mask = ftr.calc_mask()
    print(mask)
    
    ftr = (ARR2 >= ARR)
    mask = ftr.calc_mask()
    print(mask)

    SUM = ARR + ARR2
    ftr = ((SUM*0.5) > ARR)
    mask = ftr.calc_mask()
    print(mask)




