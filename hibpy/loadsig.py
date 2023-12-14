# -*- coding: utf-8 -*-
"""
Load Signal with help of SignalLoader server (sigload.exe)
SignalLoader should be started manually only once, 
after that it will be found automatically. 

Signal names are fully compatible with SigViewer
  including options  {avg101, x1.5, rar10, ...}
            prefixes HIBP::, FILE:: etc
            substitutions %SHOT%, %TJIIEBEAMII% etc

@author: reonid


Exports:  
    loadSignal(shot, sig, **kwargs)
    loadInfo(shot, cmd)
    
"""

import struct
import ctypes

import win32api, win32con, win32gui
import mmap

import numpy as np
import matplotlib.pylab as plt

from .winutils import summonServerWin

#class SignalError(Exception): pass
class SignalLoaderError(Exception): pass
class SignalsAreNotCompatible(Exception): pass 
    
TYPE_DOUBLE = 3
TYPE_SINGLE = 5
TYPE_INTEGER = 8
TYPE_SMALLINT = 11

SIG_MMAP_NAME = 'REONID@SIGNALDATA'
SIG_MMAP_SIGNATURE = 0x010AD516
WM_USER = 0x0400
WM_SIGLOADER_CMD = WM_USER + 247
WM_MY_SERVER_PING = WM_USER + 1973
SIGNAME_MAXLEN = 512

CMD_CHANGE_DEVICE = 22
CMD_CHANGE_TYPEID = 33
CMD_CHANGE_TRANSPORT_MODE = 44
CMD_CLOSE_MMAP = 55
CMD_SHOW_SERVER = 77

CMD_T10_EBEAM = 110
CMD_TJII_EBEAM = 111
CMD_HIBPII_EBEAM = 112


DEVICE_TJII = 1002
DEVICE_T10 = 1010
DEVICE_COMPASS = 1014
DEVICE_LIVEN = 1017


#------------------------------------------------------------------------------

def readSingle(file): 
    b = file.read(4)
    tpl = struct.unpack("<f", b)
    return tpl[0]

def readDouble(file): 
    b = file.read(8)
    tpl = struct.unpack("<d", b)
    return tpl[0]

def readLongInt(file): 
    b = file.read(4)
    return int.from_bytes(b, byteorder='little', signed=True)

def readFixedStr(file, L): 
    b = file.read(L)
    s = b.decode('cp1252')
    return s.split('\0')[0]

#------------------------------------------------------------------------------

class COPYDATASTRUCT_PCHAR(ctypes.Structure):
    _fields_ = [
        ('dwData', ctypes.wintypes.LPARAM),
        ('cbData', ctypes.wintypes.DWORD),
        ('lpData', ctypes.c_char_p) #c_wchar_p
        #formally lpData is c_void_p, but we do it this way for convenience
    ]

#------------------------------------------------------------------------------

def __sendCmd(win, cmd, param): 
    return win32api.SendMessage(win, WM_SIGLOADER_CMD, cmd, param)

def sendLoaderCmd(win=None, mode=None, device=None, dtype=None, 
                  close_mmap=False, show_server=None, ping=None, 
                  shot=None, hibp=None, hibp_param=None):

    if win is None: 
        win = summonServerWin('Reonid', 'SignalLoader')
       
    if   mode is None:  pass
    elif mode == 'file':  __sendCmd(win, CMD_CHANGE_TRANSPORT_MODE, 1)  
    elif mode == 'mmap':  __sendCmd(win, CMD_CHANGE_TRANSPORT_MODE, 0) 
    else: raise SignalLoaderError("Invalid mode: %s" % str(mode))
        
    if   device is not None: 
        device = device.lower()
    
    if   device is None: pass
    elif device == 't10':     __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_T10)   
    elif device == 't-10':    __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_T10)   
    elif device == 'tjii':    __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_TJII)  
    elif device == 'tj-ii':   __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_TJII)  
    elif device == 'tj2':     __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_TJII)  
    elif device == 'tj-2':    __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_TJII)  
    elif device == 'compass': __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_COMPASS)  
    elif device == 'liven':   __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_LIVEN)  
    else: raise SignalLoaderError("Invalid device: %s" % str(device))

    if   dtype is None:  pass
    elif dtype ==   'float32':  __sendCmd(win, CMD_CHANGE_TYPEID, TYPE_SINGLE)  
    elif dtype == np.float32:   __sendCmd(win, CMD_CHANGE_TYPEID, TYPE_SINGLE)  
    elif dtype ==   'float64':  __sendCmd(win, CMD_CHANGE_TYPEID, TYPE_DOUBLE) 
    elif dtype == np.float64:   __sendCmd(win, CMD_CHANGE_TYPEID, TYPE_DOUBLE) 
    else: raise SignalLoaderError("Unsupported dtype: %s" % str(dtype))
    
    if close_mmap: win32api.SendMessage(win, WM_SIGLOADER_CMD, CMD_CLOSE_MMAP, 0) 
    
    if   show_server is None:  pass
    elif show_server is True:  __sendCmd(win, CMD_SHOW_SERVER, 1)
    elif show_server is False: __sendCmd(win, CMD_SHOW_SERVER, 0)
    else: pass

    if hibp_param == 'Ebeam': 
        if   hibp == 0: return __sendCmd(win, CMD_T10_EBEAM, shot)
        elif hibp == 1: return __sendCmd(win, CMD_TJII_EBEAM, shot)
        elif hibp == 2: return __sendCmd(win, CMD_HIBPII_EBEAM, shot)
        else: raise SignalLoaderError("Unsupported hibp id: %d" % hibp)

    if ping: 
        try: 
            #ans = win32api.SendMessage(win, WM_MY_SERVER_PING, 133, 204)
            ok, ans = win32gui.SendMessageTimeout(win, WM_MY_SERVER_PING, 133, 204, win32con.SMTO_ABORTIFHUNG, 5000)
        except: 
            return 0
        return ans 

def loadSignal(shot, sig, **kwargs): 
    win = summonServerWin('Reonid', 'SignalLoader')
    if win is None: raise SignalLoaderError('Cannot find SignalLoader. Please start it manually')
    
    cmd_str = sig + '\0' # just for the case
    cmd_str = cmd_str.encode('ascii')
    
    cds = COPYDATASTRUCT_PCHAR()
    cds.dwData = shot
    cds.cbData = ctypes.sizeof(ctypes.create_string_buffer(cmd_str))
    cds.lpData = ctypes.c_char_p(cmd_str)

    #sendLoaderCmd(win, mode='mmap', device='tjii', dtype='float32')
    sendLoaderCmd(win, **kwargs)
    
    data_length = win32api.SendMessage(win, win32con.WM_COPYDATA, 0, ctypes.addressof(cds))  
    
    #if data_length == 0: return None
    if data_length == 0: raise SignalLoaderError('Cannot load signal #%d %s' % (shot, sig) )     
     
    with mmap.mmap(-1, data_length, tagname=SIG_MMAP_NAME) as mm:  
        _ = readLongInt(mm)  #_signature = readLongInt(mm)
        _ = readLongInt(mm)  #_total_size = readLongInt(mm)
        _ = readLongInt(mm)  #_shot = readLongInt(mm)
        signame = readFixedStr(mm, 512)
        L = readLongInt(mm)
        data_type = readLongInt(mm)
        
        if data_type == TYPE_SINGLE: 
            t0 = readSingle(mm)
            t1 = readSingle(mm)
            xdata = np.linspace(t0, t1, L, dtype = np.float32)          
        
            buffer = mm.read(L*4)
            ydata = np.frombuffer(buffer, np.float32, L)

        elif data_type == TYPE_DOUBLE: 
            t0 = readDouble(mm)
            t1 = readDouble(mm)
            xdata = np.linspace(t0, t1, L, dtype = np.float64)          
        
            buffer = mm.read(L*8)
            ydata = np.frombuffer(buffer, np.float64, L)
        else:
            raise SignalLoaderError('Invalid data type #%d %s' % (shot, sig) ) 

     
    sendLoaderCmd(win, close_mmap=True)
    
    return xdata, ydata, signame 

def loadInfo(shot, cmd): 
    win = summonServerWin('Reonid', 'SignalLoader')
    if win is None: raise SignalLoaderError('Cannot find SignalLoader. Please start it manually')
    
    cmd_str = cmd + '\0' # just for the case
    cmd_str = cmd_str.encode('ascii')
    
    cds = COPYDATASTRUCT_PCHAR()
    cds.dwData = shot
    cds.cbData = ctypes.sizeof(ctypes.create_string_buffer(cmd_str))
    cds.lpData = ctypes.c_char_p(cmd_str)

    #sendLoaderCmd(win, **kwargs)
    
    data_length = win32api.SendMessage(win, win32con.WM_COPYDATA, 0, ctypes.addressof(cds))  

    if data_length == 0: raise SignalLoaderError('Cannot load info %s' % (cmd) )     
     
    with mmap.mmap(-1, data_length, tagname=SIG_MMAP_NAME) as mm:  
        _ = readLongInt(mm)  #_signature = readLongInt(mm)
        _ = readLongInt(mm)  #_total_size = readLongInt(mm)
        txtLen = readLongInt(mm)  
        _ = readFixedStr(mm, 512) #signame = readFixedStr(mm, 512)
        buffer = mm.read(txtLen)
        txt = buffer.decode('cp1252')
        txt = txt.split('\0')[0]
     
    sendLoaderCmd(win, close_mmap=True)
    return txt 

#------------------------------------------------------------------------------
    
if __name__ == '__main__': 
    sendLoaderCmd(device='tjii', mode='mmap', dtype='float32')
    sendLoaderCmd(show_server=False)
    
    #sig, * = loadSignal(44381, "HIBPII::Itot{slit3, E=%EII%}", dtype='float64')
    sigx, sigy, sname = loadSignal(44381, "HIBPII::Itot{slit3, E=%TJIIEBEAMII%}", dtype='float64')
    plt.plot(sigx, sigy)
    
    E = sendLoaderCmd(shot=49858, hibp_param='Ebeam', hibp=2)
    print(E)

    txt = loadInfo(49842, '?DETAILS::ECRH1 angle 2')
    #txt = loadInfo(44381, '?DETAILS::All')
    #txt = loadInfo(44381, '?SHOTLIST::12, 13, 21..25')

    