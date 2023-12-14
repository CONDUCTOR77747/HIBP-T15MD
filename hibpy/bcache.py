# -*- coding: utf-8 -*-
"""
Load Signals from *.cache files 
Only binary formats (including zip-format) 

Load cached Thomson Scattering profiles (text format)

@author: reonid

Exports:  
    loadCachedSignal(shot, signalName)
    loadThomsonSignal(shot, signalName)
    loadCarpetFile(fileName)


"""

#import io as io
#import binascii
import zlib  
import numpy as np
import struct
import configparser as cfg
import os
import winreg 
import glob

from .winutils import WinRegKey


TYPE_DOUBLE = 3
TYPE_SINGLE = 5
TYPE_INTEGER = 8
TYPE_SMALLINT = 11

ZIP_BUF_SIGNATURE = 2123456789  
ZIP314_BUF_SIGNATURE = 314159265

ARCHIVE_CACHE_PATH_COUNT = 30

class SignalCacheError(Exception): pass

def readItem(cfgParser, section, item):
    try:
        result = cfgParser.get(section, item)
    except:
        result = ''

    # inline_comment_prefixes doesn't work if 
    # there is no whitespace between value and ";"
    result = result.split(';')[0]
    return result
    
def readCachePaths(configFileName): 
    result = []
    rpccCfg = cfg.ConfigParser(comment_prefixes=('!', ';'), 
                               inline_comment_prefixes=(';',))

    rpccCfg.read(configFileName)
    
    # Special paths
    oldCachePath = readItem(rpccCfg, "Cache", "CachePath")
    binCachePath = readItem(rpccCfg, "Cache", "BinCachePath")
    zipCachePath = readItem(rpccCfg, "Cache", "ZipCachePath")
    correctedPath = readItem(rpccCfg, "Cache", "CorrectedPath")
    
    # Archive paths
    result.append(correctedPath) # !!! first
    result.append(oldCachePath)
    result.append(binCachePath)
    result.append(zipCachePath)
    
    for i in range(1, ARCHIVE_CACHE_PATH_COUNT+1): 
        pathEntry = "ArchiveCachePath" + str(i)
        try: 
            path = readItem(rpccCfg, "Cache", pathEntry) 
            if os.path.exists(path): 
                result.append(path)        
        except: 
            continue
     
    return result   
        
def readConfigItem(configFileName, section, name): 
    rpccCfg = cfg.ConfigParser(comment_prefixes=('!', ';'), 
                               inline_comment_prefixes=(';',))
    rpccCfg.read(configFileName)
    result = readItem(rpccCfg, section, name)
    return result   

def readThomsonPath(configFileName): 
    return readConfigItem(configFileName, "Cache", "ThomsonPath")

def findCacheFile(pathList, shot, signalName): 
    for path in pathList: 
        fname = path + "\\tj2_" + str(shot) + "_" + signalName + ".cache"        
        if os.path.exists(fname): 
            return (True, fname)
    return (False, "")

def parseCacheFileName(fileName): 
    '''
    device, shot, name = parseCacheFileName(cacheFile)   
    '''
    fname_wo_path = fileName.split('\\')[-1]
    fname_wo_ext = fname_wo_path.split('.')[0]
    parts = fname_wo_ext.split('_')
    return parts[0], parts[1], parts[2]


def findRpccWrapCfg(): 
    with WinRegKey(winreg.HKEY_CURRENT_USER, "Software\\Reonid\\Common") as key: 
        result = key.readValue("RpccWrapCfg")
        if (result is not None) and (os.path.exists(result)): 
            return result
        else:
            result = "c:\\rpccwrap.cfg"
            if os.path.exists(result): 
                return result
            else: 
                result = "c:\\rpccwrap\\rpccwrap.cfg"
                if os.path.exists(result): 
                    return result
                
    
    raise SignalCacheError("File rpccwrap.cfg not found")

# -----------------------------------------------------------------------------

def bread(file, tp):  # read binary data
    '''
    Read from binary file
    '''
    tp = tp.lower()
    if   tp == 'd':    return readDouble(file)
    elif tp == 'f64':  return readDouble(file)
    elif tp == 'f':    return readSingle(file)
    elif tp == 'f32':  return readSingle(file)
    elif tp == '_s':   return readMyStr(file)     #
    elif tp == '_str': return readMyStr(file)     #
    elif tp == 'b':    return readByte(file)
    elif tp == 'byte': return readByte(file)
    elif tp == 'bool': return readBool(file)
    elif tp == '_buf': return readMyBuffer(file)  #
    elif tp == 'i16':  return readSmallInt(file)
    elif tp == 'i32':  return readLongInt(file)
    elif tp == 'i':    return readLongInt(file)
    else: raise Exception('unsupported type')     


def _readMyCompressedBuffer(file, compression_code): 
    if compression_code == 1: 
        L = readLongInt(file)
        elem_size = readLongInt(file)
        compressed_size = readLongInt(file)
        compressed_data = file.read(compressed_size)
        data = zlib.decompress(compressed_data)
        return L, elem_size, data

    elif compression_code == 314:
        L = readLongInt(file)
        elem_size = readLongInt(file)
        _ = readDouble(file)  # vmin = readDouble(file)
        _ = readDouble(file)  # vmax = readDouble(file)
        d = readDouble(file)
        k = readDouble(file)
        _ = readLongInt(file) # internal_compression = readLongInt(file)
        internal_code = readLongInt(file)
        
        L, _elem_size, int_bytes = readMyBuffer(file)
        if internal_code == 314: 
            int_array = np.frombuffer(int_bytes, np.int16, L)
        elif internal_code == 128314:
            int_array = np.frombuffer(int_bytes, np.int8, L)
        else: 
            raise Exception('readMyBuffer: Compression format not supported')
        
        k = 1/k # !!!
        dbl_array = k*(int_array + d)

        if elem_size == 8:
            assert dbl_array.dtype == np.float64
            array = dbl_array
        elif elem_size == 4:
            array = dbl_array.astype(np.float32)
        else: 
            raise Exception('readMyBuffer: Erroneous compression')
        
        data = bytes(array)
        return L, elem_size, data
    else: 
        raise Exception("readMyBuffer: Compression format not supported")

def readMyBuffer(file): 
    L = readLongInt(file)
    elem_size = readLongInt(file)

    if L == 0:
        return L, elem_size, None
    
    if elem_size == 0: 
        #raise SignalError("ZIP Buffers are not supported yet")
        if (L == ZIP_BUF_SIGNATURE): 
            return _readMyCompressedBuffer(file, 1)
        elif (L == ZIP314_BUF_SIGNATURE): 
            return _readMyCompressedBuffer(file, 314)
        else: 
            raise Exception("readMyBuffer: invalid compression format")
    else:              
        b = file.read(L*elem_size)

    return L, elem_size, b

#------------------------------------------------------------------------------
        
def readLongInt(file): 
    b = file.read(4)
    return int.from_bytes(b, byteorder='little', signed=True)

def readSmallInt(file): 
    b = file.read(2)
    return int.from_bytes(b, byteorder='little', signed=True)

def readSingle(file): 
    b = file.read(4)
    tpl = struct.unpack("<f", b)
    return tpl[0]

def readDouble(file): 
    b = file.read(8)
    tpl = struct.unpack("<d", b)
    return tpl[0]

def readByte(file): 
    bytes = file.read(1)
    return int.from_bytes(bytes, byteorder='little')

def readBool(file): 
    i = readByte(file)
    return i == 1

def readMyStr(file): 
    L = readLongInt(file)
    b = file.read(L)
    return b.decode('cp1252')

def readFixedStr(file, L): 
    b = file.read(L)
    s = b.decode('cp1252')
    return s.split('\0')[0]

def _fixedStrLen(s): 
    if s.startswith('s'): 
        sn = s.split('s')[1]
        return int(sn)
    else: 
        return None

def getTypeSize(typeid): 
    if typeid == TYPE_DOUBLE: 
        return 8
    elif (typeid == TYPE_SINGLE) or (typeid == TYPE_INTEGER): 
        return 4
    elif typeid == TYPE_SMALLINT:
        return 2

# -----------------------------------------------------------------------------

def loadCacheFile(fileName): 
    '''
    Read experimental signal from *.cache files
    '''    
    with open(fileName, "rb") as cache_file: 
        # Header ------------------------------------------------------
        signature = readLongInt(cache_file)
        assert signature == 180150000
        type_id = readSmallInt(cache_file)
        compression_code = readByte(cache_file)
        extended_header_cnt = readByte(cache_file)
        data_length = readLongInt(cache_file)
        time0  = readSingle(cache_file)
        time1  = readSingle(cache_file)
        offset = readSingle(cache_file)
        factor = readSingle(cache_file)
                        
        # Data --------------------------------------------------------
        x_array = np.linspace(time0, time1, data_length)

        # Decompression -----------------------------------------------  
        if compression_code == 0: 
            elem_size = getTypeSize(type_id)
            y_bytes = cache_file.read(data_length*elem_size)
                    
        # compressed_data = zlib.compress(data, 2)
        elif compression_code == 1: 
            compressed_size = readLongInt(cache_file)            
            _ = readLongInt(cache_file)                     # uncompressed_size
            _ = cache_file.read(4*extended_header_cnt - 8)  # reserved
            compressed_data = cache_file.read(compressed_size)
            y_bytes = zlib.decompress(compressed_data)

        # Y array -----------------------------------------------------
        if type_id == TYPE_INTEGER: 
            #for i in range(data_length): 
            #    b = y_bytes[i*4:(i+1)*4]
            #    n = int.from_bytes(b, byteorder='little', signed=True)
            #    y_array[i] = n * factor + offset
            int_array = np.frombuffer(y_bytes, np.int32, data_length)
            y_array = int_array * factor + offset
            
        elif type_id == TYPE_SMALLINT: 
            #y_array = np.empty(data_length, dtype=np.float64)
            #for i in range(data_length): 
            #    b = y_bytes[i*2:(i+1)*2]
            #    n = int.from_bytes(b, byteorder='little', signed=True)
            #    y_array[i] = n * factor + offset            
            y_array = np.frombuffer(y_bytes, np.int16, data_length)
            y_array = y_array * factor + offset
            
        elif type_id == TYPE_SINGLE: 
            #dt = np.dtype(np.float32)
            #dt = dt.newbyteorder('<')
            #y_array = np.frombuffer(y_bytes, dt, data_length)
            y_array = np.frombuffer(y_bytes, np.float32, data_length)

        elif type_id == TYPE_DOUBLE: 
            y_array = np.frombuffer(y_bytes, np.float64, data_length)

        return (x_array, y_array)
        



def loadCachedSignal(shot, signalName): 
    '''
    Read experimental signal from *.cache database
    '''
    (ok, cacheFile) = findCacheFile(cachePathList, shot, signalName)
    if ok: 
        #(xx, yy) = loadCacheFile(cacheFile)
        #return np.vstack((xx, yy))
        return loadCacheFile(cacheFile)
    else:
        raise SignalCacheError("Signal #%d %s not found in the cache" % (shot, signalName))


def loadThomsonSignal(shot, signalName): 
    if   signalName.lower() == 'ne': signalName = 'PerfilNe'
    elif signalName.lower() == 'te': signalName = 'PerfilTe'
    elif signalName.lower() == 'pe': signalName = 'PerfilPe'
            
    mask = thomsonPath + "\\%s_%d_*.dat" % (signalName, shot)   
    fnames = glob.glob(mask)

    if len(fnames) > 0: 
        #result = np.loadtxt(fnames[0], delimiter=' ', usecols=(0, 1), unpack=True)
        result = np.loadtxt(fnames[0])
        s = fnames[0].split('_')[-1]
        s = s.split('.')[0]
        time = int(s)
        return result.T, time
    else:
        raise SignalCacheError("Signal #%d %s not found in the cache" % (shot, signalName))
                          
                                                    
def loadCarpetFile(fileName): 
   '''
   Read *.carp files (result of spectrogram calculation) as 2D array

   xarr, yarr, zarr, maskarr, f_Nyquist = loadCarpetFile("test.carp")
   
   xarr, yarr: 1D arrays
   zarr, maskarr: 2D arrays
   '''
   with open(fileName, "rb") as carp_file: 
       xL, el, x_bytes = readMyBuffer(carp_file)
       yL, el, y_bytes = readMyBuffer(carp_file)
       zL, el, z_bytes = readMyBuffer(carp_file)
       mL, el, m_bytes = readMyBuffer(carp_file)

       _ = readDouble(carp_file)  # min_v = readDouble(carp_file)
       _ = readDouble(carp_file)  # max_v = readDouble(carp_file)     
       _ = readDouble(carp_file)  # mask_level = readDouble(carp_file)     
       
       _ = readMyStr(carp_file)   # title = readMyStr(carp_file)  

       _ = readLongInt(carp_file) # mask_color = readLongInt(carp_file)  # color
       _ = readByte(carp_file)    # show_cmap = readByte(carp_file)   # bool
       _ = readLongInt(carp_file) # orig_y_len = readLongInt(carp_file)
       f_Nyq = readDouble(carp_file)
       
       xx = np.frombuffer(x_bytes, np.float64, xL)
       yy = np.frombuffer(y_bytes, np.float64, yL)
       zz = np.frombuffer(z_bytes, np.float64, zL)
       zz = zz.reshape((yL, xL))
       zz = zz.T
       
       # read mask data
       if m_bytes is None: 
           mm = None
       else:     
           mm = np.frombuffer(m_bytes, np.float64, mL)
           mm = mm.reshape((yL, xL))
           mm = mm.T
           
           
       return xx, yy, zz, mm, f_Nyq
       
#------------------------------------------------------------------------------    

try: 
    rpccwrapCfg = findRpccWrapCfg()
    cachePathList = readCachePaths(rpccwrapCfg)
    thomsonPath = readThomsonPath(rpccwrapCfg)
    regimePath = readConfigItem(rpccwrapCfg, 'TJII', 'RegimePath')
except: 
    print('WARNING: cannot load rpccwrap.cfg')
