'''
Author: silentchord 3334228261@qq.com
Date: 2022-10-05 11:10:23
LastEditors: silentchord 3334228261@qq.com
LastEditTime: 2022-10-05 11:10:56
FilePath: \MAST_sparse_based_on_max\models\global_vars.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

def _init():#初始化
    global _global_dict
    _global_dict = {}
 
 
def set_value(key,value):
    """ 定义一个全局变量 """
    _global_dict[key] = value
 
 
def get_value(key,defValue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defValue