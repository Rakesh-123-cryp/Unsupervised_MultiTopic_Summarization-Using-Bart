import pandas as pd
import numpy as np

def remove_splchars(text,window_size=5):
    spl_chars = ["/", ":", "@", "*", "[", "]", "(", ")", "+", "-", "=", "|"]
    res = ""
    for i in text:
        if i in spl_chars:
            continue
        else:
            res += i
    
    check = res.split(".")
    temp = []
    ind=0
    while(ind<len(check)):
        seq = check[ind]
        
        if len(seq.split())<window_size:
            pass
        else:
            temp.append(seq)
        ind+=1
            
    return ".".join(temp)  
    # return res

def clean(text, window_size=5):
    check = text.split(".")
    temp = check
    ind = 0
    while(ind==0):
        seq = check[ind]
        if len(seq.split())<window_size:
            check.remove(seq)
        else:
            ind+=1
    return ".".join(check)