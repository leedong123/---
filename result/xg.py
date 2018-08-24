#!/usr/bin/env python
import sys
lines = []
with open('./test_out') as f:
    for i in f:
        lines.append(i.strip())


with open(sys.argv[1]) as f:
    for i,x in enumerate(f):
        label , t = x.strip().split('\t')
        label = label.strip()
        if label == '非来电' or label == '表扬及建议':
            label = '其他'
        if label != lines[i]:
            print (lines[i],x.strip())

        
