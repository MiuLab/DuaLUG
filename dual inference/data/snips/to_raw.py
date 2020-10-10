import re
import sys
g = re.compile(r"(\S+):(\S+)")
f = open(sys.argv[1]).readlines()
li = []
for line in f:
    tmp = []
    for w, l in g.findall(line):
        tmp.append(w)
    li.append(' '.join(tmp))

#word_list = sorted(list(di.keys()), key=lambda x: -di[x])
fw = open(sys.argv[2], 'w')
fw.write('\n'.join(li))