import random

len = 5000
hgt = 3
e = 0.2
a = 0.8
g = 0.5

def branch(pos, lvl):
    if pos>=len+1 or lvl>=hgt:
        return []

    return [(0.0, branch(pos+step+1, lvl+1)) for step in range(len-pos+1)]

def update(spl, pos, prv, root):
    if len(root)<=0:
        return

    spl = transform(spl, pos-1)
    cur = classify(spl)
    rwd = cur - prv

    if random.random() < e:
        slt = random.randint(0, len(root)-1)
    else:
        slt = [tpl[0] for tpl in root].index(max(root, key=lambda tpl: tpl[0]))

    root[slt][0] = (1-a) * root[slt][0] + a * (rwd + g * max(root[slt][1], key=lambda tpl: tpl[0]))
    update(spl, pos+slt+1, cur, root[slt][1])


q = branch(0, 0)

for spl in spls:
    update(spl, 0, 0.0, q)
