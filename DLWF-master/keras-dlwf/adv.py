import random

len = 5000
hgt = 3
e = 0.2
a = 0.8
g = 0.5

def transform(spl, pos, sign):
    if pos<0 or pos>=len:
        return

    if spl[pos] != [1, -1][sign]:
        spl.insert(pos, [1, -1][sign])

def branch(pos, lvl):
    if pos>=len+1 or lvl>=hgt:
        return []

    return [(0.0, branch(pos+step+1, lvl+1)), (0.0, branch(pos+step+1, lvl+1)) for step in range(len-pos+1)].append((0.0, branch(len, lvl+1)))

def update(model, spl, pos, root, sign, prev):
    if len(root)<=0:
        return

    transform(spl, pos-1, sign)
    cur = classify(model, spl)
    rwd = cur - prv

    if random.random() < e:
        slt = random.randint(0, len(root)-1)
    else:
        slt = [tpl[0] for tpl in root].index(max(root, key=lambda tpl: tpl[0]))

    root[slt][0] = (1-a) * root[slt][0] + a * (rwd + g * max(root[slt][1], key=lambda tpl: tpl[0]))
    update(spl, pos+slt/2+1, root[slt][1], slt%2, cur)

def obfs(model, spl, pos, root, sign):
    if len(root)<= 0:
        return classify(model, spl)

    transform(spl, pos-1, sign)

    slt = [tpl[0] for tpl in root].index(max(root, key=lambda tpl: tpl[0]))
    return test(spl, pos+slt/2+1, root[slt][1], slt%2)

def adv(model, train, test):
    q = branch(0, 0)

    for spl in train:
        update(model, spl, 0, q, 0, 0.0)

    rst = []

    for spl in test:
        rst.append(obfs(model, spl, 0, q, 0))

    print rst

    print min(rst)
    print max(rst)

    rst_bool = []

    for cfd in rst:
        rst_bool.append(True if cfd>0.5 else False)

    print rst_bool

    print rst_bool.count(True) / len(rst_bool)
    print rst_bool.count(False) / len(rst_bool)
