import random

len = 5000 # Length of +1,-1 array considered by classifier
hgt = 3 # Maximum number of mutations, limits training time
e = 0.2 # Q-learning exploration factor
a = 0.8 # Q-learning learning rate
g = 0.5 # Q-learning decay factor

# Pads sample with either +1 (sign 0) or -1 (sign 1) at position
def transform(spl, pos, sign):
    if pos<0 or pos>=len:
        return

    # Only pad if position is not sign
    if spl[pos] != [1, -1][sign]:
        spl.insert(pos, [1, -1][sign])

# Recursively build Q-tree based on position in the array
def branch(pos, lvl):
    if pos>=len+1 or lvl>=hgt:
        return []

    # Extend 2 branches (+1, -1) for every position remaining that is available for padding
    # Last branch to catch end state (stop mutating)
    return [(0.0, branch(pos+step+1, lvl+1)), (0.0, branch(pos+step+1, lvl+1)) for step in range(len-pos+1)].append((0.0, branch(len, lvl+1)))

# Recursively train Q-tree with sample
def update(model, spl, pos, root, sign, prev):
    if len(root)<=0:
        return

    transform(spl, pos-1, sign)
    # Current confidence level of classifier
    cur = classify(model, spl)
    # Reward is the level of increase in confidence level
    rwd = cur - prv

    # Choose which branch to take at random with exploration factor probability
    if random.random() < e:
        slt = random.randint(0, len(root)-1)
    # Choose branch that has the maximum Q-value
    else:
        slt = [tpl[0] for tpl in root].index(max(root, key=lambda tpl: tpl[0]))

    # Update Q-value
    root[slt][0] = (1-a) * root[slt][0] + a * (rwd + g * max(root[slt][1], key=lambda tpl: tpl[0]))
    # Explore selected branch
    update(spl, pos+slt/2+1, root[slt][1], slt%2, cur)

# Test Q-tree by obfuscating samples and evaluating their effectiveness
def obfs(model, spl, pos, root, sign):
    if len(root)<= 0:
        return classify(model, spl)

    transform(spl, pos-1, sign)

    # Choose branch that has maximum Q-value
    slt = [tpl[0] for tpl in root].index(max(root, key=lambda tpl: tpl[0]))
    # Take branch
    return test(spl, pos+slt/2+1, root[slt][1], slt%2)

# Main function that should be called
def adv(model, train, test):
    # Build Q-tree
    q = branch(0, 0)

    # Train Q-tree
    for spl in train:
        update(model, spl, 0, q, 0, 0.0)

    # Confidence results, 0.0 not site - 1.0 is site
    rst = []

    # Test Q-tree
    for spl in test:
        rst.append(obfs(model, spl, 0, q, 0))

    print rst

    print min(rst)
    print max(rst)

    # Binary classifier decisions, False not site & True is site
    rst_bool = []

    for cfd in rst:
        rst_bool.append(True if cfd>0.5 else False)

    print rst_bool

    print rst_bool.count(True) / len(rst_bool) # False positive (sample not site, labelled as site)
    print rst_bool.count(False) / len(rst_bool) # True negative (sample not site, labelled not site)
