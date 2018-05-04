import random

length = 2 # Length of +1,-1 array considered by classifier
hgt = 2 # Maximum number of mutations, limits training time
e = 0.2 # Q-learning exploration factor
a = 0.8 # Q-learning learning rate
g = 0.5 # Q-learning decay factor

# Pads sample with either +1 (sign 0) or -1 (sign 1) at position
def transform(spl, pos, sign):
    if pos<0 or pos>=length:
        return

    # Only pad if position is not sign
    if spl[pos] != [1, -1][sign]:
        spl.insert(pos, [1, -1][sign])

# Recursively build Q-tree based on position in the array
def branch(pos, lvl):
    if pos>=length+1 or lvl>=hgt:
        return []

    # Extend 2 branches (+1, -1) for every position remaining that is available for padding
    # Last branch to catch end state (stop mutating)
    l = [(0.0, branch(pos+step/2+1, lvl+1)) for step in 2*range(length-pos+1)]
    l.append((0.0, branch(length, lvl+1)))
    return l

# Recursively train Q-tree with sample
def update(model, spl, pos, root, sign, prev):
    print(root)
    print(len(root))
    if len(root)<=0:
        return

    transform(spl, pos-1, sign)
    # Current confidence level of classifier
    #cur = classify(model, spl)
    cur = model.predict(spl[0:3000].reshape((1,3000,1)))[0,1]
    # Reward is the level of increase in confidence level
    rwd = cur - prev

    # Choose which branch to take at random with exploration factor probability
    if random.random() < e:
        slt = random.randint(0, len(root)-1)
    # Choose branch that has the maximum Q-value
    else:
        slt = [tpl[0] for tpl in root].index(max(root, key=lambda tpl: tpl[0])[0])

    # Update Q-value
    root[slt] = ((1-a) * root[slt][0] + a * (rwd + g * max(root[slt][1], key=lambda tpl: tpl[0])[0]), root[slt][1])
    # Explore selected branch
    update(model,spl, pos+slt/2+1, root[slt][1], slt%2, cur)

# Test Q-tree by obfuscating samples and evaluating their effectiveness
def obfs(model, spl, pos, root, sign):
    if len(root)<= 0:
        #return classify(model, spl)
        return model.predict(spl[0:3000].reshape((1,3000,1)))[0,1]

    transform(spl, pos-1, sign)

    # Choose branch that has maximum Q-value
    slt = [tpl[0] for tpl in root].index(max(root, key=lambda tpl: tpl[0])[0])
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

    print float(rst_bool.count(True)) / len(rst_bool) # False positive (sample not site, labelled as site)
    print float(rst_bool.count(False)) / len(rst_bool) # True negative (sample not site, labelled not site)
