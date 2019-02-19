import os

def categorizeLength(sl):

    if sl < 5:
        return 5
    elif sl >= 5 and sl < 10:
        return 10
    elif sl >= 10 and sl < 15:
        return 15
    elif sl >= 15 and sl < 20:
        return 20
    elif sl >= 20 and sl < 25:
        return 25
    elif sl >= 25 and sl < 30:
        return 30
    elif sl >= 30 and sl < 35:
        return 35
    else:
        return 40

def verbFollowingBeforeNextPunctuation(ct, sidtid2ct):

    vfbool = False
    for i in range(ct.stid+1, len(ct.fullSentence)):
        try:
            ct_i = sidtid2ct[(ct.sid, i)]
            if ct_i.pos_coarse.lower().startswith('v'):
                vfbool = True
            if ct_i.token in string.punctuation:
                return vfbool
        except:
            continue
    return vfbool


def categorizeFractal(val, steps):

    stepvals = [x/100 for x in range(0, 1*100, int((1/5)*100))] # range steps did not seem to work with floats, hence workaround...
    for sv in stepvals:
        if val >= sv and val < sv + 1/steps:
            val = sv + 0.5 * (1/steps)
        else:
            val = val # it's 1 already anyway...
    
    return val



def getInputfiles(infolder):
    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist

# the NLTK parser crashed on round brackets.
def filterTokens(tokens):
    skipSet = ['(', ')']
    return [t for t in tokens if not t in skipSet]

def addAnnotationLayerToDict(flist, fdict, annname):
    for f in flist:
        basename = os.path.basename(f)
        fdict[basename][annname] = f
    return fdict

def getPathToRoot(ptree, route):
    if ptree.parent() == None:
        route.append(ptree.label())
        return route
    else:
        route.append(ptree.label())
        getPathToRoot(ptree.parent(), route)
    return route

def compressRoute(r): # filtering out adjacent identical tags
    delVal = "__DELETE__"
    for i in range(len(r)-1):
        if r[i] == r[i+1]:
            r[i+1] = delVal
    return [x for x in r if x != delVal]

def narrowMatches(matches, pcct, pccTokens, index):
    rightmatches = [x for x in matches if x[5] == pcct.token + '_' + pccTokens[index+1].token]
    if len(rightmatches) > 1:
        leftrightmatches = [x for x in rightmatches if x[2] == pccTokens[index-1].token + '_' + pcct.token]
        if len(leftrightmatches) > 1:
            sys.stderr.write('FATAL ERROR: Dying due to non-unique matches...\n')
            sys.exit(1)
        elif len(leftrightmatches) == 1:
            return leftrightmatches
        else:
            sys.stderr.write('FATAL ERROR: Could not find tree match at all...\n')
            sys.exit(1)
    else:
        return rightmatches

def mergePhrasalConnectives(l):
    l2 = []
    for t in l:
        nt = (t[0], [t[1]])
        l2.append(nt)
    l = l2
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if l[i][0] == l[j][0]:
                if l[i][1][-1] == l[j][1][0]-1:
                    l[i] = (l[i][0], sorted(l[i][1] + l[j][1]))
    ri = set()
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if l[i][0] == l[j][0]:
                if set(l[j][1]).issubset(set(l[i][1])):
                    ri.add(j)
    l2 = []
    for i in range(len(l)):
        if not i in ri:
            l2.append(l[i])
    l = l2
    return l

def getPostagFromTree(ptree, tokenIndex):
    # had I known it was this simple, wouldn't have to dedicate a function to it...
    return ptree.pos()[tokenIndex][1]

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

