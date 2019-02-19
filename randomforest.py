#!/usr/bin/env python3

import os
import sys
import re
import codecs
import numpy
import pandas
from nltk.parse import stanford
from nltk.tree import ParentedTree
import configparser
import dill as pickle
from sklearn.ensemble import RandomForestClassifier
from optparse import OptionParser
from collections import defaultdict
import CONLLParser
import CustomLabelEncoder
import utils

lexParser = None
pm = {}

def getSyntaxFeatures(ct):

    global pm
    fs = ' '.join([x for x in ct.fullSentence if not re.match('^\s+$', x)])
    fs = re.sub('\)', ']', re.sub('\(', '[', fs))

    if len(fs.split()) > 100:
        return None, None, None, None, None, None # let's not even try
    pt = None
    if fs in pm:
        pt = ParentedTree.convert(pm[fs])
    else:
        try:
            tree = lexParser.parse(fs.split())
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = t
                break
            pt = ParentedTree.convert(ptree)
            pm[fs] = pt
        except: # probably memory issue with the parser
            sys.stderr.write('Skipped during parsing...\n')
            return None, None, None, None, None, None
    try:
        node = pt.pos()[ct.stid-1]
        nodePosition = pt.leaf_treeposition(ct.stid-1)
        parent = pt[nodePosition[:-1]].parent()
        parentCategory = parent.label()
        
        ls = parent.left_sibling()
        lsCat = False if not ls else ls.label()
        rs = parent.right_sibling()
        rsCat = False if not rs else rs.label()
        rsContainsVP = False
        if rs:
            if list(rs.subtrees(filter=lambda x: x.label()=='VP')):
                rsContainsVP = True
        rootRoute = utils.getPathToRoot(parent, [])
        cRoute = utils.compressRoute([x for x in rootRoute])
        return parentCategory, lsCat, rsCat, rsContainsVP, rootRoute, cRoute
    except IndexError:
        sys.stderr.write('Skipping due to indexerror...\n')
        return None, None, None, None, None, None        


def getTokenIdDict(conlltokens):

    sidtid2ct = defaultdict(CONLLParser.CustomCONLLToken)
    for ct in conlltokens:
        sidtid2ct[(ct.sid, ct.stid)] = ct
    return sidtid2ct

def getFeatureMatrix(conlltokens, parsingFlag):

    sidtid2ct = getTokenIdDict(conlltokens)
    matrix = []
    for cti, ct in enumerate(conlltokens):
        if cti % 1000 == 0:
            sys.stderr.write('INFO: Processing %i of %i...\n' % (cti, len(conlltokens)))
        row = []
        word = ct.token
        next_Word = None
        if (ct.sid, ct.stid+1) in sidtid2ct:
            next_word = sidtid2ct[(ct.sid, ct.stid+1)].token
        prev_word = None
        if (ct.sid, ct.stid-1) in sidtid2ct:
            prev_word = sidtid2ct[(ct.sid, ct.stid-1)].token
        last = False
        if ct.stid == len(ct.fullSentence):
            last = True
        dist2par = 0
        if isinstance(ct.head, int):
            dist2par = ct.stid - ct.head
        parent_func = None
        if (ct.sid, ct.head) in sidtid2ct:
            parent_func = sidtid2ct[(ct.sid, ct.head)].deprel
        next_pos = None
        next_pos_coarse = None
        if (ct.sid, ct.stid+1) in sidtid2ct:
            next_pos = sidtid2ct[(ct.sid, ct.stid+1)].pos_fine
            next_pos_coarse = sidtid2ct[(ct.sid, ct.stid+1)].pos_coarse
        func = ct.deprel
        prev_func = None
        if (ct.sid, ct.stid-1) in sidtid2ct:
            prev_func = sidtid2ct[(ct.sid, ct.stid-1)].deprel
        parent_pos = None
        parent_pos_coarse = None
        if (ct.sid, ct.head) in sidtid2ct:
            parent_pos = sidtid2ct[(ct.sid, ct.head)].pos_fine
            parent_pos_coarse = sidtid2ct[(ct.sid, ct.head)].pos_coarse
        prev_pos = None
        prev_pos_coarse = None
        if (ct.sid, ct.stid-1) in sidtid2ct:
            prev_pos = sidtid2ct[(ct.sid, ct.stid-1)].pos_fine
            prev_pos_coarse = sidtid2ct[(ct.sid, ct.stid-1)].pos_coarse
        pos = ct.pos_fine
        poscoarse = ct.pos_coarse
        next_upper = False
        if (ct.sid, ct.stid+1) in sidtid2ct:
            if sidtid2ct[(ct.sid, ct.stid+1)].token[0].isupper():
                next_upper = True
        parent_upper = False
        if (ct.sid, ct.head) in sidtid2ct:
            if sidtid2ct[(ct.sid, ct.head)].token[0].isupper():
                parent_upper = True
        prev_upper = False
        if (ct.sid, ct.stid-1) in sidtid2ct:
            if sidtid2ct[(ct.sid, ct.stid-1)].token[0].isupper():
                prev_upper = True
        word_upper = False
        if ct.token[0].isupper():
            word_upper = True

        row.append(word)
        row.append(next_word)
        row.append(prev_word)
        row.append(last)
        row.append(dist2par)
        row.append(parent_func)
        row.append(next_pos)
        row.append(next_pos_coarse)
        row.append(func)
        row.append(prev_func)
        row.append(parent_pos)
        row.append(parent_pos_coarse)
        row.append(prev_pos)
        row.append(prev_pos_coarse)
        row.append(pos)
        row.append(poscoarse)
        row.append(next_upper)
        row.append(parent_upper)
        row.append(prev_upper)
        row.append(word_upper)
    
        if parsingFlag:
            pcat, lscat, rscat, rscvp, rr, cr = getSyntaxFeatures(ct)
            row.append(str(pcat))
            row.append(str(lscat))
            row.append(str(rscat))
            row.append(str(rscvp))
            row.append(str(rr))
            row.append(str(cr))

        senlen = utils.categorizeLength(len(ct.fullSentence))
        row.append(senlen)
        row.append(ct.stid) # position in sentence)
        relative_position = 0
        if not len(ct.fullSentence) == 0:
            relative_position = utils.categorizeFractal(ct.stid / len(ct.fullSentence), 5)
        row.append(relative_position)

        vfbool = utils.verbFollowingBeforeNextPunctuation(ct, sidtid2ct)
        row.append(vfbool)

        label = ct.segmentStarter
        row.append(label)

        matrix.append(row)

    return matrix

                
def randomforest(le, train, test, headers):

    df = pandas.DataFrame(train, columns=headers)
    Y = df.class_label
    labels = list(set(Y))
    Y = numpy.array([labels.index(x) for x in Y])
    X = df.iloc[:,:len(headers)-1]
    X = numpy.array(X)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X,Y)
    
    #feature_importances = pandas.DataFrame(clf.feature_importances_, index = headers[:-1], columns=['importance']).sort_values('importance', ascending=False)
    #print('info gain:\n', feature_importances)

    testdf = pandas.DataFrame(test, columns=headers)
    Y_test = testdf.class_label
    testlabels = list(set(Y_test))
    Y_test = numpy.array([testlabels.index(x) for x in Y_test])
    X_test = testdf.iloc[:,:len(headers)-1]
    X_test = numpy.array(X_test)

    results = clf.predict(X_test)

    return results


def printConllOut(conllin, labels):

    outname = conllin + '.randomforest.predicted'
    conllout = codecs.open(outname, 'w')
    c = 0
    for line in codecs.open(conllin).readlines():
        if re.search('\t', line) and not line.startswith('#'):
            lastelem = line.strip().split('\t')[-1]
            if labels[c] == 1:
                if re.search('SpaceAfter', lastelem):
                    conllout.write('\t'.join(line.strip().split('\t')[:-1]) + '\tBeginSeg=Yes|%s\n' % lastelem.split('|')[-1])
                else:
                    conllout.write('\t'.join(line.strip().split('\t')[:-1]) + '\tBeginSeg=Yes\n')
            else:
                if re.search('SpaceAfter', lastelem) and not re.search('BeginSeg', lastelem):
                    conllout.write(line)
                else:
                    conllout.write('\t'.join(line.strip().split('\t')[:-1]) + '\t_\n')
            c += 1
        else:
            conllout.write(line)
    conllout.close()
    sys.stderr.write('INFO: output written to %s\n' % outname)


if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('--train', dest='trainconll', help='train file in CONLL format')
    parser.add_option('--dev', dest='devconll', help='dev file in CONLL format')
    parser.add_option('--test', dest='testconll', help='test file in CONLL format')
    parser.add_option('--noParsing', dest='noParsing', action="store_true", default=False, help='For the supported languages ([deu, eng, spa, fra, zho]), the stanford lexparser is used to retrieve additional syntax features. Specify this (boolean) option to turn this off. Will go significantly faster (if no pre-pickled parses are available) at the cost of accuracy.')

    
    options, args = parser.parse_args()

    if not options.trainconll or not options.devconll or not options.testconll:
        parser.print_help(sys.stderr)
        sys.exit(1)

    language = os.path.basename(options.trainconll)[:3]
    parsingFlag = False
    pname = None
    plocation = None

    if language in ['deu', 'eng', 'spa', 'fra', 'zho'] and not options.noParsing:
        parsingFlag = True
        config = configparser.ConfigParser()
        config.read('config.ini')
        os.environ['JAVAHOME'] = config['lexparser']['javahome']
        os.environ['STANFORD_PARSER'] = config['lexparser']['stanfordParser']
        os.environ['STANFORD_MODELS'] = config['lexparser']['stanfordModels']
        os.environ['CLASSPATH'] = config['lexparser']['path']
        #nltk.internals.config_java(options='-xmx4G')
        if language == 'deu':
            pname = 'german_sentences.pickle'
            lexParser = stanford.StanfordParser(model_path=config['lexparser']['germanModel'])
        elif language == 'eng':
            pname = 'english_sentences.pickle'
            lexParser = stanford.StanfordParser(model_path=config['lexparser']['englishModel'])
        elif language == 'spa':
            pname = 'spanish_sentences.pickle'
            lexParser = stanford.StanfordParser(model_path=config['lexparser']['spanishModel'])
        elif language == 'fra':
            pname = 'french_sentences.pickle'
            lexParser = stanford.StanfordParser(model_path=config['lexparser']['frenchModel'])
        elif language == 'zho':
            pname = 'chinese_sentences.pickle'
            lexParser = stanford.StanfordParser(model_path=config['lexparser']['chineseModel'])
        plocation = os.path.join(os.path.join(os.getcwd(), 'picklejars'), pname)
        if os.path.exists(plocation):
            pm = pickle.load(codecs.open(plocation, 'rb'))
        else:
            sys.stderr.write('INFO: Failed to find parse dictionary (%s), re-parsing, execution may take a while...\n' % plocation)
        

    traintokens = CONLLParser.parse(options.trainconll)
    devtokens = CONLLParser.parse(options.devconll)
    testtokens = CONLLParser.parse(options.testconll)

    traintokens.extend(devtokens) # throw train and dev together

    trainmatrix = getFeatureMatrix(traintokens, parsingFlag)
    testmatrix = getFeatureMatrix(testtokens, parsingFlag)

    if parsingFlag: # re-pickle for next time
        with codecs.open(plocation, 'wb') as handle:
            pickle.dump(pm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    le = CustomLabelEncoder.CustomLabelEncoder()
    train = le.encode(trainmatrix)
    test = le.encode(testmatrix)

    headers = ['word','next_word','prev_word','last',
               'dist2par','parent_func','next_func',
               'next_pos','next_pos_coarse','func',
               'prev_func', 'parent_pos', 'parent_pos_coarse',
               'prev_pos', 'prev_pos_coarse', 'pos',
               'pos_coarse', 'next_upper', 'parent_upper',
               'prev_upper', 'word_upper', 'position_in_sentence',
               'relative_position_in_sentence',
               'verbfollowingbeforenextpunctuation',
               'class_label']
    if parsingFlag:
        headers = ['word', 'next_word', 'prev_word', 'last',
                   'dist2par', 'parent_func', 'next_func',
                   'next_pos', 'next_pos_coarse', 'func',
                   'prev_func', 'parent_pos', 'parent_pos_coarse',
                   'prev_pos', 'prev_pos_coarse', 'pos',
                   'pos_coarse', 'next_upper', 'parent_upper',
                   'prev_upper', 'word_upper', 'parentCat',
                   'leftSiblingCat', 'rightSiblingCat',
                   'rightSiblingContainsVP', 'rootRoute',
                   'compressedroute', 'position_in_sentence',
                   'relative_position_in_sentence',
                   'verbfollowingbeforenextpunctuation',
                   'class_label']

    testlabels = randomforest(le, train, test, headers)

    printConllOut(options.testconll, testlabels)
