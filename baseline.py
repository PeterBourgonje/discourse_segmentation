from optparse import OptionParser
import sys
import codecs
import re
import CONLLParser
import randomforest
import string

def baseline(tokens, cf):

    labels = []
    for i, token in enumerate(tokens):
        if token.stid == 1:
            labels.append(1)
        elif cf and tokens[i-1].token ==',':
            labels.append(1)
        else:
            labels.append(0)
    return labels


def printConllOut(conllin, labels):

    outname = conllin + '.baseline.predicted'
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
    parser.add_option('--test', dest='testconll', help='test file in CONLL format')
    parser.add_option('-c', dest='comma', action='store_true', default=False, help='include this option to assume segment boundary after commas')
    options, args = parser.parse_args()

    if not options.testconll:
        parser.print_help(sys.stderr)
        sys.exit(1)

    testtokens = CONLLParser.parse(options.testconll)

    baselinelabels = baseline(testtokens, options.comma)
    
    printConllOut(options.testconll, baselinelabels)

    
