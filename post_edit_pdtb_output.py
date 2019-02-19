from optparse import OptionParser
import re
import os
import sys
import codecs

def fix(fh):

    outf = os.path.abspath(fh) + '.fixed'
    oh = codecs.open(outf, 'w')
    lastlinemarked = False
    lines = codecs.open(fh).readlines()
    for i, line in enumerate(lines):
        newline = line
        if re.search('\t', line) and not line.startswith('#'):
            lastelem = line.strip().split('\t')[-1]
            if re.search('BeginSeg=Yes', lastelem):
                if lastlinemarked and not lines[i].split('\t')[0] == lines[i-2].split('\t')[0]:
                    newline = re.sub('BeginSeg=Yes', 'Seg=I-Conn', line)
                    lastlinemarked = True
                else:
                    newline = re.sub('BeginSeg=Yes', 'Seg=B-Conn', line)
                    lastlinemarked = True
            else:
                lastlinemarked = False
        else:
            laslinemarked = False
        oh.write(newline)
    oh.close()

if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('--test', dest='testconll', help='test file in CONLL format')
    options, args = parser.parse_args()

    if not options.testconll:
        parser.print_help(sys.stderr)
        sys.exit(1)

    fix(options.testconll)
