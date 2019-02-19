import csv
import codecs
import re
from collections import defaultdict

class CustomCONLLToken:

    def __init__(self, uid, sid, stid, token, pos_coarse, pos_fine, segmentStarter, head, deprel):
        self.uid = int(uid)
        self.sid = int(sid)
        if re.match('\d+\-\d+', stid):
            self.stid = int(stid.split('-')[0]) # PT cases
        else:
            self.stid = int(stid)
        self.token = token
        self.pos_coarse = pos_coarse
        self.pos_fine = pos_fine
        try:
            self.head = int(head)
        except ValueError: # think this is handled later on then
            self.head = head
        self.deprel = deprel
        self.segmentStarter = segmentStarter

    def addFullSentence(self, sentaslist):
        self.fullSentence = sentaslist
    def addLang(self, lang):
        self.orig_lang = lang
        
    
def parse(fh, lang=False):
    
    tokens = []
    reader = csv.reader(codecs.open(fh), delimiter='\t', quotechar='\\')
    uid = 0
    sid = 1
    sentences = []
    sent = []
    nrrows = len(codecs.open(fh).readlines())
    sid2tokens = defaultdict(list)

    for row in reader:
        if len(row) > 5:
            sTokenId = row[0]
            token = row[1]
            lemma = row[2]
            pos_coarse = row[3]
            pos_fine = row[4]
            morph_feats = row[5]
            head_tokenid = row[6]
            deprel = row[7]
            if lemma.startswith('$') or lemma == 'PUNCT': # for some tab chars, there was no surface form and lemma
                lemma = token
                pos_coarse = row[2]
                pos_fine = row[3]
                morph_feats = row[4]
                head_tokenid = row[5]
                deprel = row[6]
            segmentStarter = False
            if re.search('BeginSeg=Yes', ' '.join(row[8:])) or re.search('Seg=[BI]-Conn', ' '.join(row[8:])): #  to accomodate both EDUs and PDTB-style marking
                segmentStarter = True
            cct = CustomCONLLToken(uid, sid, sTokenId, token, pos_coarse, pos_fine, segmentStarter, head_tokenid, deprel)
            if lang:
                cct.addLang(lang)
            sent.append(token)
            tokens.append(cct)
        else:
            sentences.append(sent)
            sid2tokens[sid] = sent
            sent = []
            sid += 1
        uid += 1
        if uid == nrrows:
            sentences.append(sent)
            sid2tokens[sid] = sent


    for cct in tokens:
        fullsent = sid2tokens[cct.sid]
        cct.addFullSentence(fullsent)
    
    return tokens
