import re

LEFT = 1
RIGHT = 2

UNK   = 1
DICT  = 2
INIT  = 3
LATIN = 4
PUNC  = 5

def seek(wordlist, l, r, ch, str_offset, pos):
    ans = None
    while l <= r:
        m = (l + r) // 2
        dict_item = wordlist[m]
        word_len = len(dict_item)
        if word_len <= str_offset:
            l = m + 1
        else:
            ch_ = dict_item[str_offset]
            if ch_ < ch:
                l = m + 1
            elif ch_ > ch:
                r = m - 1
            else:
                ans = m
                if pos == LEFT:
                    r = m - 1
                else:
                    l = m + 1
    return ans

def is_better(link0, link1):
    if link0 is None:
        return True

    if link1["unk"] < link0["unk"]:
        return True

    if link1["w"] < link0["w"]:
        return True

    return False

def build_path(wordlist, s):
    left_boundary = 0
    dict_acc_list = []

    path = [{"p":None, "w": 0, "unk": 0, "type": INIT}]

    latin_s = None
    latin_e = None

    punc_s = None
    punc_e = None

    for i, ch in enumerate(s):
        dict_acc_list.append({"s":i, "l": 0, "r": len(wordlist)-1})

        # Update dict acceptors
        _dict_acc_list = dict_acc_list
        dict_acc_list = []
        for acc in _dict_acc_list:
            l = seek(wordlist, acc["l"], acc["r"], ch, i-acc["s"], LEFT)
            if l is not None:
                is_final = len(wordlist[l]) == i - acc["s"] + 1
                r = seek(wordlist, l, acc["r"], ch, i-acc["s"], RIGHT)
                dict_acc_list.append({"s":acc["s"], "l": l, "r": r, "final":is_final})

        # latin words
        if latin_s is None:
            if re.match(u"[A-Za-z]", ch):
                latin_s = i

        if latin_s is not None:
            if re.match(u"[A-Za-z]", ch):
                if i + 1 == len(s) or re.match(u"[A-Za-z]", s[i + 1]):
                    latin_e = i
            else:
                latin_s = None
                latin_e = None

        # puncuation
        if punc_s is None:
            if ch == " ":
                punc_s = i

        if punc_s is not None:
            if ch == " ":
                if len(s) == i + 1 or s[i + 1] != " ":
                    punc_e = i
            else:
                punc_s = None
                punc_e = None

        # select link
        link = None

        # links from wordlist
        for acc in dict_acc_list:
            if acc["final"]:
                p_link = path[acc["s"]]
                _link = {"p": acc["s"],
                         "w": p_link["w"] + 1,
                         "unk": p_link["unk"],
                         "type": DICT}
                if is_better(link, _link):
                    link = _link

        # link from latin word
        if latin_s is not None and latin_e is not None:
            p_link = path[latin_s]
            _link = {"p": latin_s,
                     "w": p_link["w"] + 1,
                     "unk": p_link["unk"],
                     "type": LATIN}
            if is_better(link, _link):
                link = _link

        # link from puncuation
        if punc_s is not None and punc_e is not None:
            p_link = path[punc_s]
            _link = {"p": punc_s,
                     "w": p_link["w"] + 1,
                     "unk": p_link["unk"],
                     "type": PUNC}
            if is_better(link, _link):
                link = _link

        # fallback
        if link is None:
            p_link = path[left_boundary]
            link = {"p": left_boundary,
                    "w": p_link["w"] + 1,
                    "unk": p_link["unk"] + 1,
                    "type": UNK}
        path.append(link)
        if link["type"] != UNK:
            left_boundary = i
    return path

def path_to_tokens(txt, path):
    if len(path) < 2:
        return None

    e = len(path) - 1
    toks = []

    while True:
        link = path[e]
        s = link["p"]
        if s is None:
            break
        toks.append(txt[s:e])
        e = s

    toks.reverse()
    return toks

def tokenize(wordlist, txt):
    path = build_path(wordlist, txt)
    return path_to_tokens(txt, path)

class Wordcut(object):
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def tokenize(self, s):
        return tokenize(self.wordlist, s)
