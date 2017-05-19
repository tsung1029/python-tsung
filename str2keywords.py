import re
import numpy as np

# split id and keywords by whitespece or , or ;
id_key_pattern = re.compile(r'[\s,;]+')
# split keywords by commas except for those inside () and []
keyword_pattern = re.compile(r',(?!(?:[^([]*[(\[][^)\]]*[)\]])*[^()\[\]]*[)\]])')


# Here we assume the brackets are all balanced.
# Otherwise the keyword dictionary would cause runtime failure sooner or later
class str2keywords:
    def __init__(self, string):
        self.id, string = id_key_pattern.split(string + ' ', maxsplit=1)
        # remove whitespaces
        self.id = " ".join(self.id.split())
        string = " ".join(string.split())
        string = keyword_pattern.split(string)
        if string[-1] == '':
            string.pop(-1)
        # store keywords as dictionary
        self.keywords = dict(tuple(item.split('=')) for item in string)
        for k, v in self.keywords.iteritems():
            self.keywords[k] = eval(v)
        print self.keywords
        print self.id

    def __eq__(self, other):
        return self.id == other

# some tests
if __name__ == '__main__':
    kw = str2keywords('oeau norm="ortho"')
    a = np.mgrid[:3, :3][0]
    # use ** to unpack the dictionary
    a = np.fft.fft2(a, **kw.keywords)
    print a

