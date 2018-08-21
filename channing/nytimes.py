# -*- coding: utf-8 -*-

"""\
© Copyright 2018. Channing. All rights reserved.

NYT API Interaction Utility

"""

from keys import nytimes_key
from nytimesarticle import articleAPI
api = articleAPI(nytimes_key)
print(api.query(2016, 11))
