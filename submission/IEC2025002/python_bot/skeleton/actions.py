from collections import namedtuple
FoldAction  = namedtuple('FoldAction',  [])
CallAction  = namedtuple('CallAction',  [])
CheckAction = namedtuple('CheckAction', [])
RaiseAction = namedtuple('RaiseAction', ['amount'])
