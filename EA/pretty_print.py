################
# Writes data to html
# table
################

import pickle
import numpy
import os
from prettytable import PrettyTable
import numpy.lib.recfunctions as rfn


"""Date | Time | TradedPrice | TradedQty | TradedSign_M | TradedSign_C | TradedSign | TradedSplit | TradedCross | BlockTrade | BidPrice | BidQty | BidSplit | AskPrice | AskQty | AskSplit | PriceMove | BidLim | BidCan | AskLim | AskCan | AskMar | BidMar | PMM | PMP"""

        
def write_html(r, render_to='table.html'):
        types = numpy.zeros(len(r), dtype=[('Type', 'S60'), ])

        for i, row in enumerate(r[type_fields]):
                for j, k in enumerate(row):
                        if k == True:
                                types[i][0] += type_fields[j] + ' | '

        r = rfn.merge_arrays([r, types], flatten=True)[used_columns]

        x = PrettyTable(r.dtype.names)
        for row in r:
                
                x.add_row(row)

        with open('index.html', 'r') as template:
                with open(render_to, 'w') as output:
                        s = template.read()
                        s = s.replace("{{ table }}", x.get_html_string(attributes={"class": "table table-hover table-striped"}))
                        
                        output.write(s)


type_fields = ['BidLim', 'BidCan', 'AskLim', 'AskCan', 'AskMar', 'BidMar', 'PMM', 'PMP'] 
used_columns = [ 'Time', 'Type', 'TradedPrice', 'TradedQty', 'TradedSplit', 'TradedCross', 'BlockTrade', 'BidPrice', 'BidQty', 'BidSplit', 'AskPrice', 'AskQty', 'AskSplit', 'PriceMove' ]


data_file = 'xFGBL20130702.pkl'

with open(data_file, 'rb') as input:
        r=pickle.load(input) [:100]
        write_html(r)




