# Usage

This program requires an export of shipstation's open orders. 

Obtain an export by going to All orders > “Metrics Export” view > Other Actions > Export Orders (export order line items, and use Fulmetrics format)

Once you have the csv, run the program using the command `$ python fulmetrics.py "./path/to/orders.csv"`

It will print out a list of individual bags owed, as well as a list of the top skus (boxes) owed
