#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import MySQLdb

# Conectar una base de datos tipo MySQL

db = MySQLdb.connect(host="201.165.222.2",
                     port=3306, user="statfac",
                     passwd=".f%d*q+4e{=5",
                     db="facebookstatus")

query = """
SELECT id, create_time, id_from, comment, like_count, flag, ready
FROM rmv_comments
GROUP BY comment
"""

df_mysql = pd.read_sql(query, con=db)

print 'loaded dataframe from MySQL. records:', len(df_mysql)
db.close()

print df_mysql.head(5)

df_mysql.to_csv("datos_raw.csv.gz", compression='gzip')
