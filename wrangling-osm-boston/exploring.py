#python3

###### imports
import sqlite3 as sql
import os
import pandas as pd
import numpy as np
from datetime import date

###### Define functions
def crshmidt_nodes():
	"""Prepares stats for crshmidt user
	"""
	d1 = date(2017, 4, 1)
	# when OSM was founded
	d0 = date(2004, 8, 9)
	delta = d1 - d0
	nodes = 1201241
	nodes_per_day = nodes / delta.days
	nodes_per_hour = nodes_per_day / 24
	nodes_per_waking_hour = nodes_per_day / (24-10)
	nodes_per_minute = nodes_per_waking_hour / 60

	line = (
		f"crshmidt has posted: "
		f"\n{nodes_per_day:>10.2f} nodes per day, "
		f"\n{nodes_per_hour:>10.2f} nodes per hour *assumes posting 24 hours per day, "
		f"\n{nodes_per_waking_hour:>10.2f} nodes per hour *assumes normal waking hours, "
		f"\n{nodes_per_minute:>10.2f} nodes per waking minute"
	)
	print(line)

def piechart_users(num_contrib, df, sort_col='cnt_total'):
	"""Takes in num_contributed, df, and sort col and produces a pie chart. 
	"""
	# sort by sort_col, then take num_contrib off the top
	pie_data = df.sort_values(by=sort_col, ascending=False)
	pie_data = pie_data.iloc[:num_contrib]
	sum_pie_data = pie_data.sum(axis=0, numeric_only=True)
	# generate 'other' index containing the rest of the elements from the unselected group
	other = 1 - sum_pie_data
	other[other < 0] = 0
	other['cnt_nodes'] = num_nodes - sum_pie_data['cnt_nodes']
	other['cnt_ways'] = num_ways - sum_pie_data['cnt_ways']
	other['cnt_total'] = num_total - sum_pie_data['cnt_total']
	other['u_name'] = "other"
	pie_data.loc[-1] = other
	# pie chart
	labels = pie_data['u_name']
	sizes = pie_data['cnt_total']

	fig = plt.figure(figsize=[10, 10])
	ax = fig.add_subplot(111)
	ax.pie(sizes, labels=labels, autopct='%1.0f%%', labeldistance=1.05)
	ax.set_title(f"Top {num_contrib} OSM User Contributions to all Objects")
	plt.show();
	print(
		f"Top {num_contrib} OSM Users account for: "
		f"\n{sum_pie_data['per_total']:>10.2%} of total ojects, "
		f"\n{sum_pie_data['per_nodes']:>10.2%} of total nodes, "
		f"\n{sum_pie_data['per_ways']:>10.2%} of total ways."
	)
	return pie_data, sum_pie_data

def sq_ft_to_sq_meters(cell):
    """Small function to parse sq ft and replace with unformatted sq meters
    
    Pass in cell, using .apply method of Pandas onto desired vector.
    """
    match = re.search(regex_sq_ft, cell)
    target = match['number']
    num = target.replace(',','')
    new = int(num) * 0.092903
    new = (f"{new:2.2f}")
    cell = cell.replace(target, new)
    cell = cell.replace(match['units'], "sq meters")
    return cell

	
##### link to db
conn = sql.connect('bostonV2_python.db')

# Build user table
sql_select = (
    "SELECT users.u_name as u_name_{0}, users.u_id, count(*) as cnt_{0} "
    "FROM {0}, users "
    "WHERE users.u_id = {0}.u_id "
    "GROUP BY users.u_id "
    "ORDER BY cnt_{0} desc "
)
# I want counts in my user table of contributed nodes and ways:
users = pd.concat(
    objs = [
        pd.read_sql(sql_select.format("nodes"), con=conn, index_col="u_id"), 
        pd.read_sql(sql_select.format("ways"), con=conn, index_col="u_id")
    ],
    axis=1,
    join='outer',
)
# add a grand total
users['cnt_total'] = users['cnt_nodes'] + users['cnt_ways']
# sort the df by total, nodes, then ways desc so we can see our top users more quickly
users.sort_values(
    by=['cnt_total', 'cnt_nodes','cnt_ways'],
    ascending=False, 
    inplace=True
)
# fill NaNs with 0
users.fillna(0, axis=1, inplace=True)
# cast as uint32, which takes slightly less memory than float64 (which is default)
users = users.astype(dtype={'cnt_nodes':np.uint32, 'cnt_ways': np.uint32})
users['u_name'] = users['u_name_ways']
users.drop(['u_name_ways', 'u_name_nodes'], inplace=True, axis=1)
# build the table out:
num_nodes = sum(users['cnt_nodes'])
num_ways = sum(users['cnt_ways'])
num_total = num_nodes + num_ways
num_users = len(users['u_name'])
users['per_nodes'] = users['cnt_nodes'] / num_nodes
users['per_ways'] = users['cnt_ways'] / num_ways
users['per_total'] = users['cnt_total'] / num_total
# summarize users table
print(f"There are {num_users} users mapping in Boston.")
print(f"They've mapped a total of {num_total} objects, of which {num_ways} are ways, and {num_nodes} are nodes.")

# if you want a pie chart you have to uncomment here..
#pie_data, sum_pie_data = piechart_users(num_contrib=10, df=users)

#build node_tags
sql_select = (
    "SELECT * "
    "FROM node_tags "
)
node_tags = pd.read_sql(sql_select, conn)
num_node_tags = len(node_tags)

# sq feet regex pattern
sq_feet = r"""
    ((?P<number>[\d,]+)\s(?P<units>sq.?\sft\.?)) # find feet
"""
regex_sq_ft = re.compile(sq_feet, re.VERBOSE)
# using pandas contains, to hunt down and murder American units (or replace, either way)
tags_with_feet = node_tags[node_tags['v'].str.contains(regex_sq_ft)]
tags_with_feet['v'] = tags_with_feet['v'].apply(sq_ft_to_sq_meters)
print(f"some tags that used to have sq ft: {tags_with_feet}")

# build nodes table
sql_select = (
    "SELECT * "
    "FROM nodes"
)
nodes = pd.read_sql(sql_select, con=conn, index_col="n_id")
nodes_without_tags = nodes[~nodes.index.isin(node_tags['n_id'])]
num_nodes_without_tags = len(nodes_without_tags)

print(f"There are {num_node_tags} node tags in the database. \nLeaving {num_nodes_without_tags} nodes without tags")

crshmidt_nodes = nodes[nodes['u_id'] == 1034].index.tolist()
print(f"User crshmidt has posted: {len(crshmidt_nodes)} nodes")

# and here we grab all node_tags associated with those nodes
crshmidt_node_tags = node_tags[
    (node_tags['n_id'].isin(crshmidt_nodes))
]
print(f"These nodes are associated with {len(crshmidt_node_tags)} tags")
print(f"These tags are associated with {len(pd.value_counts(crshmidt_node_tags['n_id'].values.ravel()))} unique nodes")
print(f"Meaning, they have posted {len(crshmidt_nodes) - len(pd.value_counts(crshmidt_node_tags['n_id'].values.ravel()))} "
      "untagged nodes"
     )

# here is just GIS tags
crshmidt_node_tags = node_tags[
    (node_tags['n_id'].isin(crshmidt_nodes)) & 
    (node_tags['v'].str.contains('gis|Office of Geographic|JOSM', case=False,))
]

num_crshmidt_node_tags = len(pd.value_counts(crshmidt_node_tags['n_id'].values.ravel()))
print(f"Of these nodes, {num_crshmidt_node_tags} have been tagged with GIS or JOSM")


places_of_worship = node_tags[(node_tags.k == "amenity") &  (node_tags.v == "place_of_worship")]['n_id']
places_of_worship = node_tags[node_tags['n_id'].isin(places_of_worship)]


places_to_eat = node_tags[(node_tags.k == "amenity") & (node_tags.v == "restaurant")]['n_id']
places_to_eat = node_tags[node_tags['n_id'].isin(places_to_eat)]

num_vegan_eats = pd.value_counts(places_to_eat[
    (places_to_eat.k == "cuisine") & 
    (places_to_eat.v.str.contains('vegan', case=False))
]['v'].values.ravel())

# and what about names?
vegan_node_tags = node_tags[
    (
        node_tags['n_id'].isin(
            places_to_eat[
                (places_to_eat.k == "cuisine") & 
                (places_to_eat.v.str.contains('vegan', case=False))
            ]['n_id']
        )
    ) & (
        node_tags.k.isin(["name"])
    ) 
]
print(f"turns out there are {len(num_vegan_eats)} vegan restaurants")
vegan_node_tags


