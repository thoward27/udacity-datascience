import sqlite3 as sql
import os
import xml.etree.cElementTree as ET
import time

users = (
    "CREATE TABLE IF NOT EXISTS users ("
    "u_id integer PRIMARY KEY,"
    "u_name text NOT NULL"
    ");"
)
nodes = (
    "CREATE TABLE IF NOT EXISTS nodes ("
    "n_id integer PRIMARY KEY,"
    "u_id integer NOT NULL,"
    "lat text NOT NULL,"
    "lon text NOT NULL,"
    "FOREIGN KEY (u_id) REFERENCES users (u_id) ON DELETE CASCADE ON UPDATE NO ACTION"
    ");"
)
node_tags = (
    "CREATE TABLE IF NOT EXISTS node_tags ("
    "n_id integer NOT NULL,"
    "k text NOT NULL,"
    "v text NOT NULL,"
    "PRIMARY KEY (n_id, k)"
    "FOREIGN KEY (n_id) REFERENCES nodes (n_id) ON DELETE CASCADE ON UPDATE NO ACTION"
    ");"
)
ways = (
    "CREATE TABLE IF NOT EXISTS ways ("
    "w_id integer PRIMARY KEY,"
    "u_id integer NOT NULL,"
    "FOREIGN KEY (u_id) REFERENCES users (u_id) ON DELETE CASCADE ON UPDATE NO ACTION"
    ");"
)
way_tags = (
    "CREATE TABLE IF NOT EXISTS way_tags ("
    "w_id integer NOT NULL,"
    "k text NOT NULL,"
    "v text NOT NULL,"
    "PRIMARY KEY (w_id, k),"
    "FOREIGN KEY (w_id) REFERENCES ways (w_id) ON DELETE CASCADE ON UPDATE NO ACTION"
    ");"
)
way_refs = (
    "CREATE TABLE IF NOT EXISTS way_refs ("
    "w_id NOT NULL,"
    "n_id NOT NULL,"
    "PRIMARY KEY (w_id, n_id),"
    "FOREIGN KEY (n_id) REFERENCES nodes (n_id) ON DELETE CASCADE ON UPDATE NO ACTION"
    ");"
)
relations = (
    "CREATE TABLE IF NOT EXISTS relations ("
    "r_id integer PRIMARY KEY,"
    "k text NOT NULL,"
    "v text NOT NULL"
    ");"
)
relation_members = (
    "CREATE TABLE IF NOT EXISTS relation_members ("
    "r_id NOT NULL,"
    "m_id NOT NULL,"
    "m_type NOT NULL,"
    "PRIMARY KEY (r_id, m_id, m_type),"
    "FOREIGN KEY (r_id) REFERENCES relations (r_id) ON DELETE CASCADE ON UPDATE NO ACTION"
    ");"
)
def reset_db(db_name):
    """Completely resets the database. Caution as this deletes the file, and recreates all tables."""
    if os.path.isfile(db_name): os.remove(db_name)
    conn = sql.connect(db_name)
    c = conn.cursor()
    try:
        c.execute(users)
        c.execute(nodes)
        c.execute(node_tags)
        c.execute(ways)
        c.execute(way_tags)
        c.execute(way_refs)
        c.execute(relations)
        c.execute(relation_members)
    except Exception as err:
        print("err")
        conn.rollback()
    conn.commit()
    conn.close()
    return

def fill_tables(file, db_name):
    """split file into tables"""
    conn = sql.connect(db_name)
    c = conn.cursor()
    parent = False
    p_type = ''
    context = ET.iterparse(file, events=("start", "end"))
    context = iter(context)
    event, root = next(context)
    for event, e in context:
        try:
            # handle tags and nds
            try:
                #check for tags
                if e.tag == 'tag' and parent:
                    tag_attribs = (parent, e.attrib['k'], e.attrib['v'])
                    if p_type == 'node': c.execute("INSERT OR IGNORE INTO node_tags(n_id, k, v) VALUES(?, ?, ?)", (tag_attribs))
                    if p_type == 'way': c.execute("INSERT OR IGNORE INTO way_tags(w_id, k, v) VALUES(?, ?, ?)", (tag_attribs))
                #if it's nd and parent is way
                elif e.tag == 'nd' and p_type == 'way':
                    way_ref_attribs = (parent, e.attrib['ref'])
                    c.execute("INSERT OR IGNORE INTO way_refs(w_id, n_id) VALUES(?, ?)", (way_ref_attribs))
                elif e.tag == 'nd' and p_type == 'relation':
                    print(e.tag, e.attrib, p_type)
                elif e.tag == 'nd' and p_type == 'node':
                    print(e.tag, e.attrib, p_type)
            except KeyError:
                print("bad tag: {}".format(e.attrib))
                pass
            else:
                conn.commit()
            
            # update user table
            try:
                u_id = e.attrib['uid']
                u_name = e.attrib['user']
                c.execute("INSERT OR IGNORE INTO users(u_id, u_name) VALUES(?, ?)", (u_id, u_name))
            except KeyError:
                pass
            else:
                conn.commit()
            
            # check for id
            try:
                e_id = e.attrib['id']
            except KeyError:
                conn.commit()
                root.clear()
                continue            
            # node / way / relation
            try:
                if e.tag == 'node': 
                    node_attribs = (e_id, u_id, e.attrib['lat'], e.attrib['lon'])
                    c.execute("INSERT OR IGNORE INTO nodes(n_id, u_id, lat, lon) VALUES(?, ?, ?, ?)", (node_attribs))
                    parent, p_type = e_id, 'node'
                elif e.tag == 'way':
                    way_attribs = (e_id, u_id)
                    c.execute("INSERT OR IGNORE INTO ways(w_id, u_id) VALUES(?, ?)", (way_attribs))
                    parent, p_type = e_id, 'way'
                elif e.tag == 'relation':
                    relation_attribs = (e_id, e.attrib['k'], e,attrib['v'])
                    c.execute("INSERT OR IGNORE INTO relations(r_id, k, v) VALUES(?, ?, ?)", (relation_attribs))
                    parent, p_type = e_id, 'relation'
            except KeyError:
                print("bad parent")
                pass
            else: 
                conn.commit()
            
            #catch and commit any other changes(this clears the journal which saves mem)
            conn.commit()
            root.clear()
        except KeyboardInterrupt:
            conn.commit()
            root.clear()
            input("paused")
            continue
        except Exception as err:
            conn.commit()
            input("unhandled err: {} Tag: {}={} parent: {}".format(err, e.tag, e.attrib, p_type))
            continue
    print("exiting loop")
    conn.commit()
    conn.close()
    return


reset_db('bostonv2.db')
fill_tables('boston.osm', 'bostonv2.db')