# OpenStreetMaps Boston, MA

## About
Data Wrangling project, the third project for Udacity's Data Analyst Nanodegree program. 

This project aims to clean the [OpenStreetMaps](https://www.openstreetmap.org "OpenStreetMaps") data from Boston, MA. To do this, I've used Python to iteratively parse, database, and normalize the file. Once fully normalized and inside of an SQLite3 Database, I performed analysis on the data, attempting to uncover user-committed errors and correcting them. 

## See it live!
[Click Here](https://thoward27.github.io/Udacity_Wrangling-OpenStreetMaps-Data_Boston-MA/)

## Reference Table:
* [bostonv2.db](#bostonv2.db)
* [db-reset.py](#db-reset.py)
* [exploring.py](#exploring.py)
* [full_analysis.html](#full_analysis.html)
* [full_analysis.ipynb](#full_analysis.ipynb)
* [OSM_Boston.html](#OSM_Boston.html)
* [readme.md](#readme.md)
* [sample.osm](#sample.osm)
* [sample.py](#sample.py)

## Folder Contents
| File                | Description
|---------------------|----------------------------------------------------------------------------------------------------------------------------------|
| db-reset.py         | resets and rebuilds the database using an included schema, please don't run it unless you're sure                                |
| exploring.py        | meant to be IMPORTED. Once you've done that it will print a few stats out then let you play with the preloaded data from the db  |
| full_analysis.html  | contains all work, including process, links to help, etc                                                                         |
| full_analysis.ipynb | ipython version of the .html document                                                                                            |
| OSM_Boston.html     | ** THIS IS IT ** the actual analysis is in this file.                                                                            |
| readme.md           | you're here now, welcome                                                                                                         |
| sample.py           | instructure provided python                                                                                                      |


## File Explinations and Recreation steps
This section will briefly discuss all files and the process to recreate any files that are not included in the repository due to size limitations. 

### bostonv2.db <a name="bostonv2.db"></a>
Database file, created via db-reset.py. 

### db-reset.py <a name="db-reset.py"></a>
Script for processing database osm file. 

### exploring.py <a name="exploring.py"></a>
Exploratory python script. Mirrors functionality of iPython Notebook, but in terminal form. 

### full_analysis.html <a name="full_analysis.html"></a>
Contains working notes, full methodologies, etc. 

### full_analysis.ipynb <a name="full_analysis.ipynb"></a>
iPython notebook version of the full analysis. 

### OSM_Boston.html <a name="OSM_Boston.html"></a>
Contains submitted analysis of file. 

### readme.md <a name="readme.md"></a>
This file your looking at right here!

### sample.osm <a name="sample.osm"></a>
Sample osm data created by sample.py (instructor provided script)

### sample.py <a name="sample.py"></a>
Instructor provided script modified for Python 3. 

## Thank you for reading!
