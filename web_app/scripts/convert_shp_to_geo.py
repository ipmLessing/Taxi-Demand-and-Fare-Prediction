import geopandas as gp
import shapefile
from json import dumps

# read NYC taxi data shapefile
gpf = gp.read_file('taxi_zones/taxi_zones.shp')

# convert it into gps shapefile
gpf_geo = gpf.to_crs(epsg=4326)
print(gpf.iloc[0].geometry.centroid.y, gpf_geo.iloc[0].geometry.centroid.y)

# save it into geojson
print(gpf_geo.fields)
gpf_geo.to_file('geo_taxi_zones_out.shp')

'''
fields = gpf_geo.fields[1:]
field_names = [field[0] for field in fields]
buffer = []
for sr in gpf_geo.shapeRecords():
    atr = dict(zip(field_names, sr.record))
    geom = sr.shape.__geo_interface__
    buffer.append(dict(type="Feature", geometry=geom, properties=atr)) 
   
    # write the GeoJSON file
   
geojson = open("pyshp-demo.json", "w")
geojson.write(dumps({"type": "FeatureCollection", "features": buffer}, indent=2) + "\n")
geojson.close()
'''