import gpxpy
import matplotlib.pyplot as plt
# import datetime
# from geopy import distance
# from math import sqrt, floor
# import numpy as np
import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objs as go
# import haversine

with open('/home/morzh/work/my_run_001.gpx', 'r') as fh:
    gpx = gpxpy.parse(fh)

print('number of tracks is', len(gpx.tracks))
print('number of segments is', len(gpx.tracks[0].segments))
print('number of points is', len(gpx.tracks[0].segments[0].points))

data = gpx.tracks[0].segments[0].points
df = pd.DataFrame(columns=['lon', 'lat', 'alt', 'time'])

for point in data:
    df = df.append({'lon': point.longitude, 'lat': point.latitude, 'alt': point.elevation, 'time': point.time}, ignore_index=True)

plt.plot(df['lon'], df['lat'])
plt.show()

# plt.plot(df['time'], df['alt'])
# plt.show()

_data = [go.Scatter3d(x=df['lon'], y=df['lat'], z=df['alt'], mode='lines')]
py.iplot(_data)
