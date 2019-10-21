#!/usr/bin/env python
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.cross_correlation import correlate
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
import util


debug = True
net = "IU"
sta = "ANMO"
f0 = 1./50.
# Mexico event
stime = UTCDateTime('2017-261T00:00:00')
etime = stime + 20*24*60*60

# Grab an earthquake
client = Client()
cat = client.get_events(starttime=stime, endtime=etime, 
                                minmagnitude=6.5)

eve = cat[0]
if debug:
    print(eve)

# grab some metadata
inv = client.get_stations(network=net, station=sta, starttime = stime,
                          endtime = etime, level="response")

# figure out the Rayeligh wave window
coords = inv.get_coordinates(net + '.' + sta + '.00.LHZ', eve.origins[0].time)

(dis, azi, bazi) = gps2dist_azimuth(coords['latitude'], coords['longitude'], 
                                    eve.origins[0].latitude,eve.origins[0].longitude)

dis /= 1000.
disdeg = dis*0.0089932

if debug:
    print(disdeg)

wstime = eve.origins[0].time + dis/3.5 -180.
wetime = wstime + 360.
wctime = wstime + 180.

st = client.get_waveforms(net, sta, "*", "LHZ", wstime, wetime)
st.detrend('constant')
st.taper(0.05)
st.remove_response(inventory=inv)
fig = plt.figure(1)
t = np.arange(-300, 301, 1)
xfun = correlate(st[0].data, st[1].data, 300)
plt.plot(t,xfun, label='Raw')

xfun = util.taper_fun(xfun,f0)
plt.plot(t,xfun, label='Taper')
xfun = util.filtergauss(xfun, 1./75.)
plt.plot(t, xfun, label='filter')


plt.legend()
plt.show()

