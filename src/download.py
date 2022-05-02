from astroquery.vizier import Vizier
from astropy import coordinates
from astropy import units as u

v = Vizier(keywords=['stars:white_dwarf'])

c = coordinates.SkyCoord(0, 0, unit=('deg', 'deg'), frame='icrs')
result = v.query_region(c, radius=2*u.deg)

print(len(result))
# 44

result[0].pprint()