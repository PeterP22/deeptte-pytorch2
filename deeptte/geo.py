"""Geodesic helpers shared by data preparation and serving."""
from math import asin, cos, radians, sin, sqrt


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance between two points, in km."""
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371
