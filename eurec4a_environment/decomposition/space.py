import cartopy.crs as ccrs
import cartopy.geodesic as cgeod
import numpy as np


class Circle:
    def __init__(self, lat, lon, r_degrees):
        self.lat = lat
        self.lon = lon
        self.r_degrees = r_degrees

        self.radius = self._compute_radius()
        self.geod = cgeod.Geodesic()  # Earth geodesic

    def _compute_radius(self):
        # Define the projection used to display the circle:
        ortho = ccrs.Orthographic(central_longitude=self.lon, central_latitude=self.lat)

        # Compute the required radius in projection native coordinates:
        phi1 = (
            self.lat + self.r_degrees
            if self.lat <= 0
            else self.lat - self.r_degrees
        )
        _, y1 = ortho.transform_point(self.lon, phi1, ccrs.PlateCarree())
        return abs(y1)

    def distance_to_center(self, lat, lon):
        @np.vectorize
        def _dist(lat1, lon1, lat2, lon2):
            geometry = np.array(
                [
                    [lon1, lat1],
                    [lon2, lat2],
                ]
            )
            return self.geod.geometry_length(geometry)

        return _dist(lat1=self.lat, lon1=self.lon, lat2=lat, lon2=lon)

    def contains(self, lat, lon):
        """
        Return whether a point is contained in the circle or not
        The points are assumed to be in spherical (lon, lat) coordinates
        """
        p_dist = self.distance_to_center(lat=lat, lon=lon)
        print(p_dist, self.radius)
        return p_dist < self.radius


class HALOCircle(Circle):
    # circle location and radius in degrees:
    lat = 13 + (18 / 60)
    lon = -(57 + (43 / 60))
    r_degrees = 1

    def __init__(self):
        super().__init__(lat=self.lat, lon=self.lon, r_degrees=self.r_degrees)
