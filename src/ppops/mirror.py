"""
mirror.py
-------------
Handles POPS mirror geometry calculations.
"""


def effective_focal_length(radius_of_curvature: float) -> float:
    """Calculate the radius of curvature for a spherical mirror.

    Parameters
    ----------
    radius_of_curvature : float
        Radius of curvature of the mirror.

    Returns
    -------
    float
        Radius of curvature of the mirror. Units are the same as
        radius_of_curvature.

    Citations
    ---------
    Libretexts. (2025, March 26). 2.3: Spherical Mirrors. Physics 
    LibreTexts.
    https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/University_Physics_III_-_Optics_and_Modern_Physics_(OpenStax)/02%3A_Geometric_Optics_and_Image_Formation/2.03%3A_Spherical_Mirrors
    """
    return radius_of_curvature / 2


def mirror_depth(mirror_radius: float, radius_of_curvature: float) -> float:
    """Calculate the depth of a spherical mirror.

    Parameters
    ----------
    mirror_radius : float
        Radius of the mirror. Must have the same units as 
        radius_of_curvature.
    radius_of_curvature : float
        Radius of curvature of the mirror. Must have the same units as 
        mirror_radius.

    Returns
    -------
    float
        Depth of the mirror. Units are the same as the input parameters.
    """
    return radius_of_curvature - (radius_of_curvature**2 - mirror_radius**2)**0.5

if __name__ == "__main__":
    print(14.229-mirror_depth(12.5, 20))
