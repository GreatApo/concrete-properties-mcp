# Concrete Properties MCP
# This module handles the connection with the Concrete Properties Python library.

from mcp.server.fastmcp import Image
from pydantic import BaseModel
from config import *
import math

# Concrete Properties libraries
from concreteproperties.material import Concrete, SteelBar
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
    SteelElasticPlastic,
)
from concreteproperties.concrete_section import ConcreteSection
from concreteproperties.pre import add_bar
from sectionproperties.pre.geometry import Geometry, Polygon, CompoundGeometry
from sectionproperties.pre.library import concrete_rectangular_section
from sectionproperties.analysis.section import Section
from concreteproperties.results import TransformedGrossProperties

# For plots
import matplotlib.pyplot as plt
import io


# Input data classes
class Point(BaseModel):
    """
    Represents a point in 2D space.
    
    Attributes:
        x (float): X-coordinate of the point (mm).
        y (float): Y-coordinate of the point (mm).
    """
    x: float
    """X-coordinate of the point (mm)."""
    y: float
    """Y-coordinate of the point (mm)."""

class Rebar(BaseModel):
    """
    Represents a rebar in the concrete section.

    Attributes:
        x (float): X-coordinate of the rebar (mm).
        y (float): Y-coordinate of the rebar (mm).
        diameter (float): Diameter of the rebar (mm).
    """
    x: float
    """X-coordinate of the rebar (mm)."""
    y: float
    """Y-coordinate of the rebar (mm)."""
    diameter: float
    """Diameter of the rebar (mm)."""

    def area(self) -> float:
        """
        Calculate the area of the rebar.

        Returns:
            float: Area of the rebar (mm^2).
        """
        return math.pi * (self.diameter / 2) ** 2

# Return data classes
class InteractionPoint():
    """
    Represents a point in the axial-moment interaction diagram.

    Attributes:
        n (float): Axial load (kN).
        m (float): Moment (kN.m).
    """

    def __init__(self, n: float, m: float):
        self.n = n
        self.m = m

# Properties class (initialised with user settings)
class Properties():
    def __init__(self, config : Config):
        self.mesh_size = config.mesh_size # mm

        self.concrete_material_density = config.concrete_material_density
        self.concrete_rectangular_stress_block_alpha = config.concrete_rectangular_stress_block_alpha
        self.concrete_rectangular_stress_block_gamma = config.concrete_rectangular_stress_block_gamma
        self.concrete_rectangular_stress_block_ultimate_strain = config.concrete_rectangular_stress_block_ultimate_strain
        self.concrete_flexural_tensile_strength = config.concrete_flexural_tensile_strength

        self.rebar_mesh_points = config.rebar_mesh_points
        self.rebar_material_density = config.rebar_material_density
        self.rebar_material_fracture_strain = config.rebar_material_fracture_strain

    # Helper functions
    def get_concrete_material(self, conc_elastic_modulus : float = 30.1e3, f_c : float = 40) -> Concrete:
        """
        Returns a dummy concrete material.

        Args:
            conc_elastic_modulus (float): Elastic modulus of the concrete (MPa).
            f_c (float): Concrete compressive strength (MPa).
        """

        return Concrete(
            name = "Concrete",
            density = self.concrete_material_density,
            stress_strain_profile = ConcreteLinear(elastic_modulus = conc_elastic_modulus),
            ultimate_stress_strain_profile = RectangularStressBlock(
                compressive_strength = f_c,
                alpha = self.concrete_rectangular_stress_block_alpha,
                gamma = self.concrete_rectangular_stress_block_gamma,
                ultimate_strain = self.concrete_rectangular_stress_block_ultimate_strain,
            ),
            flexural_tensile_strength = self.concrete_flexural_tensile_strength,
            colour = "lightgrey",
        )

    def get_rebar_material(self, rebars_elastic_modulus : float = 200_000, fy : float = 500) -> SteelBar:
        """
        Returns a dummy steel rebar material.

        Args:
            rebars_elastic_modulus (float): Elastic modulus of the rebar (MPa).
            fy (float): Yield strength of the rebar (MPa).
        """

        return SteelBar(
            name="Steel Rebar",
            density = self.rebar_material_density,
            stress_strain_profile = SteelElasticPlastic(
                yield_strength = fy,
                elastic_modulus = rebars_elastic_modulus,
                fracture_strain = self.rebar_material_fracture_strain,
            ),
            colour="grey",
        )

    def get_geometry_from_points(self, perimeter_points : list[Point], rebars : list[Rebar], concrete_material : Concrete, rebar_material : SteelBar) -> CompoundGeometry:
        """
        Create a geometry from a list of points and add rebars to it.
        
        Args:
            perimeter_points (list[Point]): List of points defining the perimeter of the section.
            rebars (list[Rebar]): List of rebar objects to be added to the geometry.
            concrete_material (Concrete): Concrete material object.
            rebar_material (SteelBar): Rebar material object.
        """
        pointsFormat = [(pt.x, pt.y) for pt in perimeter_points]
        # CompoundGeometry is required
        geom = CompoundGeometry([Geometry(Polygon(pointsFormat), concrete_material)])

        # Add rebars
        for rebar in rebars:
            geom = add_bar(geom, rebar.area(), rebar_material, rebar.x, rebar.y, self.rebar_mesh_points)
        return geom

    def get_concrete_axial_moment(
            self, 
            geom: CompoundGeometry,
            moment: str = "m_x",
            figure_instead_of_points : bool = False
        ) -> list[InteractionPoint] | Image:
        """
        Compute the concrete section axial - moment/bending interaction diagram points or image.

        Args:
            geom (CompoundGeometry): Geometry of the concrete section.
            moment (str): Moment type ("m_x" or "m_y").
            figure_instead_of_points (bool): If True, return an image of the diagram instead of points.

        Returns:
            list[InteractionPoint] | Image: List of the axial - moment interaction diagram point values or the actual image.
        """

        # Concrete section
        conc_sec = ConcreteSection(geom)

        # Moment interaction diagram points
        if moment == "m_x":
            theta = 0.0
        elif moment == "m_y":
            theta = math.pi / 2
        mi_res = conc_sec.moment_interaction_diagram(theta = theta, progress_bar=False)
        nList, mList = mi_res.get_results_lists(moment)

        if figure_instead_of_points:
            if moment == "m_x":
                label_m = "Moment X (kN.m)"
            else:
                label_m = "Moment Y (kN.m)"
            return self.get_diagram_image(mList, nList, 1e-6, 1e-3, label_m, "Axial Load (kN)")
        
        return [InteractionPoint(n / 1e3, m / 1e6) for n, m in zip(nList, mList)]

    def get_diagram_image(self, 
                          valuesX : list[float], valuesY : list[float], 
                          scaleX: float = 1e-6, scaleY: float = 1e-3,
                          titleX: str = "Moment (kN.m)", titleY: str = "Axial Load (kN)") -> Image:
        """
        Generate an image of the axial-moment interaction diagram.

        Args:
            valuesX (list[float]): X-axis values (Axial Load).
            valuesY (list[float]): Y-axis values (Moment).
            scaleX (float): Scale factor for X-axis values.
            scaleY (float): Scale factor for Y-axis values.
            titleX (str): Title of the X-axis.
            titleY (str): Title of the Y-axis.
            title (str): Title of the plot.

        Returns:
            Image: The generated image of the diagram.
        """
        fig, ax = plt.subplots()

        # Scale results
        scaledValuesX = [v * scaleX for v in valuesX]
        scaledValuesY = [v * scaleY for v in valuesY]

        # Plot diagram
        ax.plot(scaledValuesX, scaledValuesY, "o-")

        # Labels
        ax.set_xlabel(titleX)
        ax.set_ylabel(titleY)

        # Save figure and return image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        return Image(data=buf.getvalue(), format="png")

    # Area properties tool-functions
    def rectangular_concrete_area_properties(
            self, 
            depth: float,
            width: float,
            dia_top: float,
            n_top: int,
            c_top: float,
            dia_bot: float,
            n_bot: int,
            c_bot: float,
            dia_side: float = 0.0,
            n_side: int = 0,
            c_side: float = 0.0,
            n_circle: int = 4,
            conc_elastic_modulus : float = 30.1e3,
            rebars_elastic_modulus : float = 200_000
        ) -> TransformedGrossProperties:
        """
        Compute rectangular concrete section geometric properties.

        Args:
            d: Concrete section depth
            b: Concrete section width
            dia_top: Diameter of the top reinforcing bars
            n_top: Number of top, equally spaced reinforcing bars
            c_top: Clear cover to the top reinforcing bars
            dia_bot: Diameter of the bottom reinforcing bars
            n_bot: Number of bottom, equally spaced reinforcing bars
            c_bot: Clear cover to the bottom reinforcing bars
            dia_side: Diameter of the side reinforcing bars. Defaults to ``0.0``.
            n_side: Number of side, equally spaced reinforcing bars. Defaults to ``0``.
            c_side: Clear cover to the side reinforcing bars. Defaults to ``0.0``.
            n_circle: Number of points used to discretise the circular reinforcing bars. Defaults to ``4``.
            conc_elastic_modulus (float): Elastic modulus of the concrete (MPa).
            rebars_elastic_modulus (float): Elastic modulus of the rebars (MPa).

        Returns:
            TransformedGrossProperties: Section geometric results.
        """

        geom = concrete_rectangular_section(
            d=depth,
            b=width,
            dia_top=dia_top,
            area_top = math.pi * (dia_top / 2) ** 2,
            n_top=n_top,
            c_top=c_top,
            dia_bot=dia_bot,
            area_bot = math.pi * (dia_bot / 2) ** 2,
            n_bot=n_bot,
            c_bot=c_bot,
            dia_side=dia_side,
            area_side = math.pi * (dia_side / 2) ** 2,
            n_side=n_side,
            c_side=c_side,
            n_circle=n_circle,
            conc_mat = self.get_concrete_material(conc_elastic_modulus),
            steel_mat = self.get_rebar_material(rebars_elastic_modulus),
        )

        # Concrete section
        conc_sec = ConcreteSection(geom)
        transformed_props : TransformedGrossProperties = conc_sec.get_transformed_gross_properties(elastic_modulus = conc_elastic_modulus)

        return transformed_props

    def arbitrary_concrete_area_properties(
            self, 
            perimeter_points: list[Point], 
            conc_elastic_modulus : float = 30.1e3,
            rebars : list[Rebar] = [],
            rebars_elastic_modulus : float = 200_000,
            align_to_centroid : bool = True,
        ) -> TransformedGrossProperties:
        """
        Compute arbitrary concrete section geometric properties.

        Args:
            perimeter_points (list[Point]): Section perimeter points (mm).
            conc_elastic_modulus (float): Elastic modulus of the concrete (MPa).
            rebars (list[Rebar]): Location and size of each rebar in the section (mm).
            rebars_elastic_modulus (float): Elastic modulus of the rebars (MPa).
            align_to_centroid (bool): Whether to align the section to its centroid, so that properties are calculated based on the centroid.

        Returns:
            TransformedGrossProperties: Section geometric results.
        """
        
        # Define materials
        concrete_material = self.get_concrete_material(conc_elastic_modulus)
        rebar_material = self.get_rebar_material(rebars_elastic_modulus)

        # Geometry from points
        geom = self.get_geometry_from_points(perimeter_points, rebars, concrete_material, rebar_material)
        
        if align_to_centroid:
            geom = geom.align_center()
        #geom.create_mesh(mesh_sizes=mesh_size)

        # Concrete section
        conc_sec = ConcreteSection(geom)
        transformed_props : TransformedGrossProperties = conc_sec.get_transformed_gross_properties(elastic_modulus=conc_elastic_modulus)

        #axes = conc_sec.plot_section()

        return transformed_props

    # Ultimate bending tool-functions
    def rectangular_concrete_bending_capacity(
            self, 
            depth: float,
            width: float,
            dia_top: float,
            n_top: int,
            c_top: float,
            dia_bot: float,
            n_bot: int,
            c_bot: float,
            dia_side: float = 0.0,
            n_side: int = 0,
            c_side: float = 0.0,
            conc_elastic_modulus : float = 30.1e3,
            f_c : float = 40,
            rebars_elastic_modulus : float = 200_000,
            f_y : float = 500,
            axial_load : float = 0.0,
        ) -> dict[str, str]:
        """
        Compute the rectangular concrete section bending capacities (Mx+, Mx- and My) under the given axial load.

        Args:
            d: Concrete section depth
            b: Concrete section width
            dia_top: Diameter of the top reinforcing bars
            n_top: Number of top, equally spaced reinforcing bars
            c_top: Clear cover to the top reinforcing bars
            dia_bot: Diameter of the bottom reinforcing bars
            n_bot: Number of bottom, equally spaced reinforcing bars
            c_bot: Clear cover to the bottom reinforcing bars
            dia_side: Diameter of the side reinforcing bars. Defaults to ``0.0``.
            n_side: Number of side, equally spaced reinforcing bars. Defaults to ``0``.
            c_side: Clear cover to the side reinforcing bars. Defaults to ``0.0``.
            conc_elastic_modulus (float): Elastic modulus of the concrete (MPa).
            f_c (float): Concrete compressive strength (MPa).
            rebars_elastic_modulus (float): Elastic modulus of the rebars (MPa).
            f_y (float): Yield strength of the rebar (MPa).
            axial_load (float): Axial load applied on the section (N).

        Returns:
            dict[str, str]: Dictionary with the bending capacities (Mx+, Mx-, My) and axial load.
        """

        geom = concrete_rectangular_section(
            d=depth,
            b=width,
            dia_top=dia_top,
            area_top = math.pi * (dia_top / 2) ** 2,
            n_top=n_top,
            c_top=c_top,
            dia_bot=dia_bot,
            area_bot = math.pi * (dia_bot / 2) ** 2,
            n_bot=n_bot,
            c_bot=c_bot,
            dia_side=dia_side,
            area_side = math.pi * (dia_side / 2) ** 2,
            n_side=n_side,
            c_side=c_side,
            n_circle=self.rebar_mesh_points,
            conc_mat = self.get_concrete_material(conc_elastic_modulus, f_c),
            steel_mat = self.get_rebar_material(rebars_elastic_modulus, f_y),
        )

        # Concrete section
        conc_sec = ConcreteSection(geom)

        sag_axial_res = conc_sec.ultimate_bending_capacity(n = axial_load)
        hog_axial_res = conc_sec.ultimate_bending_capacity(theta = math.pi, n = axial_load)
        weak_axial_res = conc_sec.ultimate_bending_capacity(theta = math.pi / 2, n = axial_load)
        
        return {
            "M_x+": f"{sag_axial_res.m_xy / 1e6:.1f} kN.m ",
            "M_x-": f"{hog_axial_res.m_xy / 1e6:.1f} kN.m ",
            "M_y": f"{weak_axial_res.m_xy / 1e6:.1f} kN.m ",
            "Axial load": f"{weak_axial_res.n / 1e3:.0f} kN"
        }

    # Axial - moment interaction diagram points tool-functions
    def rect_concrete_axial_moment_x_points(
            self, 
            depth: float,
            width: float,
            dia_top: float,
            n_top: int,
            c_top: float,
            dia_bot: float,
            n_bot: int,
            c_bot: float,
            dia_side: float = 0.0,
            n_side: int = 0,
            c_side: float = 0.0,
            conc_elastic_modulus : float = 30.1e3,
            f_c : float = 40,
            rebars_elastic_modulus : float = 200_000,
            f_y : float = 500
        ) -> list[InteractionPoint]:
        """
        Compute the rectangular concrete section axial - moment/bending (around X, usually the Major axis) interaction diagram points.

        Args:
            d: Concrete section depth
            b: Concrete section width
            dia_top: Diameter of the top reinforcing bars
            n_top: Number of top, equally spaced reinforcing bars
            c_top: Clear cover to the top reinforcing bars
            dia_bot: Diameter of the bottom reinforcing bars
            n_bot: Number of bottom, equally spaced reinforcing bars
            c_bot: Clear cover to the bottom reinforcing bars
            dia_side: Diameter of the side reinforcing bars. Defaults to ``0.0``.
            n_side: Number of side, equally spaced reinforcing bars. Defaults to ``0``.
            c_side: Clear cover to the side reinforcing bars. Defaults to ``0.0``.
            conc_elastic_modulus (float): Elastic modulus of the concrete (MPa).
            f_c (float): Concrete compressive strength (MPa).
            rebars_elastic_modulus (float): Elastic modulus of the rebars (MPa).
            f_y (float): Yield strength of the rebar (MPa).

        Returns:
            list[InteractionPoint]: List of the axial - moment interaction diagram point values.
        """
        geom = concrete_rectangular_section(
            d=depth,
            b=width,
            dia_top=dia_top,
            area_top = math.pi * (dia_top / 2) ** 2,
            n_top=n_top,
            c_top=c_top,
            dia_bot=dia_bot,
            area_bot = math.pi * (dia_bot / 2) ** 2,
            n_bot=n_bot,
            c_bot=c_bot,
            dia_side=dia_side,
            area_side = math.pi * (dia_side / 2) ** 2,
            n_side=n_side,
            c_side=c_side,
            n_circle=self.rebar_mesh_points,
            conc_mat = self.get_concrete_material(conc_elastic_modulus, f_c),
            steel_mat = self.get_rebar_material(rebars_elastic_modulus, f_y),
        )
        
        return self.get_concrete_axial_moment(geom, "m_x", False)

    def rect_concrete_axial_moment_y_points(
            self, 
            depth: float,
            width: float,
            dia_top: float,
            n_top: int,
            c_top: float,
            dia_bot: float,
            n_bot: int,
            c_bot: float,
            dia_side: float = 0.0,
            n_side: int = 0,
            c_side: float = 0.0,
            conc_elastic_modulus : float = 30.1e3,
            f_c : float = 40,
            rebars_elastic_modulus : float = 200_000,
            f_y : float = 500
        ) -> list[InteractionPoint]:
        """
        Compute the rectangular concrete section axial - moment/bending (around Y, usually the Minor axis) interaction diagram points.

        Args:
            d: Concrete section depth
            b: Concrete section width
            dia_top: Diameter of the top reinforcing bars
            n_top: Number of top, equally spaced reinforcing bars
            c_top: Clear cover to the top reinforcing bars
            dia_bot: Diameter of the bottom reinforcing bars
            n_bot: Number of bottom, equally spaced reinforcing bars
            c_bot: Clear cover to the bottom reinforcing bars
            dia_side: Diameter of the side reinforcing bars. Defaults to ``0.0``.
            n_side: Number of side, equally spaced reinforcing bars. Defaults to ``0``.
            c_side: Clear cover to the side reinforcing bars. Defaults to ``0.0``.
            conc_elastic_modulus (float): Elastic modulus of the concrete (MPa).
            f_c (float): Concrete compressive strength (MPa).
            rebars_elastic_modulus (float): Elastic modulus of the rebars (MPa).
            f_y (float): Yield strength of the rebar (MPa).

        Returns:
            list[InteractionPoint]: List of the axial - moment interaction diagram point values.
        """
        geom = concrete_rectangular_section(
            d=depth,
            b=width,
            dia_top=dia_top,
            area_top = math.pi * (dia_top / 2) ** 2,
            n_top=n_top,
            c_top=c_top,
            dia_bot=dia_bot,
            area_bot = math.pi * (dia_bot / 2) ** 2,
            n_bot=n_bot,
            c_bot=c_bot,
            dia_side=dia_side,
            area_side = math.pi * (dia_side / 2) ** 2,
            n_side=n_side,
            c_side=c_side,
            n_circle=self.rebar_mesh_points,
            conc_mat = self.get_concrete_material(conc_elastic_modulus, f_c),
            steel_mat = self.get_rebar_material(rebars_elastic_modulus, f_y),
        )

        return self.get_concrete_axial_moment(geom, "m_y", False)

    def rect_concrete_axial_moment_x_image(
            self, 
            depth: float,
            width: float,
            dia_top: float,
            n_top: int,
            c_top: float,
            dia_bot: float,
            n_bot: int,
            c_bot: float,
            dia_side: float = 0.0,
            n_side: int = 0,
            c_side: float = 0.0,
            conc_elastic_modulus : float = 30.1e3,
            f_c : float = 40,
            rebars_elastic_modulus : float = 200_000,
            f_y : float = 500
        ) -> Image:
        """
        Plots the rectangular concrete section axial - moment/bending (around X, usually the Major axis) interaction diagram points.

        Args:
            d: Concrete section depth
            b: Concrete section width
            dia_top: Diameter of the top reinforcing bars
            n_top: Number of top, equally spaced reinforcing bars
            c_top: Clear cover to the top reinforcing bars
            dia_bot: Diameter of the bottom reinforcing bars
            n_bot: Number of bottom, equally spaced reinforcing bars
            c_bot: Clear cover to the bottom reinforcing bars
            dia_side: Diameter of the side reinforcing bars,. Defaults to ``0.0``.
            n_side: Number of side, equally spaced reinforcing bars. Defaults to ``0``.
            c_side: Clear cover to the side reinforcing bars. Defaults to ``0.0``.
            conc_elastic_modulus (float): Elastic modulus of the concrete (MPa).
            f_c (float): Concrete compressive strength (MPa).
            rebars_elastic_modulus (float): Elastic modulus of the rebars (MPa).
            f_y (float): Yield strength of the rebar (MPa).

        Returns:
            Image: The generated image of the diagram.
        """
        geom = concrete_rectangular_section(
            d=depth,
            b=width,
            dia_top=dia_top,
            area_top = math.pi * (dia_top / 2) ** 2,
            n_top=n_top,
            c_top=c_top,
            dia_bot=dia_bot,
            area_bot = math.pi * (dia_bot / 2) ** 2,
            n_bot=n_bot,
            c_bot=c_bot,
            dia_side=dia_side,
            area_side = math.pi * (dia_side / 2) ** 2,
            n_side=n_side,
            c_side=c_side,
            n_circle=self.rebar_mesh_points,
            conc_mat = self.get_concrete_material(conc_elastic_modulus, f_c),
            steel_mat = self.get_rebar_material(rebars_elastic_modulus, f_y),
        )
        
        return self.get_concrete_axial_moment(geom, "m_x", True)

    def rect_concrete_axial_moment_y_image(
            self, 
            depth: float,
            width: float,
            dia_top: float,
            n_top: int,
            c_top: float,
            dia_bot: float,
            n_bot: int,
            c_bot: float,
            dia_side: float = 0.0,
            n_side: int = 0,
            c_side: float = 0.0,
            conc_elastic_modulus : float = 30.1e3,
            f_c : float = 40,
            rebars_elastic_modulus : float = 200_000,
            f_y : float = 500
        ) -> Image:
        """
        Plots the rectangular concrete section axial - moment/bending (around Y, usually the Minor axis) interaction diagram points.

        Args:
            d: Concrete section depth
            b: Concrete section width
            dia_top: Diameter of the top reinforcing bars
            n_top: Number of top, equally spaced reinforcing bars
            c_top: Clear cover to the top reinforcing bars
            n_bot: Number of bottom, equally spaced reinforcing bars
            dia_bot: Diameter of the bottom reinforcing bars
            c_bot: Clear cover to the bottom reinforcing bars
            dia_side: Diameter of the side reinforcing bars. Defaults to ``0.0``.
            n_side: Number of side, equally spaced reinforcing bars. Defaults to ``0``.
            c_side: Clear cover to the side reinforcing bars. Defaults to ``0.0``.
            conc_elastic_modulus (float): Elastic modulus of the concrete (MPa).
            f_c (float): Concrete compressive strength (MPa).
            rebars_elastic_modulus (float): Elastic modulus of the rebars (MPa).
            f_y (float): Yield strength of the rebar (MPa).

        Returns:
            Image: The generated image of the diagram.
        """
        geom = concrete_rectangular_section(
            d=depth,
            b=width,
            dia_top=dia_top,
            area_top = math.pi * (dia_top / 2) ** 2,
            n_top=n_top,
            c_top=c_top,
            dia_bot=dia_bot,
            area_bot = math.pi * (dia_bot / 2) ** 2,
            n_bot=n_bot,
            c_bot=c_bot,
            dia_side=dia_side,
            area_side = math.pi * (dia_side / 2) ** 2,
            n_side=n_side,
            c_side=c_side,
            n_circle=self.rebar_mesh_points,
            conc_mat = self.get_concrete_material(conc_elastic_modulus, f_c),
            steel_mat = self.get_rebar_material(rebars_elastic_modulus, f_y),
        )
        
        return self.get_concrete_axial_moment(geom, "m_y", True)


# This is for testing purposes only
if __name__ == "__main__":
    config = Config()
    #print("Config", config.data)
    print("mesh_size", config.mesh_size)
    print("concrete_material_density", config.concrete_material_density)
    print("concrete_rectangular_stress_block_alpha", config.concrete_rectangular_stress_block_alpha)
    print("concrete_rectangular_stress_block_gamma", config.concrete_rectangular_stress_block_gamma)
    print("concrete_rectangular_stress_block_ultimate_strain", config.concrete_rectangular_stress_block_ultimate_strain)
    print("concrete_flexural_tensile_strength", config.concrete_flexural_tensile_strength)
    print("rebar_mesh_points", config.rebar_mesh_points)
    print("rebar_material_density", config.rebar_material_density)
    print("rebar_material_fracture_strain", config.rebar_material_fracture_strain)

    PropsInst = Properties(config)

    # Plot materials
    conc_mat = PropsInst.get_concrete_material()
    #conc_mat.stress_strain_profile.plot_stress_strain(title=conc_mat.name)
    rebar_mat = PropsInst.get_rebar_material()
    #rebar_mat.stress_strain_profile.plot_stress_strain(title=rebar_mat.name)

    # Plot section
    geom = concrete_rectangular_section(
        d=600,
        b=400,
        dia_top=20,
        area_top = math.pi * (20 / 2) ** 2,
        n_top=3,
        c_top=30,
        dia_bot=24,
        area_bot = math.pi * (24 / 2) ** 2,
        n_bot=3,
        c_bot=30,
        dia_side=0.0,
        area_side = 0.0,
        n_side=0,
        c_side=0.0,
        n_circle=config.rebar_mesh_points,
        conc_mat = conc_mat,
        steel_mat = rebar_mat
    )
    print("Areas", math.pi * (20 / 2) ** 2, math.pi * (24 / 2) ** 2)
    geom.create_mesh(mesh_sizes=config.mesh_size)
    conc_sec = ConcreteSection(geom)
    #conc_sec.plot_section()

    # Define section points
    section_points = [
        Point(**{'x':0, 'y':0}),
        Point(**{'x':300, 'y':0}),
        Point(**{'x':300, 'y':500}),
        Point(**{'x':0, 'y':500})
    ]
    
    # Print section properties
    props = PropsInst.arbitrary_concrete_area_properties(section_points)
    #props.print_results(fmt=".3e")

    # Print moment capacity x
    moment_capacity = PropsInst.rectangular_concrete_bending_capacity(
        depth=500,
        width=300,
        dia_top=16,
        n_top=2,
        c_top=25,
        dia_bot=16,
        n_bot=2,
        c_bot=25
    )
    print("Moment Capacity x:", moment_capacity)

    # Print moment capacity y
    moment_capacity = PropsInst.rectangular_concrete_bending_capacity(
        depth=500,
        width=300,
        dia_top=16,
        n_top=2,
        c_top=25,
        dia_bot=16,
        n_bot=2,
        c_bot=25
    )
    print("Moment Capacity y:", moment_capacity)

    # Print axial-moment interaction points
    axial_moment_x_points = PropsInst.rect_concrete_axial_moment_x_points(
        depth=600,
        width=400,
        dia_top=16,
        n_top=2,
        c_top=25,
        dia_bot=16,
        n_bot=2,
        c_bot=25
    )
    print("Axial-Moment Interaction Points (X): [Axial Load (kN), Moment (kN.m)]")
    for point in axial_moment_x_points:
        print(f"{point.n:.2f}, {point.m:.2f}")

    # Print axial-moment interaction points
    axial_moment_y_points = PropsInst.rect_concrete_axial_moment_y_points(
        depth=600,
        width=400,
        dia_top=16,
        n_top=2,
        c_top=25,
        dia_bot=16,
        n_bot=2,
        c_bot=25
    )
    print("Axial-Moment Interaction Points (Y): [Axial Load (kN), Moment (kN.m)]")
    for point in axial_moment_y_points:
        print(f"{point.n:.2f}, {point.m:.2f}")
