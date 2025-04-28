# Concrete Properties MCP
# The main server module for the Concrete Properties MCP (Model Creation Protocol) server.
# This module handles the connection with the Concrete Properties and Concrete Properties Python libraries.

from mcp.server.fastmcp import FastMCP, Context
import logging
from config import *
import properties as Properties

# Initialize FastMCP server
mcp = FastMCP("concrete-properties", dependencies=["concreteproperties", "matplotlib"])

# Start logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('concrete_properties_mcp.log', encoding='utf-8')
    ])
logger = logging.getLogger('concrete_properties_mcp_server')
logger.info("Starting Concrete Properties MCP server")

# Load configuration
logger.info(f"Loading configuration...")
config = Config()
PropsInstance = Properties.Properties(config)

# Register the tools with the server
logger.info(f"Registering concrete properties tools...")
rectangular_concrete_area_properties = mcp.tool()(PropsInstance.rectangular_concrete_area_properties)
arbitrary_concrete_area_properties = mcp.tool()(PropsInstance.arbitrary_concrete_area_properties)

rectangular_concrete_bending_capacity = mcp.tool()(PropsInstance.rectangular_concrete_bending_capacity)
rect_concrete_axial_moment_x_points = mcp.tool()(PropsInstance.rect_concrete_axial_moment_x_points)
rect_concrete_axial_moment_y_points = mcp.tool()(PropsInstance.rect_concrete_axial_moment_y_points)
rect_concrete_axial_moment_x_image = mcp.tool()(PropsInstance.rect_concrete_axial_moment_x_image)
rect_concrete_axial_moment_y_image = mcp.tool()(PropsInstance.rect_concrete_axial_moment_y_image)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    