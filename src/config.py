# Concrete Properties MCP
# Config loader

import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('concrete_properties_mcp_server')

class Config:

    def __init__(self):
        # Load config
        configPath = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(configPath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Could not load config file: {str(e)}")
            # Return default config if loading fails
            self.data = {
                "server": {
                    "name": "Concrete Properties MCP",
                    "version": "1.0.0"
                },
                "options": {
                    "mesh_size": 50, # Mesh size (mm)
                    "concrete": {
                        "material_density": 2.4e-6, # Material density (kg/mm^3)
                        "rectangular_stress_block_alpha": 0.802, # Alpha value for rectangular stress block
                        "rectangular_stress_block_gamma": 0.89, # Gamma value for rectangular stress block
                        "rectangular_stress_block_ultimate_strain": 0.003, # Ultimate strain (mm/mm)
                        "flexural_tensile_strength": 3.4 # Flexural tensile strength (MPa)
                    },
                    "rebar": {
                        "mesh_points": 4, # Number of mesh points for rebars
                        "material_density": 7.85e-6, # Material density (kg/mm^3)
                        "material_fracture_strain": 0.05 # Material fracture strain
                    }
                }
            }

    @property
    def serverName(self) -> str:
        return self.data['server']['name']
    
    @property
    def mesh_size(self) -> float:
        return float(self.data['options']['mesh_size'])
    
    @property
    def concrete_material_density(self) -> float:
        return float(self.data['options']['concrete']['material_density'])
    
    @property
    def concrete_rectangular_stress_block_alpha(self) -> float:
        return float(self.data['options']['concrete']['rectangular_stress_block_alpha'])
    
    @property
    def concrete_rectangular_stress_block_gamma(self) -> float:
        return float(self.data['options']['concrete']['rectangular_stress_block_gamma'])
    
    @property
    def concrete_rectangular_stress_block_ultimate_strain(self) -> float:
        return float(self.data['options']['concrete']['rectangular_stress_block_ultimate_strain'])
    
    @property
    def concrete_flexural_tensile_strength(self) -> float:
        return float(self.data['options']['concrete']['flexural_tensile_strength'])

    @property
    def rebar_mesh_points(self) -> int:
        return int(self.data['options']['rebar']['mesh_points'])

    @property
    def rebar_material_density(self) -> float:
        return float(self.data['options']['rebar']['material_density'])

    @property
    def rebar_material_fracture_strain(self) -> float:
        return float(self.data['options']['rebar']['material_fracture_strain'])


# This is for testing purposes only
if __name__ == "__main__":
    config = Config()
    print(f"Server Name: {config.serverName}")
    print(f"Server Version: {config.serverVersion}")
