import asyncio
from ui.control_panel import ControlPanel
from models.risk_assessment import RiskAssessmentModel
from deployment.hybrid_manager import HybridDeploymentManager
from memory.adaptive_memory import AdaptiveMemory
from deployment.sequence_coordinator import DeploymentCoordinator

async def main():
    # Initialize components
    control_panel = ControlPanel()
    risk_model = RiskAssessmentModel()
    deployment_manager = HybridDeploymentManager()
    memory_system = AdaptiveMemory('memory.db')
    coordinator = DeploymentCoordinator()
    
    # Start the system
    control_panel.render()
    await coordinator.execute_deployment()

if __name__ == "__main__":
    asyncio.run(main())

