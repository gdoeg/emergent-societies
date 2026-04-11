"""
Simple test runner to visualize agent relationships and trust system.

This script creates a small population of agents with varying cooperation
tendencies and runs them through multiple interaction cycles, printing
trust and interaction counts after each step.
"""

import random
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.agent import Agent
from simulation.environment import Environment

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('environment_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_relationships(agents, step_num):
    """Print all agent relationships in a clean, readable format."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}")
    print(f"{'='*60}")
    
    for agent in agents:
        if not agent.relationships:
            print(f"\nAgent {agent.agent_id}: (no interactions yet)")
            continue
            
        print(f"\nAgent {agent.agent_id}:")
        for other_id, rel_data in sorted(agent.relationships.items()):
            trust = rel_data['trust']
            interactions = rel_data['interaction_count']
            print(f"  -> {other_id} | trust={trust:.2f}, interactions={interactions}")


def main():
    """Run a simple environment simulation with relationship tracking."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create agents with varying cooperation tendencies
    agents = [
        Agent("Alice", resources=10, cooperation_tendency=0.9),   # Very cooperative
        Agent("Bob", resources=10, cooperation_tendency=0.7),     # Mostly cooperative
        Agent("Charlie", resources=10, cooperation_tendency=0.5), # Neutral
        Agent("Diana", resources=10, cooperation_tendency=0.3),   # Mostly defects
        Agent("Eve", resources=10, cooperation_tendency=0.1),     # Almost always defects
    ]
    
    print("AGENT RELATIONSHIP & TRUST SIMULATION")
    print("\nInitial Agent Configuration:")
    for agent in agents:
        print(f"  {agent.agent_id}: cooperation_tendency={agent.cooperation_tendency:.1f}, resources={agent.resources}")
    
    logger.info(f"Created {len(agents)} agents")
    
    # Initialize environment
    env = Environment(agents)
    
    # Import Gini computation for tracking
    from metrics.economics import compute_gini
    
    # Run simulation for 15 steps
    num_steps = 15
    
    for step in range(num_steps):
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting step {step + 1}/{num_steps}")
        logger.info(f"{'='*50}")
        
        env.step()
        print_relationships(agents, step + 1)
        
        # Log resource distribution and Gini after each step
        resources = [a.resources for a in agents]
        gini = compute_gini(resources)
        logger.info(f"Step {step + 1} completed - Gini: {gini:.4f}, Resources: {resources}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total steps completed: {env.cycle_count}")
    
    print("\nFinal resources:")
    resources = []
    for agent in agents:
        print(f"  {agent.agent_id}: {agent.resources}")
        resources.append(agent.resources)
    
    # Calculate and display Gini coefficient
    from metrics.economics import compute_gini
    final_gini = compute_gini(resources)
    print(f"\nWealth Distribution:")
    print(f"  Gini Coefficient: {final_gini:.4f}")
    print(f"  Total Wealth: {sum(resources)}")
    print(f"  Average Wealth: {sum(resources)/len(resources):.2f}")
    print(f"  Min Wealth: {min(resources)}")
    print(f"  Max Wealth: {max(resources)}")
    logger.info(f"Final Gini coefficient: {final_gini:.4f}")
    
    print("\nMost trusted relationships:")
    trust_pairs = []
    for agent in agents:
        for other_id, rel_data in agent.relationships.items():
            trust_pairs.append((agent.agent_id, other_id, rel_data['trust']))
    
    trust_pairs.sort(key=lambda x: x[2], reverse=True)
    for agent_id, other_id, trust in trust_pairs[:5]:
        print(f"  {agent_id} -> {other_id}: trust={trust:.2f}")


if __name__ == "__main__":
    main()
