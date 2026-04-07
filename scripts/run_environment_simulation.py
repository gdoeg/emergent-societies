"""
Simple test runner to visualize agent relationships and trust system.

This script creates a small population of agents with varying cooperation
tendencies and runs them through multiple interaction cycles, printing
trust and interaction counts after each step.
"""

import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.agent import Agent
from simulation.environment import Environment


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
    
    # Initialize environment
    env = Environment(agents)
    
    # Run simulation for 15 steps
    num_steps = 15
    
    for step in range(num_steps):
        env.step()
        print_relationships(agents, step + 1)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total steps completed: {env.cycle_count}")
    
    print("\nFinal resources:")
    for agent in agents:
        print(f"  {agent.agent_id}: {agent.resources}")
    
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
