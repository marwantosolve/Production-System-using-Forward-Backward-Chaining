import re
import os
from collections import defaultdict
import sys


class ProductionSystem:
  def __init__(self):
    self.rules = []
    self.facts = {}
    # Maps conclusion predicates to rule indices
    self.rule_indices = defaultdict(list)
    self.output_file = None

  def set_output_file(self, filename):
    """Set the output file for storing reasoning process."""
    self.output_file = open(filename, 'w')

  def close_output_file(self):
    """Close the output file if it's open."""
    if self.output_file and not self.output_file.closed:
      self.output_file.close()

  def log(self, message):
    """Print to console and write to file if available."""
    print(message)
    if self.output_file and not self.output_file.closed:
      self.output_file.write(message + "\n")

  def parse_rules_file(self, filename):
    """Parse rules from a file and store them in a structured format."""
    if not os.path.exists(filename):
      raise FileNotFoundError(f"Rules file not found: {filename}")

    with open(filename, 'r') as file:
      for i, line in enumerate(file):
        line = line.strip()
        if not line:
          continue

        # Extract IF-THEN parts
        match = re.match(r'IF (.+) THEN (.+)', line)
        if match:
          condition, conclusion = match.groups()
          # Split conditions by AND
          conditions = [c.strip() for c in condition.split('AND')]
          # Parse each condition
          parsed_conditions = []
          for cond in conditions:
            parsed_cond = self.parse_condition(cond)
            if parsed_cond:
              parsed_conditions.append(parsed_cond)

          # Parse conclusion
          parsed_conclusion = self.parse_conclusion(conclusion)
          if parsed_conclusion:
            rule = {
                'conditions': parsed_conditions,
                'conclusion': parsed_conclusion,
                'original': line  # Store original rule text for reference
            }
            self.rules.append(rule)
            # Update rule index for backward chaining
            self.rule_indices[parsed_conclusion['predicate']].append(
                len(self.rules) - 1)

  def parse_facts_file(self, filename):
    """Parse facts from a file and store them in a structured format."""
    if not os.path.exists(filename):
      raise FileNotFoundError(f"Facts file not found: {filename}")

    with open(filename, 'r') as file:
      for line in file:
        line = line.strip()
        if not line or line.startswith('#'):
          continue

        # Handle different fact formats
        if '=' in line:
          # Format: key = value
          parts = line.split('=', 1)  # Split on first '=' only
          if len(parts) != 2:
            print(f"Warning: skipping complex fact: {line}")
            continue
          key, value = parts
          self.facts[key.strip()] = value.strip()
        elif 'is' in line:
          # Format: attribute is value
          parts = line.split('is', 1)  # Split on first 'is' only
          if len(parts) != 2:
            print(f"Warning: skipping complex fact: {line}")
            continue
          attribute, value = parts
          self.facts[attribute.strip()] = value.strip()
        else:
          # Format: single predicate (e.g., "skin_smell")
          self.facts[line] = True

  def parse_condition(self, condition):
    """Parse a single condition."""
    condition = condition.strip()

    # Check if this is an OR condition first
    if ' OR ' in condition:
      or_conditions = []
      for part in condition.split(' OR '):
        parsed_part = self.parse_condition(part.strip())
        if parsed_part:
          or_conditions.append(parsed_part)
      return {
          'type': 'or',
          'conditions': or_conditions
      }

    # Equality condition (e.g., "diameter = 7")
    if '=' in condition:
      parts = condition.split('=')
      if len(parts) != 2:  # Handle case where there are multiple '=' signs
        print(f"Warning: complex equality condition: {condition}")
        attr = parts[0].strip()
        value = '='.join(parts[1:]).strip()
      else:
        attr, value = parts
      return {
          'type': 'equality',
          'attribute': attr.strip(),
          'value': value.strip()
      }

    # Comparison condition (e.g., "diameter < 10")
    elif '<' in condition:
      parts = condition.split('<')
      if len(parts) != 2:
        print(f"Warning: complex comparison condition: {condition}")
        return None
      attr, value = parts
      return {
          'type': 'comparison',
          'operator': '<',
          'attribute': attr.strip(),
          'value': float(value.strip())
      }
    elif '>' in condition:
      parts = condition.split('>')
      if len(parts) != 2:
        print(f"Warning: complex comparison condition: {condition}")
        return None
      attr, value = parts
      return {
          'type': 'comparison',
          'operator': '>',
          'attribute': attr.strip(),
          'value': float(value.strip())
      }

    # Predicate condition (e.g., "color is red")
    elif 'is' in condition:
      parts = condition.split('is', 1)  # Split on first 'is' only
      if len(parts) != 2:
        print(f"Warning: complex predicate condition: {condition}")
        return None
      attr, value = parts
      return {
          'type': 'predicate',
          'attribute': attr.strip(),
          'value': value.strip()
      }

    # Simple condition (e.g., "skin_smell")
    else:
      return {
          'type': 'simple',
          'predicate': condition
      }

  def parse_conclusion(self, conclusion):
    """Parse a rule conclusion."""
    conclusion = conclusion.strip()

    # Assignment conclusion (e.g., "fruit is banana")
    if 'is' in conclusion:
      parts = conclusion.split('is', 1)  # Split on first 'is' only
      if len(parts) != 2:
        print(f"Warning: complex conclusion: {conclusion}")
        return None
      attr, value = parts
      return {
          'type': 'assignment',
          'predicate': attr.strip(),
          'value': value.strip()
      }

    # Simple conclusion (e.g., "citrus_fruit")
    else:
      return {
          'type': 'simple',
          'predicate': conclusion
      }

  def condition_satisfied(self, condition):
    """Check if a condition is satisfied given the current facts."""
    if condition['type'] == 'equality':
      attr = condition['attribute']
      value = condition['value']
      return attr in self.facts and str(self.facts[attr]) == value

    elif condition['type'] == 'comparison':
      attr = condition['attribute']
      value = condition['value']
      op = condition['operator']

      if attr not in self.facts:
        return False

      fact_value = float(self.facts[attr])
      if op == '<':
        return fact_value < value
      elif op == '>':
        return fact_value > value

    elif condition['type'] == 'predicate':
      attr = condition['attribute']
      value = condition['value']
      return attr in self.facts and self.facts[attr] == value

    elif condition['type'] == 'or':
      # Debug the OR condition
      for subcond in condition['conditions']:
        if self.condition_satisfied(subcond):
          return True

      # Additional check for assignment predicates in OR conditions
      # This is needed for rules like "fruit is lemon OR fruit is orange"
      for subcond in condition['conditions']:
        if subcond['type'] == 'predicate':
          attr = subcond['attribute']
          value = subcond['value']
          # Check if we have a fact like "fruit": "orange" that matches the condition
          if attr in self.facts and self.facts[attr] == value:
            return True

      return False

    elif condition['type'] == 'simple':
      predicate = condition['predicate']
      return predicate in self.facts and self.facts[predicate] is True

    return False

  def apply_conclusion(self, conclusion):
    """Apply a rule conclusion to update the facts."""
    if conclusion['type'] == 'assignment':
      predicate = conclusion['predicate']
      value = conclusion['value']
      self.facts[predicate] = value
      return True
    elif conclusion['type'] == 'simple':
      predicate = conclusion['predicate']
      self.facts[predicate] = True
      return True
    return False

  def forward_chaining(self, goal=None):
    """Implement forward chaining algorithm."""
    self.log("\n=== Forward Chaining ===")
    self.log("Initial facts: " + str(self.facts))

    changes = True
    cycle = 1

    while changes:
      changes = False
      self.log(f"\nCycle {cycle}:")

      for i, rule in enumerate(self.rules):
        # Check if all conditions are satisfied
        all_conditions_satisfied = True
        for cond in rule['conditions']:
          if not self.condition_satisfied(cond):
            all_conditions_satisfied = False
            break

        if all_conditions_satisfied:
          conclusion = rule['conclusion']
          predicate = conclusion['predicate']

          # Check if this conclusion would add new information
          if (predicate not in self.facts) or (conclusion['type'] == 'assignment' and self.facts[predicate] != conclusion['value']):
            self.apply_conclusion(conclusion)
            changes = True

            if conclusion['type'] == 'assignment':
              self.log(
                  f"Rule {i+1} fired: {predicate} is {conclusion['value']}")
            else:
              self.log(f"Rule {i+1} fired: {predicate}")

      if changes:
        self.log("Current facts: " + str(self.facts))
      cycle += 1

    self.log("\nNo more rules can be applied.")
    self.log("Final facts: " + str(self.facts))

    # Check if goal was reached
    if goal:
      if ' is ' in goal:
        attr, value = goal.split(' is ')
        goal_reached = attr in self.facts and self.facts[attr] == value
      else:
        goal_reached = goal in self.facts and self.facts[goal] is True

      if goal_reached:
        self.log(f"Goal '{goal}' was reached.")
        return True
      else:
        self.log(f"Goal '{goal}' was NOT reached.")
        return False

    return True

  def backward_chaining(self, goal):
    """Implement backward chaining algorithm."""
    self.log("\n=== Backward Chaining ===")
    self.log("Initial facts: " + str(self.facts))
    self.log(f"Goal: {goal}")

    # Create a stack to keep track of goals to prove
    visited = set()
    cycle = 1

    result = self._backward_chaining_recursive(goal, visited, cycle)

    if result:
      self.log(f"\nGoal '{goal}' was proven!")
    else:
      self.log(f"\nGoal '{goal}' could NOT be proven.")

    self.log("Final facts: " + str(self.facts))
    return result

  def _backward_chaining_recursive(self, goal, visited, cycle, depth=0):
    """Recursive implementation of backward chaining."""
    indent = "  " * depth
    self.log(f"\nCycle {cycle} (Depth {depth}):")
    self.log(f"{indent}Trying to prove: {goal}")

    # Check if goal is already known
    if ' is ' in goal and ' OR ' not in goal:  # Only try to split non-OR goals
      attr, value = goal.split(' is ')
      if attr in self.facts and self.facts[attr] == value:
        self.log(f"{indent}Goal '{goal}' is already a known fact.")
        return True
    else:
      if goal in self.facts and self.facts[goal] is True:
        self.log(f"{indent}Goal '{goal}' is already a known fact.")
        return True

    # Avoid circular reasoning
    if goal in visited:
      self.log(f"{indent}Avoiding circular reasoning for goal: {goal}")
      return False

    visited.add(goal)

    # Special case for OR conditions
    if " OR " in goal:
      self.log(f"{indent}Processing OR condition: {goal}")
      subgoals = []
      # Handle complex OR conditions that might include "is" statements
      for part in goal.split(" OR "):
        part = part.strip()
        subgoals.append(part)

      for subgoal in subgoals:
        self.log(f"{indent}Trying subgoal of OR: {subgoal}")
        if self._backward_chaining_recursive(subgoal, visited.copy(), cycle + 1, depth + 1):
          self.log(f"{indent}OR condition satisfied with: {subgoal}")
          return True
      self.log(f"{indent}No subgoal in OR condition could be proven")
      return False

    # Find rules with this goal as conclusion
    goal_predicate = goal.split(' is ')[0] if ' is ' in goal else goal
    rule_indices = self.rule_indices.get(goal_predicate, [])

    if not rule_indices:
      self.log(f"{indent}No rules found with conclusion: {goal_predicate}")
      return False

    self.log(
        f"{indent}Found {len(rule_indices)} rules that could lead to {goal_predicate}")

    for i in rule_indices:
      rule = self.rules[i]
      self.log(f"{indent}Trying rule {i+1}")

      # Check if rule conclusion matches the goal
      conclusion = rule['conclusion']

      # For assignment conclusions, check if attribute and value match
      matches_goal = False
      if conclusion['type'] == 'assignment':
        if ' is ' in goal:
          goal_parts = goal.split(' is ')
          if len(goal_parts) == 2:  # Make sure we have exactly two parts
            goal_attr, goal_value = goal_parts
            matches_goal = (conclusion['predicate'] == goal_attr.strip() and
                            conclusion['value'] == goal_value.strip())
        else:
          matches_goal = False
      else:  # Simple conclusions
        matches_goal = (conclusion['predicate'] == goal)

      if not matches_goal:
        self.log(f"{indent}Rule conclusion doesn't match goal")
        continue

      # Try to satisfy all conditions
      all_conditions_satisfied = True
      for condition in rule['conditions']:
        if self.condition_satisfied(condition):
          self.log(
              f"{indent}  Condition already satisfied: {self._condition_to_string(condition)}")
          continue

        # For complex conditions, we need to break them down
        subcondition_satisfied = False

        if condition['type'] == 'predicate':
          subgoal = f"{condition['attribute']} is {condition['value']}"
          self.log(f"{indent}  Need to prove subcondition: {subgoal}")
          subcondition_satisfied = self._backward_chaining_recursive(
              subgoal, visited.copy(), cycle + 1, depth + 1)

        elif condition['type'] == 'simple':
          subgoal = condition['predicate']
          self.log(f"{indent}  Need to prove subcondition: {subgoal}")
          subcondition_satisfied = self._backward_chaining_recursive(
              subgoal, visited.copy(), cycle + 1, depth + 1)

        elif condition['type'] == 'or':
          self.log(f"{indent}  Evaluating OR condition")
          # Build the complex OR condition
          or_goal = " OR ".join([self._condition_to_goal(subcond)
                                for subcond in condition['conditions']])
          self.log(f"{indent}  Attempting to prove OR condition: {or_goal}")
          subcondition_satisfied = self._backward_chaining_recursive(
              or_goal, visited.copy(), cycle + 1, depth + 1)

        elif condition['type'] == 'equality' or condition['type'] == 'comparison':
          # These are fact-based conditions, they must be in facts to be satisfied
          if not self.condition_satisfied(condition):
            self.log(
                f"{indent}  Cannot satisfy fact-based condition: {self._condition_to_string(condition)}")
            subcondition_satisfied = False

        if not subcondition_satisfied:
          self.log(
              f"{indent}  Failed to satisfy condition: {self._condition_to_string(condition)}")
          all_conditions_satisfied = False
          break

      if all_conditions_satisfied:
        # Apply conclusion if all conditions are satisfied
        self.log(
            f"{indent}All conditions satisfied for rule {i+1}, applying conclusion")
        self.apply_conclusion(conclusion)

        # Print current facts after applying conclusion
        self.log(f"{indent}Current facts: {self.facts}")
        return True

    # If we've tried all rules and none worked
    self.log(f"{indent}Could not prove {goal} using any rules")
    return False

  def _condition_to_goal(self, condition):
    """Convert a condition to a goal format for backward chaining."""
    if condition['type'] == 'predicate':
      return f"{condition['attribute']} is {condition['value']}"
    elif condition['type'] == 'simple':
      return condition['predicate']
    else:
      return self._condition_to_string(condition)

  def _condition_to_string(self, condition):
    """Convert a condition to a readable string for debugging."""
    if condition['type'] == 'equality':
      return f"{condition['attribute']} = {condition['value']}"

    elif condition['type'] == 'comparison':
      return f"{condition['attribute']} {condition['operator']} {condition['value']}"

    elif condition['type'] == 'predicate':
      return f"{condition['attribute']} is {condition['value']}"

    elif condition['type'] == 'or':
      subconds = [self._condition_to_string(
          c) for c in condition['conditions']]
      return " OR ".join(subconds)

    elif condition['type'] == 'simple':
      return condition['predicate']

    return str(condition)


def main():
  # Get file paths from command line arguments or use default
  if len(sys.argv) > 2:
    rules_file = sys.argv[1]
    facts_file = sys.argv[2]
  else:
    rules_file = input("Enter the path to rules file: ")
    facts_file = input("Enter the path to facts file: ")

  # Check if files exist
  if not os.path.exists(rules_file):
    print(f"Error: Rules file '{rules_file}' does not exist.")
    return

  if not os.path.exists(facts_file):
    print(f"Error: Facts file '{facts_file}' does not exist.")
    return

  ps = ProductionSystem()
  # Change to proof.txt as requested in the assignment
  ps.set_output_file("proof.txt")

  # Parse rules and facts
  try:
    ps.parse_rules_file(rules_file)
    ps.parse_facts_file(facts_file)
    ps.log("Successfully parsed rules and facts files.")
  except Exception as e:
    ps.log(f"Error parsing files: {e}")
    ps.close_output_file()
    return

  # Extract goal from facts.txt
  goal = None
  with open(facts_file, 'r') as file:
    for line in file:
      if line.startswith('#goal'):
        goal = line.strip().split('#goal ')[1].strip()
        break

  if not goal:
    ps.log("Warning: No goal specified in facts file! Proceeding with forward chaining only.")
    # Run forward chaining without a specific goal
    ps.forward_chaining()
    ps.close_output_file()
    return

  # Run forward chaining
  ps.forward_chaining(goal)

  # Reset facts to initial state for backward chaining
  ps.facts = {}  # Clear facts
  ps.parse_facts_file(facts_file)  # Re-parse facts file

  # Run backward chaining
  ps.backward_chaining(goal)
  ps.close_output_file()


if __name__ == "__main__":
  main()
