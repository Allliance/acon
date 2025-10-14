import fire
import re
import math
import ast
from productive_agents.env.officebench.apps import APPS_ROOT

# Demo provides documentation on how to use the calculator functionality
DEMO = (
    'calculate a mathematical expression written in python syntax. Any math module functions are allowed.\n'
    '{"app": "calculator", "action": "calculate", "expression": [MATH_EXPRESSION_TO_EVALUATE]}'
)

def construct_action(work_dir, args: dict, py_file_path=f'{APPS_ROOT}/calculator_app/calculator_calculate.py'):
    return "python3 {} --expression '{}'".format(
        py_file_path, args["expression"]
    )

def calculator_calculate(expression):
    """
    Evaluate a mathematical expression and return the result.
    
    Args:
        expression (str): The mathematical expression to evaluate.
        
    Returns:
        float or int: The result of the evaluation.
    """
    try:
        # Import math module for all available math functions
        import math
        
        # Get all available math functions
        math_functions = {name: getattr(math, name) for name in dir(math) 
                         if not name.startswith('_') and callable(getattr(math, name))}
        
        # Add math constants
        math_constants = {name: getattr(math, name) for name in dir(math)
                         if not name.startswith('_') and not callable(getattr(math, name))}
        
        # Create a safe evaluation environment with math functions and constants
        safe_dict = {**math_functions, **math_constants, 'min': min, 'max': max, 'sum': sum, 'abs': abs, 'sorted': sorted, 'len': len}
        
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        # Add support for list arguments
        # First, check if the expression is safe using ast.parse
        try:
            ast.parse(expression)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in expression: {str(e)}")
        
        # Check for potentially unsafe operations or attributes
        if any(unsafe_word in expression for unsafe_word in 
               ['import', 'exec', 'eval', 'getattr', 'setattr', 'delattr', 
                'compile', 'open', '__', 'globals', 'locals', 'subprocess']):
            raise ValueError("Potentially unsafe operations detected in expression")
        
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return result
    except Exception as e:
        return f"Error: [calculator] {str(e)}"

def main(expression):
    """
    Main function to handle calculator evaluation from the command line.
    
    Args:
        expression (str): The mathematical expression to evaluate.
        
    Returns:
        str: Observation message with the result.
    """
    try:
        result = calculator_calculate(expression)
        return f"OBSERVATION: {result}"
    except Exception as e:
        return f"OBSERVATION: Error calculating expression: {str(e)}"

if __name__ == '__main__':
    fire.Fire(main)