#!/usr/bin/env python3
"""
Script to add Google-style docstrings to all functions in the vCache codebase.
"""

import ast
import os
import re
from typing import List, Tuple


class DocstringAdder(ast.NodeTransformer):
    """
    AST transformer to add docstrings to functions and classes.
    """
    
    def __init__(self, source_lines: List[str]):
        """
        Initialize the docstring adder.
        
        Args:
            source_lines: Lines of the source code.
        """
        self.source_lines = source_lines
        self.changes = []
    
    def visit_FunctionDef(self, node):
        """
        Visit function definitions and add docstrings if missing.
        
        Args:
            node: The AST node representing a function definition.
            
        Returns:
            The modified node.
        """
        self.generic_visit(node)
        
        # Check if function already has a docstring
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            # Function already has a docstring, check if it needs reformatting
            existing_docstring = node.body[0].value.value
            if not self._is_google_style_docstring(existing_docstring):
                # Reformat existing docstring
                new_docstring = self._create_google_style_docstring(node, existing_docstring)
                self.changes.append((node.lineno, 'reformat', new_docstring))
        else:
            # Function doesn't have a docstring, add one
            new_docstring = self._create_google_style_docstring(node)
            self.changes.append((node.lineno, 'add', new_docstring))
        
        return node
    
    def visit_ClassDef(self, node):
        """
        Visit class definitions and add docstrings if missing.
        
        Args:
            node: The AST node representing a class definition.
            
        Returns:
            The modified node.
        """
        self.generic_visit(node)
        
        # Check if class already has a docstring
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            # Class already has a docstring, check if it needs reformatting
            existing_docstring = node.body[0].value.value
            if not self._is_google_style_docstring(existing_docstring):
                # Reformat existing docstring
                new_docstring = self._create_class_docstring(node, existing_docstring)
                self.changes.append((node.lineno, 'reformat_class', new_docstring))
        else:
            # Class doesn't have a docstring, add one
            new_docstring = self._create_class_docstring(node)
            self.changes.append((node.lineno, 'add_class', new_docstring))
        
        return node
    
    def _is_google_style_docstring(self, docstring: str) -> bool:
        """
        Check if a docstring follows Google style.
        
        Args:
            docstring: The docstring to check.
            
        Returns:
            True if the docstring follows Google style.
        """
        # Simple check for Google style: contains "Args:" and "Returns:" sections
        return 'Args:' in docstring and 'Returns:' in docstring
    
    def _create_google_style_docstring(self, node, existing_docstring: str = None) -> str:
        """
        Create a Google-style docstring for a function.
        
        Args:
            node: The AST node representing the function.
            existing_docstring: Existing docstring to extract description from.
            
        Returns:
            The formatted Google-style docstring.
        """
        # Extract function name and arguments
        func_name = node.name
        args = []
        
        for arg in node.args.args:
            if arg.arg != 'self':  # Skip 'self' parameter
                args.append(arg.arg)
        
        # Create description
        if existing_docstring:
            # Try to extract description from existing docstring
            description = self._extract_description(existing_docstring)
        else:
            description = f"{func_name.replace('_', ' ').capitalize()}."
        
        # Build docstring
        docstring_parts = [f'        """', f'        {description}']
        
        if args:
            docstring_parts.append('')
            docstring_parts.append('        Args:')
            for arg in args:
                docstring_parts.append(f'            {arg}: Description of {arg}.')
        
        # Add Returns section if function has return annotation or return statements
        if node.returns or self._has_return_statement(node):
            docstring_parts.append('')
            docstring_parts.append('        Returns:')
            docstring_parts.append('            Description of return value.')
        
        docstring_parts.append('        """')
        
        return '\n'.join(docstring_parts)
    
    def _create_class_docstring(self, node, existing_docstring: str = None) -> str:
        """
        Create a Google-style docstring for a class.
        
        Args:
            node: The AST node representing the class.
            existing_docstring: Existing docstring to extract description from.
            
        Returns:
            The formatted Google-style docstring.
        """
        class_name = node.name
        
        if existing_docstring:
            description = self._extract_description(existing_docstring)
        else:
            description = f"{class_name.replace('_', ' ')} class."
        
        docstring_parts = [
            '    """',
            f'    {description}',
            '    """'
        ]
        
        return '\n'.join(docstring_parts)
    
    def _extract_description(self, docstring: str) -> str:
        """
        Extract description from existing docstring.
        
        Args:
            docstring: The existing docstring.
            
        Returns:
            The extracted description.
        """
        # Remove leading/trailing whitespace and quotes
        cleaned = docstring.strip().strip('"""').strip("'''").strip()
        
        # Take first line or sentence as description
        lines = cleaned.split('\n')
        if lines:
            first_line = lines[0].strip()
            if first_line:
                return first_line
        
        return "Function description."
    
    def _has_return_statement(self, node) -> bool:
        """
        Check if function has return statements.
        
        Args:
            node: The AST node representing the function.
            
        Returns:
            True if the function has return statements.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                return True
        return False


def process_file(file_path: str) -> bool:
    """
    Process a single Python file to add docstrings.
    
    Args:
        file_path: Path to the Python file to process.
        
    Returns:
        True if the file was modified.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Transform the AST
        transformer = DocstringAdder(lines)
        transformer.visit(tree)
        
        if not transformer.changes:
            return False
        
        # Apply changes to the content
        modified_lines = lines[:]
        
        # Sort changes by line number in reverse order to avoid offset issues
        transformer.changes.sort(key=lambda x: x[0], reverse=True)
        
        for line_no, change_type, docstring in transformer.changes:
            if change_type in ['add', 'add_class']:
                # Insert docstring after the function/class definition
                insert_line = line_no  # line_no is 1-based, but we want to insert after
                modified_lines.insert(insert_line, docstring)
            elif change_type in ['reformat', 'reformat_class']:
                # Find and replace existing docstring
                # This is more complex and would require more sophisticated parsing
                pass
        
        # Write back the modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_lines) + '\n')
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """
    Main function to process all Python files in the vCache project.
    """
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('/workspace/vCache'):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to process")
    
    modified_count = 0
    for file_path in python_files:
        print(f"Processing {file_path}")
        if process_file(file_path):
            modified_count += 1
    
    print(f"Modified {modified_count} files")


if __name__ == '__main__':
    main()