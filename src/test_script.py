import os, sys  # Unused imports and multiple imports on one line

# Function with intentional linting issues
def example_function ( ):
  print( "Hello world!" )  # Improper spacing and missing docstring
  x = 42 # Unused variable

  for i in range (5 ): print( i)  # Improper spacing, single-line for loop

# Another function to test long lines and lack of formatting
def another_example(): print("This is an intentionally long line to check if the linter catches and formats it properly because it exceeds the recommended line length")

# Unused function
def unused_function():
    pass

example_function()
