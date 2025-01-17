# Intentional issues to test the linter

import os, sys  # Unused imports and multiple imports on one line

def lint_test_function( ):
    print( "This is a test function with bad spacing!" )  # Improper spacing

    for i in range( 5 ):print(i)  # Missing indentation and bad formatting

    x = 42  # Unused variable

    return "Lint Test"  # Unnecessary string concatenation and no newline at end of file
