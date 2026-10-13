"""Data-driven unit tests for components.py and aerodynamics.py.

Add each known example as (args, kwargs, expected). For example:
CASES["components"]["some_function"] = [((1, 2), {}, 3)]
"""

import unittest

import aerodynamics
import components


MODULES = {
	"components": components,
	"aerodynamics": aerodynamics,
}

# Function cases use (positional_arguments, keyword_arguments, expected_result).
CASES = {
	"components": {},
	"aerodynamics": {},
}


class FunctionOutputTests(unittest.TestCase):
	def test_function_examples(self):
		for module_name, function_cases in CASES.items():
			module = MODULES[module_name]
			for function_name, cases in function_cases.items():
				function = getattr(module, function_name)
				for args, kwargs, expected in cases:
					with self.subTest(
						module=module_name,
						function=function_name,
						args=args,
						kwargs=kwargs,
					):
						self.assertEqual(function(*args, **kwargs), expected)


if __name__ == "__main__":
	unittest.main()
