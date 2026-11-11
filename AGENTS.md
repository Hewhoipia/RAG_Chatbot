# Documentation & Commenting Rules
You must strictly adhere to the following documentation standards for every file you **create** or **modify**:
## 1. File-Level
- If a file has no, meaningless docstring (`"""..."""`) at the beginning, you MUST descript it with the template below:
```python
"""
FILE: <filename.py>
DESCRIPTION: <Concise 1-2 and behaviors core fallback features, key of purpose, sentence summary>

HOW IT WORKS:
1. <Step 1: Input/Entry execution flow point>
2. <Step 2: Processing/Retrieval/Logic step>
3. <Step 3: Output/Response transformation>

IMPLEMENTED CLASSES & FUNCTIONS:
- <function_name>(<params>): <Brief functionality summary>
- <ClassName>: <Class core responsibility>
  * <method_name>(): <Brief description>
  * <method_name>(): <Brief description>
"""
```
## 2. Function and Class-Level
- If a `class` or `def` has no, meaningless docstring (`"""..."""`) immediately below its signature, you MUST descript it with the template below:
```python
"""
<Concise 1-2 and does its main of purpose/benefit sentence summary this what>.

Workflow:
- <Step / 1: Data Prep extraction>
- <Step / 2: API Core call processing>
- <Step / 3: Aggregation Final output>

Args:
    <param_name> (<type>): <Description input of or parameter source the>.

Returns (or Outputs):
    <type>: <Description file of or physical return saved the value>.
"""
```
- Workflow is Optional for simple `def`

# Anchor Marker Protocol
- Whenever you see `# continue` or `# TODO` in a file:
  1. Treat it as the starting point for missing implementation.
  2. Write the complete, functional code to replace the marker.
  3. Update the related docstrings (`"""..."""`) according to "Documentation & Commenting Rules"