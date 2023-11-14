import ast
import networkx as nx
import matplotlib.pyplot as plt

class ASTExplorer(ast.NodeVisitor):
    def __init__(self, graph):
        self.graph = graph
        self.current_parent = None

    def visit_ClassDef(self, node):
        class_name = node.name
        self.graph.add_node(class_name, type='class')
        if self.current_parent:
            self.graph.add_edge(self.current_parent, class_name)
        
        previous_parent = self.current_parent
        self.current_parent = class_name
        # Visit only the function definitions within the class
        for n in node.body:
            if isinstance(n, ast.FunctionDef):
                self.visit(n)
        self.current_parent = previous_parent

    def visit_FunctionDef(self, node):
        func_name = node.name
        full_func_name = f"{self.current_parent}.{func_name}" if self.current_parent else func_name
        self.graph.add_node(full_func_name, type='function')
        if self.current_parent:
            self.graph.add_edge(self.current_parent, full_func_name)

# Create a directed graph
graph = nx.DiGraph()

# Parse the Python script
filename = "tizNEAT2.py"  # Replace with your script's path
with open(filename, "r") as file:
    tree = ast.parse(file.read())

# Explore the AST and update the graph
explorer = ASTExplorer(graph)
explorer.visit(tree)

# Draw the graph
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
plt.show()
