"""
COMP-5710 Workshop 6: Data Flow Analysis (code)
Author: Jacob Murrah
Date: 10/14/2025
NOTE: See data-structure-usage.txt for data structure usage details.
"""

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class DataFlowAnalyzer:
    """Extract simple data-flow information from a Python source file."""

    def __init__(self, file_name: str) -> None:
        self.source = Path(file_name).read_text()
        self.tree = ast.parse(self.source)  # parses the source code into an AST
        self.assignment_map: Dict[str, Dict[str, ast.AST]] = {}

    # method to extract the parse tree
    def get_parse_tree(self) -> ast.AST:
        """Return the module AST (requirement: extract the parse tree)."""
        return self.tree

    # method to parse assignments for the parse tree
    def parse_assignments(self) -> None:
        """Populate assignment map per scope (requirement: parse assignments)."""
        collector = _AssignmentCollector()
        collector.visit(self.tree)
        self.assignment_map = collector.assignment_map

    # method to extract assignment operations
    def extract_assignment_operations(
        self, scope: Optional[str] = None
    ) -> List[ast.Assign]:
        """Collect assignments whose value is a binary operation (requirement)."""
        assignments: List[ast.Assign] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
                if scope is None:
                    assignments.append(node)
                else:
                    parent_scope = _enclosing_scope_name(node, self.tree)
                    if parent_scope == scope:
                        assignments.append(node)

        return assignments

    # method to generate the flow
    def generate_flow(self, tracked_value: Any) -> str:
        """Build the taint flow string (requirement: generate the flow)."""
        module_env: Dict[str, Any] = {}
        for name, value_node in self.assignment_map.get("module", {}).items():
            try:
                module_env[name] = self._evaluate_literal(value_node)
            except ValueError:
                continue

        origin_name = self._find_symbol_by_value(tracked_value, module_env)
        if origin_name is None:
            raise ValueError(
                f"Could not resolve an assignment for value {tracked_value!r}"
            )

        flow_nodes: List[Any] = [tracked_value, origin_name]
        call_assign = self._find_call_using_name(origin_name)
        call = (
            call_assign.value
            if isinstance(call_assign, ast.Assign)
            and isinstance(call_assign.value, ast.Call)
            else None
        )
        if call is None:
            return self._format_flow(flow_nodes)

        func_name = self._callable_name(call.func)
        func_def = self._get_function_def(func_name) if func_name else None
        if func_def is None:
            return self._format_flow(flow_nodes)

        arg_values = [self._evaluate_expression(arg, module_env) for arg in call.args]
        param_names = [arg.arg for arg in func_def.args.args]
        param_env = dict(zip(param_names, arg_values))

        tainted_params = [
            name for name in param_names if param_env.get(name) == tracked_value
        ]
        flow_nodes.extend(tainted_params)

        tail = self._trace_function_flow(func_def, param_env, set(tainted_params))
        flow_nodes.extend(tail)

        return self._format_flow(flow_nodes)

    def run_pipeline(self, tracked_value: Any) -> str:
        self.get_parse_tree()
        self.parse_assignments()
        self.extract_assignment_operations()
        return self.generate_flow(tracked_value)

    def _evaluate_literal(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self._evaluate_literal(node.operand)
            return -operand
        if isinstance(node, ast.Tuple):
            return tuple(self._evaluate_literal(elt) for elt in node.elts)
        raise ValueError(f"Unsupported literal node: {ast.dump(node)}")

    def _evaluate_expression(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise KeyError(f"Unknown name: {node.id}")
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._evaluate_expression(node.operand, env)
        if isinstance(node, ast.BinOp):
            left = self._evaluate_expression(node.left, env)
            right = self._evaluate_expression(node.right, env)
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            raise ValueError(f"Unsupported binary operation: {ast.dump(node.op)}")
        if isinstance(node, ast.Compare):
            return self._evaluate_compare(node, env)
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    def _evaluate_compare(self, node: ast.Compare, env: Dict[str, Any]) -> bool:
        left = self._evaluate_expression(node.left, env)
        right = self._evaluate_expression(node.comparators[0], env)
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.LtE):
            return left <= right
        raise ValueError(f"Unsupported comparison operator: {ast.dump(op)}")

    def _find_symbol_by_value(
        self, tracked_value: Any, env: Dict[str, Any]
    ) -> Optional[str]:
        for name, value in env.items():
            if value == tracked_value:
                return name
        return None

    def _find_call_using_name(self, name: str) -> Optional[ast.Assign]:
        for node in self.tree.body:
            assign = self._match_call_assignment(node)
            if assign is not None and isinstance(assign.value, ast.Call):
                if any(
                    self._expression_contains_any(arg, {name})
                    for arg in assign.value.args
                ):
                    return assign
        return None

    def _match_call_assignment(self, node: ast.AST) -> Optional[ast.Assign]:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            return node
        if isinstance(node, ast.If):
            for stmt in node.body:
                result = self._match_call_assignment(stmt)
                if result is not None:
                    return result
        return None

    def _callable_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _trace_function_flow(
        self,
        func_def: ast.FunctionDef,
        env: Dict[str, Any],
        tainted: Set[str],
    ) -> List[Any]:
        local_env = dict(env)
        tainted_names = set(tainted)
        flow: List[Any] = []
        for stmt in func_def.body:
            additions, returned = self._process_function_statement(
                stmt, local_env, tainted_names
            )
            if additions:
                flow.extend(additions)
            if returned:
                break
        return flow

    def _process_function_statement(
        self,
        stmt: ast.stmt,
        env: Dict[str, Any],
        tainted: Set[str],
        control_tainted: bool = False,
    ) -> Tuple[List[Any], bool]:
        if isinstance(stmt, ast.Assign):
            additions = self._process_assignment(stmt, env, tainted, control_tainted)
            return additions, False
        if isinstance(stmt, ast.If):
            branch = (
                stmt.body if self._evaluate_expression(stmt.test, env) else stmt.orelse
            )
            branch_flow: List[Any] = []
            branch_taint = control_tainted or self._expression_contains_any(
                stmt.test, tainted
            )
            for inner in branch:
                additions, returned = self._process_function_statement(
                    inner, env, tainted, branch_taint
                )
                if additions:
                    branch_flow.extend(additions)
                if returned:
                    return branch_flow, True
            return branch_flow, False
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return [], True
            return [], True
        return [], False

    def _process_assignment(
        self,
        node: ast.Assign,
        env: Dict[str, Any],
        tainted: Set[str],
        control_tainted: bool,
    ) -> List[Any]:
        additions: List[Any] = []
        for target_node, value_node in _assignment_pairs(node.targets, node.value):
            target_name = _target_to_name(target_node)
            if target_name is None:
                continue
            value = self._evaluate_expression(value_node, env)
            env[target_name] = value
            contains_taint = self._expression_contains_any(value_node, tainted)
            influenced = contains_taint or control_tainted
            if influenced:
                if target_name not in tainted:
                    tainted.add(target_name)
                if contains_taint:
                    additions.append(target_name)
                else:
                    additions.extend([value, target_name])
        return additions

    def _expression_contains_any(self, node: ast.AST, targets: Set[str]) -> bool:
        if not targets:
            return False
        return any(
            isinstance(child, ast.Name) and child.id in targets
            for child in ast.walk(node)
        )

    def _format_flow(self, nodes: List[Any]) -> str:
        return "->".join(str(node) for node in nodes)

    def _get_function_def(self, name: str) -> Optional[ast.FunctionDef]:
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None


def _assignment_pairs(
    targets: List[ast.expr], value: ast.AST
) -> List[Tuple[ast.expr, ast.AST]]:
    pairs: List[Tuple[ast.expr, ast.AST]] = []
    if (
        len(targets) == 1
        and isinstance(targets[0], ast.Tuple)
        and isinstance(value, ast.Tuple)
    ):
        target_tuple = targets[0]
        if len(target_tuple.elts) == len(value.elts):
            pairs.extend(zip(target_tuple.elts, value.elts))
            return pairs
    for target in targets:
        pairs.append((target, value))
    return pairs


def _target_to_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _enclosing_scope_name(target: ast.AST, tree: ast.AST) -> Optional[str]:
    parents: Dict[ast.AST, Optional[ast.AST]] = {tree: None}
    stack: List[ast.AST] = [tree]
    while stack:
        current = stack.pop()
        for child in ast.iter_child_nodes(current):
            parents[child] = current
            stack.append(child)

    node: Optional[ast.AST] = target
    while node in parents:
        parent = parents[node]
        if isinstance(parent, ast.FunctionDef):
            return parent.name
        node = parent
    return None


class _AssignmentCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.assignment_map: Dict[str, Dict[str, ast.AST]] = defaultdict(dict)
        self.scope_stack: List[str] = ["module"]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope_stack.append(node.name)
        for stmt in node.body:
            self.visit(stmt)
        self.scope_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        scope = self.scope_stack[-1]
        for target_node, value_node in _assignment_pairs(node.targets, node.value):
            target_name = _target_to_name(target_node)
            if target_name is None:
                continue
            self.assignment_map[scope][target_name] = value_node


def main() -> None:
    analyzer = DataFlowAnalyzer("calc.py")
    print(analyzer.run_pipeline(1000))


if __name__ == "__main__":
    main()
