"""
Debug node-to-infoset mapping to understand the structure.
"""

import pyspiel
from matrix_cfr import GameTreeConverter

game = pyspiel.load_game("kuhn_poker")
converter = GameTreeConverter(game)
matrix_repr = converter.build_matrices()

print("Infoset to actions mapping:")
print("=" * 80)
for infoset, actions in sorted(matrix_repr.infoset_to_actions.items())[:10]:
    print(f"\nInfoset: {infoset}")
    print(f"  Actions: {actions}")

    # Find nodes for this infoset
    for action in actions:
        if (infoset, action) in matrix_repr.action_index_to_node:
            node_id = matrix_repr.action_index_to_node[(infoset, action)]
            node = matrix_repr.nodes[node_id]
            print(f"  Action {action} → Node {node_id}: "
                  f"depth={node.depth}, player={node.player}, "
                  f"terminal={node.is_terminal}, parent={node.parent_id}")

            # Also check parent
            if node.parent_id >= 0:
                parent = matrix_repr.nodes[node.parent_id]
                print(f"    Parent node {node.parent_id}: "
                      f"depth={parent.depth}, player={parent.player}, "
                      f"infoset={parent.infoset}")

print("\n" + "=" * 80)
print("KEY INSIGHT:")
print("action_index_to_node[(infoset, action)] = CHILD node (after taking action)")
print("We need utility at PARENT node (the decision point)")
print("=" * 80)
