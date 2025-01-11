from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from langgraph.graph import Graph


def save_graph(
    graph: "Graph", filename: str = "agent_practice/newsletter_agent/images/graph.png"
) -> None:
    """Visualize and save the graph.

    Args:
        graph (Graph): The graph to visualize and save.
        filename (str): The filename to save the graph. Default is "images/graph.png"
    """
    try:
        graph_image = graph.get_graph().draw_mermaid_png()
        # Convert bytes to numpy array
        nparr = np.frombuffer(graph_image, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Save image using cv2
        cv2.imwrite(filename, img)
    except Exception as e:
        print(f"Failed to save graph visualization: {e}")
