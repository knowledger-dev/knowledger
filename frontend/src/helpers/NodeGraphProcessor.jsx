// helper function to process and return data from regular query
export const NodeGraphProcessor = (data, user) => {
  class GraphNode {
    constructor(value) {
      this.id = value._id;
      this.content = value.content;
      this.similar_notes = value.similar_notes;
      this.processed_content = value.processed_content;
      this.summary = value.summary;
      this.timestamp = value.timestamp;
      this.edges = [];
    }

    addEdge(node) {
      this.edges.push(node);
    }
  }

  class Graph {
    constructor() {
      this.nodes = [];
    }

    getNode(id) {
      console.log("Getting node with id: ", id);
      console.log("Nodes: ", this.nodes);
      return this.nodes.find((node) => node.id === id);
    }

    addNode(value) {
      const newNode = new GraphNode(value);
      this.nodes.push(newNode);
      return newNode;
    }

    addEdge(node1, node2) {
      node1.addEdge(node2);
      node2.addEdge(node1); // Assuming it's an undirected graph
    }

    printGraph() {
      this.nodes.forEach((node) => {
        const edges = node.edges.map((edge) => edge.value);
        console.log(`${node.value} -> ${edges.join(", ")}`);
      });
    }
  }

  const graph = new Graph();
  const idSet = [];

  console.log("NodeGraphProcessor Received: ", data);

  const exploreNode = async (node) => {
    do {
      for (let i = 0; i < node.similar_notes.length; i++) {
        let similarNote = node.similar_notes[i];

        console.log(i);

        const response = await fetch(
          `https://knowledger.onrender.com/notes/${similarNote}`,
          {
            method: "GET",
            headers: {
              "Content-Type": "application/json",

              Authorization: `Bearer ${user.access_token}`,
            },
          }
        );
        const data = await response.json();
        similarNote = data;
        console.log("NODE", node);
        console.log("SIMILAR NOTE", similarNote);
        if (!idSet.includes(similarNote.id)) {
          console.log("Adding node: ", similarNote);
          graph.addNode(similarNote);
          idSet.concat([similarNote.id]);
        }
        similarNote = graph.getNode(similarNote._id);
        console.log("Adding edge between: ", node, " AND ", similarNote);
        graph.addEdge(node, similarNote);
        // exploreNode(similarNote);
      }
    } while (node.similar_notes.length > 0);
    return;
  };

  data.forEach((node) => {
    exploreNode(node);
  });

  return graph;
};

import PropTypes from "prop-types";

NodeGraphProcessor.propTypes = {
  data: PropTypes.array.isRequired,
};
