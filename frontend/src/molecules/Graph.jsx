import { ForceGraph2D } from "react-force-graph";
import { forceCollide } from "d3-force-3d";
import PropTypes from "prop-types";
import { useEffect, useRef, useState } from "react";

const applyGraphForces = (graphRef, linkDistance, nodeRadius) => {
  graphRef.current.d3Force("link").iterations(1).distance(linkDistance);

  graphRef.current.d3Force("charge").strength(0).distanceMin(4).distanceMax(4);

  graphRef.current.d3Force(
    "collide",
    forceCollide(nodeRadius).strength(0.5).iterations(1)
  );
  graphRef.current.d3ReheatSimulation();
};

export default function Graph({
  data,
  isDarkMode,
  focusedNode,
  setFocusedNode,
}) {
  const graphRef = useRef(null);
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  const handleResize = () => {
    setDimensions({
      width: window.innerWidth,
      height: window.innerHeight,
    });
    if (graphRef.current) {
      const centerY = window.innerHeight / (data.nodes.length > 50 ? 2 : 4);
      graphRef.current.width = window.innerWidth;
      const zoomFactor = data.nodes.length > 50 ? 0.5 : 1;
      graphRef.current.zoom(zoomFactor, 500);
      setTimeout(() => {
        graphRef.current.centerAt(0, centerY, 2000);
      }, 500);
    }
  };

  useEffect(() => {
    if (graphRef.current && data) {
      const linkDistance = 40; // make it constant for now as radius is that of the highest
      console.log("Change link distance!", linkDistance);
      const largestNodeRadius = Math.max(...data.nodes.map((node) => node.val));
      applyGraphForces(graphRef, linkDistance, largestNodeRadius);
      graphRef.current.width = dimensions.width;
    }
  }, [data, dimensions]);

  useEffect(() => {
    if (focusedNode) {
      const fNode = data.nodes.find((n) => n.name === focusedNode);
      if (fNode) {
        const padding = Math.min(window.innerWidth, window.innerHeight) * 0.2; // Padding, arbitrary, can be dependent on node label size later
        graphRef.current.zoomToFit(
          2000,
          padding,
          (node) => node.name === fNode.name
        );
        graphRef.current.centerAt(fNode.x, fNode.y + 10, 2000); // Centering, arbitrary, adding so that it is not blocked by search bar
      }
      setFocusedNode(null);
    }
  }, [focusedNode, data, setFocusedNode]);

  useEffect(() => {
    window.addEventListener("resize", handleResize);
    window.addEventListener("DOMContentLoaded", handleResize);
    handleResize();

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("DOMContentLoaded", handleResize);
    };
  }, [data]);

  return (
    <section className="font-inter font-semibold w-full fixed top-0">
      {data && (
        <ForceGraph2D
          ref={graphRef}
          graphData={data}
          width={dimensions.width}
          height={dimensions.height}
          // nodeCanvasObject={(node, ctx, globalScale) => {
          //   const label = node.name;
          //   const fontSize = 12 / globalScale;
          //   ctx.font = `${fontSize}px Sans-Serif`;
          //   ctx.fillStyle = isDarkMode ? "white" : "black";
          //   ctx.textAlign = "center";
          //   ctx.textBaseline = "middle";
          //   ctx.fillText(label, node.x, node.y);
          // }}
          backgroundColor={
            isDarkMode ? "rgba(0, 0, 0, 1)" : "rgba(255, 255, 255, 1)"
          }
          linkWidth={3}
          linkColor={(link) => {
            if (link.source.val > 4 && link.target.val > 9) {
              return isDarkMode
                ? "rgba(255, 87, 51, 0.8)"
                : "rgba(255, 87, 51, 0.8)";
            } else if (link.source.val > 6 && link.target.val > 4) {
              return isDarkMode
                ? "rgba(51, 255, 189, 0.8)"
                : "rgba(51, 255, 189, 0.8)";
            } else {
              return isDarkMode
                ? "rgba(51, 85, 255, 0.8)"
                : "rgba(51, 85, 255, 0.8)";
            }
          }}
          nodeColor={
            isDarkMode
              ? (node) =>
                  node.name === focusedNode
                    ? "rgba(80, 10, 10, 1)"
                    : "rgba(80, 10, 10, 1)"
              : (node) =>
                  node.name === focusedNode
                    ? "rgba(80, 10, 10, 1)"
                    : "rgba(80, 10, 10, 1)"
          }
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={1}
        />
      )}
    </section>
  );
}

Graph.propTypes = {
  data: PropTypes.object.isRequired,
  isDarkMode: PropTypes.bool.isRequired,
  focusedNode: PropTypes.string,
  setFocusedNode: PropTypes.func.isRequired,
};
